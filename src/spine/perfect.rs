//! Perfect Routing System Module
//!
//! A minimal system module that provides optimal spine assignment using
//! bipartite edge coloring. This module focuses purely on routing without
//! any migration or fragmentation management.
//!
//! # Routing Strategy
//!
//! - **Ring flows** (AllReduce, StridedRing): Routed via bipartite edge coloring
//!   to minimize spine conflicts. Each color maps to a spine switch.
//! - **All-to-all flows**: Routed via round-robin across available spines.
//!   These flows are dense and don't benefit as much from optimal coloring.

use std::collections::HashMap;

use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::ml_worker::WorkerId;
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::system::SystemModule;
use crate::monkeytree::perfect_matching::{compute_edge_coloring, collapse_colors};
// Re-use the SpinePerfectRouter from monkeytree
pub use crate::monkeytree::perfect::SpinePerfectRouter;

use super::topology::SpineTree;

// -----------------------------------------------------------------------------
// Internal state: derive flows from active jobs
// -----------------------------------------------------------------------------

/// Flow template info - stores worker IDs, not host indices.
#[derive(Debug, Clone)]
struct FlowTemplateSpec {
    job_id: JobId,
    job_flow_idx: usize,
    src_worker_id: WorkerId,
    dst_worker_id: WorkerId,
    /// Whether this is a ring flow (AllReduce, StridedRing) or all-to-all flow.
    is_ring_flow: bool,
}

#[derive(Debug, Default, Clone)]
struct JobInfo {
    flows: Vec<FlowTemplateSpec>,
}

#[derive(Debug, Default)]
struct PerfectCore {
    jobs: HashMap<JobId, JobInfo>,
}

impl PerfectCore {
    fn record_job(&mut self, job: &MLJob) {
        let mut info = JobInfo { flows: Vec::new() };

        for (&src_worker_id, worker) in job.workers.iter() {
            for ev in &worker.template_events {
                if ev.kind == crate::simulator::ml_worker::WorkerEventKind::FlowSend {
                    let send = ev.flow_send.as_ref().expect("FlowSend missing payload");
                    let dst_worker_id = send.dst_worker;
                    let flow_idx = *job
                        .send_template_to_flow_idx
                        .get(&(src_worker_id, ev.template_id))
                        .expect("missing send_template_to_flow_idx mapping");
                    info.flows.push(FlowTemplateSpec {
                        job_id: job.id,
                        job_flow_idx: flow_idx,
                        src_worker_id,
                        dst_worker_id,
                        is_ring_flow: send.is_ring_flow(),
                    });
                }
            }
        }

        info.flows.sort_by_key(|f| (f.job_id, f.job_flow_idx));
        self.jobs.insert(job.id, info);
    }

    fn remove_job(&mut self, job_id: JobId) {
        self.jobs.remove(&job_id);
    }
}

// -----------------------------------------------------------------------------
// PerfectRoutingSystem: Minimal system module for perfect routing
// -----------------------------------------------------------------------------

/// Minimal system module that provides optimal routing via edge coloring.
///
/// This module:
/// - Tracks active jobs and their flow patterns
/// - Computes optimal spine assignments using bipartite edge coloring
/// - Prioritizes ring flows (edge coloring) over all-to-all flows (round-robin)
/// - Does NOT perform migration or fragmentation management
#[derive(Debug, Default)]
pub struct PerfectRoutingSystem {
    core: PerfectCore,
    /// Largest flow count seen so far (for peak-load assignment dump).
    max_flows_seen: usize,
}

impl PerfectRoutingSystem {
    pub fn new() -> Self {
        Self {
            core: PerfectCore::default(),
            max_flows_seen: 0,
        }
    }

    /// Write a single peak-load snapshot of the flow->spine assignment to `path`.
    /// Emits per-spine counts, a per-(src_leaf, spine) matrix, and the full flow list.
    fn write_assignment_dump(
        path: &str,
        num_spines: usize,
        num_tors: usize,
        time_us: u64,
        rows: &[(JobId, usize, usize, usize, usize, bool)],
    ) {
        use std::io::Write;
        let mut per_spine_ring = vec![0usize; num_spines];
        let mut per_spine_other = vec![0usize; num_spines];
        let mut leaf_spine = vec![vec![0usize; num_spines]; num_tors];
        for &(_j, _f, src_leaf, _dl, spine, is_ring) in rows {
            if spine < num_spines {
                if is_ring { per_spine_ring[spine] += 1; } else { per_spine_other[spine] += 1; }
                if src_leaf < num_tors { leaf_spine[src_leaf][spine] += 1; }
            }
        }
        let mut f = match std::fs::File::create(path) {
            Ok(f) => f,
            Err(e) => { eprintln!("[PerfectRouting] dump create failed ({}): {}", path, e); return; }
        };
        let _ = writeln!(f, "# PerfectRouting assignment snapshot (peak load)");
        let _ = writeln!(f, "num_spines={} num_tors={} time_us={} total_flows={}",
                         num_spines, num_tors, time_us, rows.len());
        let _ = writeln!(f, "\n## per-spine flow counts: spine ring other total");
        for s in 0..num_spines {
            let _ = writeln!(f, "spine {} {} {} {}", s, per_spine_ring[s], per_spine_other[s],
                             per_spine_ring[s] + per_spine_other[s]);
        }
        let _ = writeln!(f, "\n## per-(src_leaf,spine) total flow counts (rows=leaf, cols=spine)");
        let mut hdr = String::from("leaf\\sp");
        for s in 0..num_spines { hdr.push_str(&format!(" {:>5}", s)); }
        let _ = writeln!(f, "{}", hdr);
        for l in 0..num_tors {
            let mut line = format!("{:>6}", l);
            for s in 0..num_spines { line.push_str(&format!(" {:>5}", leaf_spine[l][s])); }
            let _ = writeln!(f, "{}", line);
        }
        let _ = writeln!(f, "\n## full per-flow assignment: job flow src_leaf dst_leaf spine is_ring");
        for &(j, fl, sl, dl, sp, ir) in rows {
            let _ = writeln!(f, "{} {} {} {} {} {}", j, fl, sl, dl, sp, if ir {1} else {0});
        }
    }

    /// Recompute routes for all active jobs using edge coloring for ring flows
    /// and round-robin for all-to-all flows.
    fn recompute_routes(&mut self, ctx: &MLContext, topo: &SpineTree<SpinePerfectRouter>) {
        let placements = ctx.placements.borrow();
        let hosts_per_leaf = topo.hosts_per_leaf;
        let num_spines = topo.num_spines;
        let num_tors = topo.num_leaves;

        // Separate ring flows (priority: bipartite edge coloring) from 
        // other flows (pipeline activations, all-to-all)
        // Ring flows use (src_tor, dst_tor) for edge coloring
        // Other flows use (src_host, dst_host) for consistent routing per host pair
        let mut ring_flows: Vec<(usize, usize, (JobId, usize))> = Vec::new();
        let mut other_flows: Vec<(usize, usize, (JobId, usize))> = Vec::new();

        // Optional peak-load assignment dump (enabled via PERFECT_ASSIGN_DUMP).
        let dump_path = std::env::var("PERFECT_ASSIGN_DUMP").ok();
        // (job, flow_idx, src_leaf, dst_leaf, spine, is_ring)
        let mut dump_rows: Vec<(JobId, usize, usize, usize, usize, bool)> = Vec::new();

        for (&jid, info) in self.core.jobs.iter() {
            let job_placements = match placements.get(&jid) {
                Some(p) => p,
                None => continue,
            };

            for f in &info.flows {
                let src_host = match job_placements.get(&f.src_worker_id) {
                    Some(&h) => h,
                    None => continue,
                };
                let dst_host = match job_placements.get(&f.dst_worker_id) {
                    Some(&h) => h,
                    None => continue,
                };

                let src_tor = src_host / hosts_per_leaf;
                let dst_tor = dst_host / hosts_per_leaf;

                // Only include cross-ToR flows
                if src_tor != dst_tor {
                    if f.is_ring_flow {
                        // Ring flows: use ToR indices for edge coloring
                        ring_flows.push((src_tor, dst_tor, (jid, f.job_flow_idx)));
                    } else {
                        // Other flows: use host indices for per-host-pair routing
                        other_flows.push((src_host, dst_host, (jid, f.job_flow_idx)));
                    }
                }
            }
        }

        let mut router = topo.router.borrow_mut();
        router.clear();

        if ring_flows.is_empty() && other_flows.is_empty() {
            return;
        }

        // 1. Route ring flows using bipartite edge coloring
        if !ring_flows.is_empty() {
            let edges: Vec<(usize, usize, usize)> = ring_flows
                .iter()
                .enumerate()
                .map(|(idx, &(src, dst, _))| (src, dst, idx))
                .collect();

            let coloring = compute_edge_coloring(num_tors, edges);

            println!(
                "[PerfectRouting] Ring flows: {} flows, {} colors needed, {} spines available",
                ring_flows.len(),
                coloring.num_colors,
                num_spines
            );

            // Collapse colors if we have more than num_spines
            let final_coloring = if coloring.num_colors > num_spines {
                println!(
                    "[PerfectRouting] Collapsing {} colors into {} spines",
                    coloring.num_colors,
                    num_spines
                );
                collapse_colors(&coloring, num_spines)
            } else {
                coloring.edge_to_color.clone()
            };

            // Install spine assignments for ring flows
            for (edge_idx, &(src_tor, dst_tor, (job_id, job_flow_idx))) in ring_flows.iter().enumerate() {
                if let Some(&spine_idx) = final_coloring.get(&edge_idx) {
                    router.inject_spine_assignment(job_id, job_flow_idx, spine_idx);
                    if dump_path.is_some() {
                        dump_rows.push((job_id, job_flow_idx, src_tor, dst_tor, spine_idx, true));
                    }
                }
            }
        }

        // 2. Route other flows (all-to-all, pipeline activations) by (src_host, dst_host)
        // All flows with the same source and destination host get the same spine
        if !other_flows.is_empty() {
            // Group flows by (src_host, dst_host) and assign each unique pair to a spine.
            // Per-source-leaf round-robin: each leaf spreads its outgoing all-to-all
            // flows evenly across ALL spines, starting from a staggered (per-leaf)
            // offset so leaves don't all pile onto spine 0. A single global counter
            // (next_spine % num_spines) aliases against the strided-ring flow pattern
            // at certain spine counts, creating per-leaf hotspots; keying the counter
            // on the source leaf guarantees even spread regardless of num_spines.
            let mut pair_to_spine: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
            let mut leaf_counter: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            
            for &(src_host, dst_host, _) in other_flows.iter() {
                pair_to_spine.entry((src_host, dst_host)).or_insert_with(|| {
                    let src_leaf = src_host / hosts_per_leaf;
                    let cnt = leaf_counter.entry(src_leaf).or_insert(0);
                    let spine = (src_leaf + *cnt) % num_spines;
                    *cnt += 1;
                    spine
                });
            }
            
            println!(
                "[PerfectRouting] Other flows: {} flows, {} unique host pairs, {} spines",
                other_flows.len(),
                pair_to_spine.len(),
                num_spines
            );

            // Assign each flow based on its (src_host, dst_host) pair
            for &(src_host, dst_host, (job_id, job_flow_idx)) in other_flows.iter() {
                let spine_idx = pair_to_spine[&(src_host, dst_host)];
                router.inject_spine_assignment(job_id, job_flow_idx, spine_idx);
                if dump_path.is_some() {
                    dump_rows.push((job_id, job_flow_idx, src_host / hosts_per_leaf,
                                    dst_host / hosts_per_leaf, spine_idx, false));
                }
            }
        }

        // Peak-load snapshot: only write when this recompute has the most flows so far.
        if let Some(path) = dump_path {
            if dump_rows.len() > self.max_flows_seen {
                self.max_flows_seen = dump_rows.len();
                drop(router);
                Self::write_assignment_dump(&path, num_spines, num_tors, ctx.time_us.get(), &dump_rows);
            }
        }
    }
}

impl<S, FS> SystemModule<SpineTree<SpinePerfectRouter>, S, FS> for PerfectRoutingSystem
where
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_init(
        &mut self,
        _ctx: &MLContext,
        _topo: &SpineTree<SpinePerfectRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // No initialization needed
    }

    fn on_job_scheduled(
        &mut self,
        _now_us: u64,
        ctx: &MLContext,
        _job_id: JobId,
        job: &MLJob,
        topo: &SpineTree<SpinePerfectRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // Optional per-job placement footprint log (enabled via PERFECT_PLACE_LOG).
        if std::env::var("PERFECT_PLACE_LOG").is_ok() {
            let placements = ctx.placements.borrow();
            if let Some(p) = placements.get(&job.id) {
                let hpl = topo.hosts_per_leaf;
                let mut per_tor: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
                for (_w, hh) in p.iter() {
                    *per_tor.entry(*hh / hpl).or_insert(0) += 1;
                }
                let fp: Vec<String> = per_tor.iter().map(|(t, c)| format!("{}:{}", t, c)).collect();
                println!("JOBPLACE jid={} nw={} ntors={} tors=[{}]",
                         job.id, p.len(), per_tor.len(), fp.join(","));
            }
        }

        // Record job for routing and recompute routes
        self.core.record_job(job);
        self.recompute_routes(ctx, topo);
    }

    fn on_job_completed(
        &mut self,
        _now_us: u64,
        ctx: &MLContext,
        job_id: JobId,
        _job: &MLJob,
        topo: &SpineTree<SpinePerfectRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // Remove job and recompute routes
        self.core.remove_job(job_id);
        self.recompute_routes(ctx, topo);
    }

    fn on_reconfigure(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _topo: &SpineTree<SpinePerfectRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) -> Option<crate::simulator::system::MigrationPlan> {
        // No migration support - just return None
        None
    }

    fn on_migration_end(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        _job: &MLJob,
        _topo: &SpineTree<SpinePerfectRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // No migration support
    }
}
