//! MonkeyTree + Crux Combined System Module
//!
//! Combines MonkeyTree's fragmentation monitoring and ILP-based migration
//! with Crux's system-directed routing based on job intensity.

use std::collections::HashMap;
use std::time::Instant;

use petgraph::graph::NodeIndex;

use crate::network::routing::Path;
use crate::network::topology::{LinkId, Topology};
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::ml_worker::WorkerId;
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::system::{SystemModule, MigrationPlan, MIGRATION_FLOW_IDX_BASE};
use crate::spine::{SpineTree, SpineCruxRouter, SpineTreeTopology};

use super::fragmentation::{compute_segment_fragmentation, SegmentId, JobSegment, print_segment_fragmentation_summary};
use super::ilp::{SegmentILPInput, solve_segment_migration_ilp, compute_segment_migrations, SolveStatus};
use super::system::MonkeyTreeConfig;

// -----------------------------------------------------------------------------
// Internal state: derive flows from active jobs (adapted from Crux)
// -----------------------------------------------------------------------------

/// Flow template info - stores worker IDs, not host indices.
/// Host positions are looked up from ctx.placements at route computation time.
#[derive(Debug, Clone)]
struct FlowTemplateSpec {
    job_id: JobId,
    job_flow_idx: usize,
    src_worker_id: WorkerId,
    dst_worker_id: WorkerId,
}

#[derive(Debug, Default, Clone)]
struct JobInfo {
    flows: Vec<FlowTemplateSpec>,
}

#[derive(Debug, Default)]
struct CruxCore {
    jobs: HashMap<JobId, JobInfo>,
}

impl CruxCore {
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

    fn convert_node_path_to_links<T: SpineTreeTopology>(&self, topo: &T, node_path: Vec<NodeIndex>) -> Path {
        let mut link_path = Vec::with_capacity(node_path.len().saturating_sub(1));
        let graph = topo.topology();
        for window in node_path.windows(2) {
            let edge_idx = graph
                .find_edge(window[0], window[1])
                .expect("Path constructed over nonexistent edge");
            let link = graph.edge_weight(edge_idx).unwrap();
            link_path.push(link.id);
        }
        link_path
    }
}

// -----------------------------------------------------------------------------
// MonkeyTreeCrux: Combined system module
// -----------------------------------------------------------------------------

/// MonkeyTree + Crux combined system module
///
/// Combines:
/// - MonkeyTree's fragmentation monitoring and ILP-based migration
/// - Crux's system-directed routing based on job intensity
#[derive(Debug, Default)]
pub struct MonkeyTreeCrux {
    config: MonkeyTreeConfig,
    /// Crux's job tracking and routing state
    crux_core: CruxCore,
}

impl MonkeyTreeCrux {
    pub fn new(config: MonkeyTreeConfig) -> Self {
        Self {
            config,
            crux_core: CruxCore::default(),
        }
    }

    pub fn with_threshold(threshold: usize) -> Self {
        Self::new(MonkeyTreeConfig {
            fragmentation_threshold: threshold,
            block_size: 1,
        })
    }

    // -------------------------------------------------------------------------
    // Crux routing logic
    // -------------------------------------------------------------------------

    fn job_intensity_us(&self, ctx: &MLContext, link_bps: f64, job_id: JobId) -> f64 {
        let jobs = ctx.active_jobs.borrow();
        let info = match jobs.get(&job_id) {
            Some(i) => i,
            None => return 0.0,
        };
        let mut total_compute_us: u128 = 0;
        let mut total_send_bytes: u128 = 0;
        for desc in &info.worker_descriptions {
            for step in &desc.steps {
                total_compute_us = total_compute_us.saturating_add(step.compute_us as u128);
                if let Some(flow) = &step.flow {
                    total_send_bytes = total_send_bytes.saturating_add(flow.size_bytes as u128);
                }
            }
        }
        let compute_us = total_compute_us as f64;
        let data_us = if link_bps > 0.0 {
            (total_send_bytes as f64) * 8_000_000.0 / link_bps
        } else {
            0.0
        };
        compute_us + data_us
    }

    fn path_score(link_load: &HashMap<LinkId, usize>, link_path: &[LinkId]) -> usize {
        link_path
            .iter()
            .map(|lid| *link_load.get(lid).unwrap_or(&0))
            .max()
            .unwrap_or(0)
    }

    fn compute_best_path(
        topo: &SpineTree<SpineCruxRouter>,
        core: &CruxCore,
        src: NodeIndex,
        dst: NodeIndex,
        link_load: &HashMap<LinkId, usize>,
    ) -> Path {
        let src_leaf = topo.get_host_leaf(src);
        let dst_leaf = topo.get_host_leaf(dst);
        if src_leaf == dst_leaf {
            let nodes = vec![src, src_leaf, dst];
            return core.convert_node_path_to_links(topo, nodes);
        }
        let mut best: Option<(usize, Path)> = None;
        for spine_idx in 0..topo.num_spines() {
            let spine = topo.get_spine(spine_idx);
            let nodes = vec![src, src_leaf, spine, dst_leaf, dst];
            let path = core.convert_node_path_to_links(topo, nodes);
            let score = Self::path_score(link_load, &path);
            if best.as_ref().map_or(true, |(s, _)| score < *s) {
                best = Some((score, path));
            }
        }
        best.map(|(_, p)| p).expect("MonkeyTreeCrux: no path found")
    }

    fn recompute_routes(&mut self, ctx: &MLContext, topo: &SpineTree<SpineCruxRouter>) {
        let mut link_load: HashMap<LinkId, usize> = HashMap::new();
        let link_bps = topo.link_bandwidth_bps();
        let placements = ctx.placements.borrow();

        // Collect all flows with current host positions from ctx.placements
        // (job_id, job_flow_idx, src_host, dst_host)
        let mut flows: Vec<(JobId, usize, usize, usize)> = Vec::new();
        for (&jid, info) in self.crux_core.jobs.iter() {
            let job_placements = match placements.get(&jid) {
                Some(p) => p,
                None => continue, // Job no longer active
            };
            
            for f in &info.flows {
                // Look up current host positions from placements
                let src_host = match job_placements.get(&f.src_worker_id) {
                    Some(&h) => h,
                    None => continue, // Worker not found
                };
                let dst_host = match job_placements.get(&f.dst_worker_id) {
                    Some(&h) => h,
                    None => continue, // Worker not found
                };
                flows.push((jid, f.job_flow_idx, src_host, dst_host));
            }
        }

        // Sort by descending job intensity
        let mut job_intensity: HashMap<JobId, f64> = HashMap::new();
        flows.sort_by(|a, b| {
            let ia = *job_intensity
                .entry(a.0)
                .or_insert_with(|| self.job_intensity_us(ctx, link_bps, a.0));
            let ib = *job_intensity
                .entry(b.0)
                .or_insert_with(|| self.job_intensity_us(ctx, link_bps, b.0));
            ib.partial_cmp(&ia)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
        });

        // Install templates on the router
        // Group flows by (src_host, dst_host) so all flows between the same hosts use the same path
        {
            let mut router = topo.router.borrow_mut();
            router.clear_templates();

            // Cache of already-computed paths by (src_host, dst_host)
            let mut path_cache: HashMap<(usize, usize), Path> = HashMap::new();

            for (job_id, job_flow_idx, src_h, dst_h) in flows.into_iter() {
                let path = if let Some(cached) = path_cache.get(&(src_h, dst_h)) {
                    cached.clone()
                } else {
                    let src = topo.get_host_by_index(src_h).expect("invalid src host");
                    let dst = topo.get_host_by_index(dst_h).expect("invalid dst host");
                    let new_path = Self::compute_best_path(topo, &self.crux_core, src, dst, &link_load);
                    for lid in new_path.iter() {
                        *link_load.entry(*lid).or_insert(0) += 1;
                    }
                    path_cache.insert((src_h, dst_h), new_path.clone());
                    new_path
                };
                router.inject_template(job_id, job_flow_idx, path);
            }
        }
    }

    // -------------------------------------------------------------------------
    // MonkeyTree migration logic
    // -------------------------------------------------------------------------

    fn check_and_plan_migration(
        &self,
        ctx: &MLContext,
        topo: &SpineTree<SpineCruxRouter>,
    ) -> Option<MigrationPlan> {
        // Use segment-based fragmentation for pipeline-aware optimization
        let frag = compute_segment_fragmentation(ctx, topo);

        println!(
            "[MonkeyTreeCrux] Fragmentation: max={}, threshold={}, fragmented_segments={}",
            frag.max_fragmentation,
            self.config.fragmentation_threshold,
            frag.fragmented_segments.len()
        );

        if frag.max_fragmentation <= self.config.fragmentation_threshold {
            return None;
        }

        // Print detailed fragmentation state
        print_segment_fragmentation_summary(&frag, ctx.time_us.get());
        
        println!("[MonkeyTreeCrux] Threshold exceeded! Solving ILP...");

        let placements = ctx.placements.borrow();
        let hosts_per_leaf = topo.hosts_per_leaf;
        let num_tors = topo.num_leaves;

        let fragmented_segments: Vec<SegmentId> = frag.fragmented_segments.iter().copied().collect();

        // Build segment map
        let segments: HashMap<SegmentId, JobSegment> = frag.segments.iter()
            .map(|s| (s.id, s.clone()))
            .collect();

        // Initial allocation per segment
        let mut initial_allocation: HashMap<(SegmentId, usize), usize> = HashMap::new();
        for segment in &frag.segments {
            if let Some(worker_hosts) = placements.get(&segment.id.job_id) {
                for &wid in &segment.worker_ids {
                    if let Some(&host) = worker_hosts.get(&wid) {
                        let tor = host / hosts_per_leaf;
                        *initial_allocation.entry((segment.id, tor)).or_insert(0) += 1;
                    }
                }
            }
        }

        // Non-fragmented workers per ToR
        let mut nonfrag_workers_per_tor = vec![0; num_tors];
        for segment in &frag.segments {
            if !frag.fragmented_segments.contains(&segment.id) {
                if let Some(worker_hosts) = placements.get(&segment.id.job_id) {
                    for &wid in &segment.worker_ids {
                        if let Some(&host) = worker_hosts.get(&wid) {
                            let tor = host / hosts_per_leaf;
                            nonfrag_workers_per_tor[tor] += 1;
                        }
                    }
                }
            }
        }

        let ilp_input = SegmentILPInput {
            fragmented_segments,
            segments,
            num_tors,
            initial_allocation,
            tor_capacity: hosts_per_leaf,
            target_lambda: self.config.fragmentation_threshold,
            nonfrag_workers_per_tor,
            block_size: self.config.block_size,
            pod_config: None,
        };

        let ilp_start = Instant::now();
        let solution = match solve_segment_migration_ilp(&ilp_input) {
            Ok(sol) => sol,
            Err(e) => {
                eprintln!("[MonkeyTreeCrux] ILP error: {}", e);
                return None;
            }
        };
        let ilp_duration = ilp_start.elapsed();

        println!("[MonkeyTreeCrux] ILP solve time: {:.3}ms", ilp_duration.as_secs_f64() * 1000.0);

        if solution.status != SolveStatus::Optimal {
            println!("[MonkeyTreeCrux] ILP status: {:?}", solution.status);
            return None;
        }

        if solution.num_moves == 0 {
            println!("[MonkeyTreeCrux] No moves needed.");
            return None;
        }

        println!("[MonkeyTreeCrux] ILP solved: {} moves", solution.num_moves);

        let placements_map: HashMap<JobId, crate::utils::DHashMap<WorkerId, usize>> = 
            placements.iter().map(|(k, v)| (*k, v.clone())).collect();

        let migrations = compute_segment_migrations(&ilp_input, &solution, &placements_map, hosts_per_leaf);

        if migrations.is_empty() {
            return None;
        }

        let total_workers: usize = migrations.iter().map(|m| m.worker_to_host.len()).sum();
        println!(
            "[MonkeyTreeCrux] Migration plan: {} jobs, {} workers",
            migrations.len(),
            total_workers
        );

        // Inject migration flow routes into the Crux router
        // Migration flows go from current host (src) to new host (dst)
        self.inject_migration_routes(&placements, &migrations, topo);

        Some(MigrationPlan { jobs: migrations })
    }

    /// Inject routing templates for migration flows.
    /// Each migration flow for worker W uses job_flow_idx = MIGRATION_FLOW_IDX_BASE + W.
    fn inject_migration_routes(
        &self,
        current_placements: &crate::utils::DHashMap<JobId, crate::utils::DHashMap<WorkerId, usize>>,
        migrations: &[crate::simulator::system::JobMigration],
        topo: &SpineTree<SpineCruxRouter>,
    ) {
        let mut router = topo.router.borrow_mut();
        let link_load: HashMap<LinkId, usize> = HashMap::new(); // Empty for migration routes

        for job_migration in migrations {
            let job_id = job_migration.job_id;
            let current_hosts = match current_placements.get(&job_id) {
                Some(hosts) => hosts,
                None => continue,
            };

            for (&worker_id, &new_host) in job_migration.worker_to_host.iter() {
                let old_host = match current_hosts.get(&worker_id) {
                    Some(&h) => h,
                    None => continue,
                };

                if old_host == new_host {
                    continue; // No migration needed
                }

                // Compute route from old_host to new_host
                let src = topo.get_host_by_index(old_host).expect("invalid src host");
                let dst = topo.get_host_by_index(new_host).expect("invalid dst host");
                let path = Self::compute_best_path(topo, &self.crux_core, src, dst, &link_load);

                // Inject template with migration-specific index
                let migration_flow_idx = MIGRATION_FLOW_IDX_BASE + worker_id;
                router.inject_template(job_id, migration_flow_idx, path);
            }
        }
    }
}

impl<S, FS> SystemModule<SpineTree<SpineCruxRouter>, S, FS> for MonkeyTreeCrux
where
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_init(
        &mut self,
        _ctx: &MLContext,
        _topo: &SpineTree<SpineCruxRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // No initialization needed
    }

    fn on_job_scheduled(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        job: &MLJob,
        _topo: &SpineTree<SpineCruxRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // Record job for Crux routing
        self.crux_core.record_job(job);
    }

    fn on_job_completed(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        job_id: JobId,
        _job: &MLJob,
        _topo: &SpineTree<SpineCruxRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // Remove job from Crux tracking
        self.crux_core.remove_job(job_id);
    }

    fn on_reconfigure(
        &mut self,
        _now_us: u64,
        ctx: &MLContext,
        topo: &SpineTree<SpineCruxRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) -> Option<MigrationPlan> {
        // 1. Always recompute Crux routes (based on current placements)
        self.recompute_routes(ctx, topo);

        // 2. Check fragmentation and plan migration if needed
        self.check_and_plan_migration(ctx, topo)
    }

    fn on_migration_end(
        &mut self,
        _now_us: u64,
        ctx: &MLContext,
        job_id: JobId,
        job: &MLJob,
        topo: &SpineTree<SpineCruxRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // Re-record the job with updated worker IDs after rank reassignment
        self.crux_core.remove_job(job_id);
        self.crux_core.record_job(job);
        
        // Recompute all routes after migration completes
        // (placements have changed, so routes need to reflect new host positions)
        println!("[MonkeyTreeCrux] Recomputing routes after migration");
        self.recompute_routes(ctx, topo);
    }
}

