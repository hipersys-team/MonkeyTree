//! MonkeyTree + Perfect Routing combined system module for rail-optimized topologies.
//!
//! Combines pod-level fragmentation monitoring with edge-coloring-based spine assignment.

use std::collections::HashMap;
use std::time::Instant;

use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::ml_worker::WorkerId;
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::system::{SystemModule, MigrationPlan, MIGRATION_FLOW_IDX_BASE};
use crate::rail::{RailTree, RailPerfectRouter};

use crate::monkeytree::perfect_matching::{compute_edge_coloring, collapse_colors};
use super::rail_fragmentation::compute_pod_fragmentation;
use super::fragmentation::{SegmentId, JobSegment, print_segment_fragmentation_summary};
use super::ilp::{SegmentILPInput, solve_segment_migration_ilp, compute_segment_migrations, SolveStatus};
use super::system::MonkeyTreeConfig;

// ---------------------------------------------------------------------------
// Flow tracking (shared with perfect routing)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FlowTemplateSpec {
    pub job_id: JobId,
    pub job_flow_idx: usize,
    pub src_worker_id: WorkerId,
    pub dst_worker_id: WorkerId,
    pub is_ring_flow: bool,
}

#[derive(Debug, Default, Clone)]
pub struct JobInfo {
    pub flows: Vec<FlowTemplateSpec>,
}

#[derive(Debug, Default)]
pub struct RailPerfectCore {
    pub jobs: HashMap<JobId, JobInfo>,
}

impl RailPerfectCore {
    pub fn record_job(&mut self, job: &MLJob) {
        let mut info = JobInfo { flows: Vec::new() };
        for (&src_worker_id, worker) in job.workers.iter() {
            for ev in &worker.template_events {
                if ev.kind == crate::simulator::ml_worker::WorkerEventKind::FlowSend {
                    let send = ev.flow_send.as_ref().expect("FlowSend missing payload");
                    let flow_idx = *job
                        .send_template_to_flow_idx
                        .get(&(src_worker_id, ev.template_id))
                        .expect("missing send_template_to_flow_idx mapping");
                    info.flows.push(FlowTemplateSpec {
                        job_id: job.id,
                        job_flow_idx: flow_idx,
                        src_worker_id,
                        dst_worker_id: send.dst_worker,
                        is_ring_flow: send.is_ring_flow(),
                    });
                }
            }
        }
        info.flows.sort_by_key(|f| (f.job_id, f.job_flow_idx));
        self.jobs.insert(job.id, info);
    }

    pub fn remove_job(&mut self, job_id: JobId) { self.jobs.remove(&job_id); }
}

// ---------------------------------------------------------------------------
// RailMonkeyTreePerfect
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct RailMonkeyTreePerfect {
    config: MonkeyTreeConfig,
    core: RailPerfectCore,
}

impl RailMonkeyTreePerfect {
    pub fn new(config: MonkeyTreeConfig) -> Self {
        Self { config, core: RailPerfectCore::default() }
    }

    pub fn with_threshold(threshold: usize) -> Self {
        Self::new(MonkeyTreeConfig { fragmentation_threshold: threshold, block_size: 1 })
    }

    fn recompute_routes(&mut self, ctx: &MLContext, topo: &RailTree<RailPerfectRouter>) {
        let placements = ctx.placements.borrow();
        let block_size = topo.block_size;
        let num_spines = topo.num_spines;
        let total_rails = topo.num_pods * block_size;
        let gpus_per_pod = topo.blocks_per_pod * block_size;

        let mut ring_flows: Vec<(usize, usize, (JobId, usize))> = Vec::new();
        let mut other_flows: Vec<(usize, usize, (JobId, usize))> = Vec::new();

        for (&jid, info) in self.core.jobs.iter() {
            let jp = match placements.get(&jid) { Some(p) => p, None => continue };
            for f in &info.flows {
                let src_host = match jp.get(&f.src_worker_id) { Some(&h) => h, None => continue };
                let dst_host = match jp.get(&f.dst_worker_id) { Some(&h) => h, None => continue };
                let src_pod = src_host / gpus_per_pod;
                let dst_pod = dst_host / gpus_per_pod;

                if src_pod != dst_pod {
                    let src_gpu = src_host % block_size;
                    let dst_gpu = dst_host % block_size;
                    let src_rail_global = src_pod * block_size + src_gpu;
                    let dst_rail_global = dst_pod * block_size + dst_gpu;

                    if f.is_ring_flow {
                        ring_flows.push((src_rail_global, dst_rail_global, (jid, f.job_flow_idx)));
                    } else {
                        other_flows.push((src_host, dst_host, (jid, f.job_flow_idx)));
                    }
                }
            }
        }

        let mut router = topo.router.borrow_mut();
        router.clear();

        if ring_flows.is_empty() && other_flows.is_empty() { return; }

        if !ring_flows.is_empty() {
            let edges: Vec<(usize, usize, usize)> = ring_flows.iter()
                .enumerate()
                .map(|(idx, &(src, dst, _))| (src, dst, idx))
                .collect();

            let coloring = compute_edge_coloring(total_rails, edges);
            println!("[RailMonkeyTreePerfect] Ring flows: {} flows, {} colors, {} spines",
                ring_flows.len(), coloring.num_colors, num_spines);

            let final_coloring = if coloring.num_colors > num_spines {
                collapse_colors(&coloring, num_spines)
            } else {
                coloring.edge_to_color.clone()
            };

            for (edge_idx, &(_, _, (job_id, job_flow_idx))) in ring_flows.iter().enumerate() {
                if let Some(&spine_idx) = final_coloring.get(&edge_idx) {
                    router.inject_spine_assignment(job_id, job_flow_idx, spine_idx);
                }
            }
        }

        if !other_flows.is_empty() {
            let mut pair_to_spine: HashMap<(usize, usize), usize> = HashMap::new();
            let mut next_spine = 0;
            for &(src_host, dst_host, _) in other_flows.iter() {
                pair_to_spine.entry((src_host, dst_host)).or_insert_with(|| {
                    let s = next_spine % num_spines;
                    next_spine += 1;
                    s
                });
            }
            for &(src_host, dst_host, (job_id, job_flow_idx)) in other_flows.iter() {
                router.inject_spine_assignment(job_id, job_flow_idx, pair_to_spine[&(src_host, dst_host)]);
            }
        }
    }

    fn check_and_plan_migration(&self, ctx: &MLContext, topo: &RailTree<RailPerfectRouter>) -> Option<MigrationPlan> {
        let frag = compute_pod_fragmentation(ctx, topo);

        println!(
            "[RailMonkeyTreePerfect] Fragmentation: max={}, threshold={}, fragmented_segments={}",
            frag.max_fragmentation, self.config.fragmentation_threshold, frag.fragmented_segments.len()
        );

        if frag.max_fragmentation <= self.config.fragmentation_threshold {
            return None;
        }

        print_segment_fragmentation_summary(&frag, ctx.time_us.get());
        println!("[RailMonkeyTreePerfect] Threshold exceeded! Solving ILP...");

        let placements = ctx.placements.borrow();
        let gpus_per_pod = topo.blocks_per_pod * topo.block_size;
        let num_pods = topo.num_pods;

        let fragmented_segments: Vec<SegmentId> = frag.fragmented_segments.iter().copied().collect();
        let segments: HashMap<SegmentId, JobSegment> = frag.segments.iter().map(|s| (s.id, s.clone())).collect();

        let mut initial_allocation: HashMap<(SegmentId, usize), usize> = HashMap::new();
        for segment in &frag.segments {
            if let Some(wh) = placements.get(&segment.id.job_id) {
                for &wid in &segment.worker_ids {
                    if let Some(&host) = wh.get(&wid) {
                        *initial_allocation.entry((segment.id, host / gpus_per_pod)).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut nonfrag_workers_per_pod = vec![0; num_pods];
        for segment in &frag.segments {
            if !frag.fragmented_segments.contains(&segment.id) {
                if let Some(wh) = placements.get(&segment.id.job_id) {
                    for &wid in &segment.worker_ids {
                        if let Some(&host) = wh.get(&wid) {
                            nonfrag_workers_per_pod[host / gpus_per_pod] += 1;
                        }
                    }
                }
            }
        }

        let ilp_input = SegmentILPInput {
            fragmented_segments,
            segments,
            num_tors: num_pods,
            initial_allocation,
            tor_capacity: gpus_per_pod,
            target_lambda: self.config.fragmentation_threshold,
            nonfrag_workers_per_tor: nonfrag_workers_per_pod,
            block_size: self.config.block_size,
            pod_config: None,
        };

        let ilp_start = Instant::now();
        let solution = match solve_segment_migration_ilp(&ilp_input) {
            Ok(sol) => sol,
            Err(e) => { eprintln!("[RailMonkeyTreePerfect] ILP error: {}", e); return None; }
        };
        println!("[RailMonkeyTreePerfect] ILP solve time: {:.3}ms", ilp_start.elapsed().as_secs_f64() * 1000.0);

        if solution.status != SolveStatus::Optimal { return None; }
        if solution.num_moves == 0 { return None; }

        println!("[RailMonkeyTreePerfect] ILP solved: {} moves", solution.num_moves);

        let placements_map: HashMap<JobId, crate::utils::DHashMap<WorkerId, usize>> =
            placements.iter().map(|(k, v)| (*k, v.clone())).collect();
        let migrations = compute_segment_migrations(&ilp_input, &solution, &placements_map, gpus_per_pod);

        if migrations.is_empty() { return None; }

        self.inject_migration_routes(&placements, &migrations, topo);

        Some(MigrationPlan { jobs: migrations })
    }

    fn inject_migration_routes(
        &self,
        current_placements: &crate::utils::DHashMap<JobId, crate::utils::DHashMap<WorkerId, usize>>,
        migrations: &[crate::simulator::system::JobMigration],
        topo: &RailTree<RailPerfectRouter>,
    ) {
        let mut router = topo.router.borrow_mut();
        let block_size = topo.block_size;
        let num_spines = topo.num_spines;
        let total_rails = topo.num_pods * block_size;
        let gpus_per_pod = topo.blocks_per_pod * block_size;

        let mut migration_edges: Vec<(usize, usize, (JobId, usize))> = Vec::new();

        for job_migration in migrations {
            let job_id = job_migration.job_id;
            let current_hosts = match current_placements.get(&job_id) { Some(h) => h, None => continue };

            for (&worker_id, &new_host) in job_migration.worker_to_host.iter() {
                let old_host = match current_hosts.get(&worker_id) { Some(&h) => h, None => continue };
                if old_host == new_host { continue; }

                let src_pod = old_host / gpus_per_pod;
                let dst_pod = new_host / gpus_per_pod;

                if src_pod != dst_pod {
                    let src_gpu = old_host % block_size;
                    let dst_gpu = new_host % block_size;
                    let src_rail = src_pod * block_size + src_gpu;
                    let dst_rail = dst_pod * block_size + dst_gpu;
                    let migration_flow_idx = MIGRATION_FLOW_IDX_BASE + worker_id;
                    migration_edges.push((src_rail, dst_rail, (job_id, migration_flow_idx)));
                }
            }
        }

        if migration_edges.is_empty() { return; }

        let edges: Vec<(usize, usize, usize)> = migration_edges.iter()
            .enumerate()
            .map(|(idx, &(src, dst, _))| (src, dst, idx))
            .collect();

        let coloring = compute_edge_coloring(total_rails, edges);
        let final_coloring = if coloring.num_colors > num_spines {
            collapse_colors(&coloring, num_spines)
        } else {
            coloring.edge_to_color.clone()
        };

        for (edge_idx, &(_, _, (job_id, mig_flow_idx))) in migration_edges.iter().enumerate() {
            if let Some(&spine_idx) = final_coloring.get(&edge_idx) {
                router.inject_spine_assignment(job_id, mig_flow_idx, spine_idx);
            }
        }
    }
}

impl<S, FS> SystemModule<RailTree<RailPerfectRouter>, S, FS> for RailMonkeyTreePerfect
where S: JobScheduler, FS: FlowScheduler,
{
    fn on_init(&mut self, _ctx: &MLContext, _topo: &RailTree<RailPerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) {}

    fn on_job_scheduled(&mut self, _now_us: u64, _ctx: &MLContext, _job_id: JobId, job: &MLJob, _topo: &RailTree<RailPerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) {
        self.core.record_job(job);
    }

    fn on_job_completed(&mut self, _now_us: u64, _ctx: &MLContext, job_id: JobId, _job: &MLJob, _topo: &RailTree<RailPerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) {
        self.core.remove_job(job_id);
    }

    fn on_reconfigure(&mut self, _now_us: u64, ctx: &MLContext, topo: &RailTree<RailPerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) -> Option<MigrationPlan> {
        self.recompute_routes(ctx, topo);
        self.check_and_plan_migration(ctx, topo)
    }

    fn on_migration_end(&mut self, _now_us: u64, ctx: &MLContext, job_id: JobId, job: &MLJob, topo: &RailTree<RailPerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) {
        self.core.remove_job(job_id);
        self.core.record_job(job);
        self.recompute_routes(ctx, topo);
    }
}
