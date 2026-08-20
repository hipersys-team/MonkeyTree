//! Base MonkeyTree System Module for rail-optimized topologies.
//!
//! Monitors pod-level fragmentation and triggers ILP-based migrations
//! when fragmentation exceeds a threshold. Works with any RailTreeRouter.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::time::Instant;

use crate::simulator::ml_simulator::MLContext;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::system::{SystemModule, MigrationPlan};
use crate::rail::{RailTree, RailTreeRouter};

use super::rail_fragmentation::compute_pod_fragmentation;
use super::fragmentation::{SegmentId, JobSegment, print_segment_fragmentation_summary};
use super::ilp::{SegmentILPInput, solve_segment_migration_ilp, compute_segment_migrations, SolveStatus};
use super::system::MonkeyTreeConfig;

/// MonkeyTree system module for rail topologies.
///
/// Fragmentation is computed over pods. The ILP treats each pod as a placement
/// unit (analogous to ToRs in the spine-tree version).
pub struct RailMonkeyTreeSystem<R: RailTreeRouter> {
    config: MonkeyTreeConfig,
    _marker: PhantomData<R>,
}

impl<R: RailTreeRouter> RailMonkeyTreeSystem<R> {
    pub fn new(config: MonkeyTreeConfig) -> Self {
        Self { config, _marker: PhantomData }
    }

    pub fn with_threshold(threshold: usize) -> Self {
        Self::new(MonkeyTreeConfig {
            fragmentation_threshold: threshold,
            block_size: 1,
        })
    }
}

impl<R: RailTreeRouter> Default for RailMonkeyTreeSystem<R> {
    fn default() -> Self { Self::new(MonkeyTreeConfig::default()) }
}

impl<R: RailTreeRouter> std::fmt::Debug for RailMonkeyTreeSystem<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RailMonkeyTreeSystem").field("config", &self.config).finish()
    }
}

impl<R, S, FS> SystemModule<RailTree<R>, S, FS> for RailMonkeyTreeSystem<R>
where
    R: RailTreeRouter,
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_init(&mut self, _ctx: &MLContext, _topo: &RailTree<R>, _scheduler: &mut S, _flow_scheduler: &mut FS) {}

    fn on_job_scheduled(&mut self, _now_us: u64, _ctx: &MLContext, _job_id: JobId, _job: &MLJob, _topo: &RailTree<R>, _scheduler: &mut S, _flow_scheduler: &mut FS) {}

    fn on_job_completed(&mut self, _now_us: u64, _ctx: &MLContext, _job_id: JobId, _job: &MLJob, _topo: &RailTree<R>, _scheduler: &mut S, _flow_scheduler: &mut FS) {}

    fn on_reconfigure(
        &mut self,
        _now_us: u64,
        ctx: &MLContext,
        topo: &RailTree<R>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) -> Option<MigrationPlan> {
        let frag = compute_pod_fragmentation(ctx, topo);

        println!(
            "[RailMonkeyTree] Fragmentation check: max={}, threshold={}, fragmented_segments={}",
            frag.max_fragmentation, self.config.fragmentation_threshold, frag.fragmented_segments.len()
        );

        if frag.max_fragmentation <= self.config.fragmentation_threshold {
            return None;
        }

        print_segment_fragmentation_summary(&frag, ctx.time_us.get());
        println!("[RailMonkeyTree] Fragmentation threshold exceeded! Solving ILP...");

        let placements = ctx.placements.borrow();
        let gpus_per_pod = topo.blocks_per_pod * topo.block_size;
        let num_pods = topo.num_pods;

        let fragmented_segments: Vec<SegmentId> = frag.fragmented_segments.iter().copied().collect();

        let segments: HashMap<SegmentId, JobSegment> = frag.segments.iter()
            .map(|s| (s.id, s.clone()))
            .collect();

        let mut initial_allocation: HashMap<(SegmentId, usize), usize> = HashMap::new();
        for segment in &frag.segments {
            if let Some(worker_hosts) = placements.get(&segment.id.job_id) {
                for &wid in &segment.worker_ids {
                    if let Some(&host) = worker_hosts.get(&wid) {
                        let pod = host / gpus_per_pod;
                        *initial_allocation.entry((segment.id, pod)).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut nonfrag_workers_per_pod = vec![0; num_pods];
        for segment in &frag.segments {
            if !frag.fragmented_segments.contains(&segment.id) {
                if let Some(worker_hosts) = placements.get(&segment.id.job_id) {
                    for &wid in &segment.worker_ids {
                        if let Some(&host) = worker_hosts.get(&wid) {
                            let pod = host / gpus_per_pod;
                            nonfrag_workers_per_pod[pod] += 1;
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
            Err(e) => {
                eprintln!("[RailMonkeyTree] ILP error: {}", e);
                return None;
            }
        };
        let ilp_duration = ilp_start.elapsed();

        println!("[RailMonkeyTree] ILP solve time: {:.3}ms", ilp_duration.as_secs_f64() * 1000.0);

        if solution.status != SolveStatus::Optimal {
            println!("[RailMonkeyTree] ILP status: {:?}", solution.status);
            return None;
        }

        if solution.num_moves == 0 {
            println!("[RailMonkeyTree] No moves needed.");
            return None;
        }

        println!("[RailMonkeyTree] ILP solved: {} moves required", solution.num_moves);

        let placements_map: HashMap<JobId, crate::utils::DHashMap<crate::simulator::ml_worker::WorkerId, usize>> =
            placements.iter().map(|(k, v)| (*k, v.clone())).collect();

        // gpus_per_pod plays the role of hosts_per_leaf for the migration solver
        let migrations = compute_segment_migrations(&ilp_input, &solution, &placements_map, gpus_per_pod);

        if migrations.is_empty() {
            return None;
        }

        let total_workers: usize = migrations.iter().map(|m| m.worker_to_host.len()).sum();
        println!("[RailMonkeyTree] Migration plan: {} jobs, {} workers", migrations.len(), total_workers);

        Some(MigrationPlan { jobs: migrations })
    }
}
