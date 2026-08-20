//! MonkeyTree + SGLB Combined System Module
//!
//! Combines MonkeyTree's fragmentation monitoring and ILP-based migration
//! with SGLB's load-aware per-flow routing and periodic flow remapping.
//!
//! Unlike MonkeyTree+Perfect or MonkeyTree+Crux, SGLB routing is fully reactive
//! (decisions made at flow-start time), so no centralized route computation is needed.
//! Migration flows are routed automatically by SGLBRouter like any other flow.

use std::collections::HashMap;
use std::time::Instant;

use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::ml_worker::WorkerId;
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::system::{SystemModule, MigrationPlan, TimerId};
use crate::spine::{SpineTree, SGLBRouter, SGLBConfig};

use super::fragmentation::{
    compute_segment_fragmentation, SegmentId, JobSegment,
    print_segment_fragmentation_summary,
};
use super::ilp::{SegmentILPInput, solve_segment_migration_ilp, compute_segment_migrations, SolveStatus};
use super::system::MonkeyTreeConfig;

const SGLB_REMAP_TIMER_ID: TimerId = 1;

/// MonkeyTree + SGLB combined system module.
///
/// Combines:
/// - MonkeyTree's fragmentation monitoring and ILP-based migration
/// - SGLB's load-aware routing with periodic flow remapping
#[derive(Debug)]
pub struct MonkeyTreeSGLB {
    config: MonkeyTreeConfig,
    sglb_config: SGLBConfig,
    timer_started: bool,
}

impl MonkeyTreeSGLB {
    pub fn new(config: MonkeyTreeConfig, sglb_config: SGLBConfig) -> Self {
        Self {
            config,
            sglb_config,
            timer_started: false,
        }
    }

    fn schedule_remap_timer(&self, ctx: &MLContext) {
        if self.sglb_config.remap_interval_us > 0 {
            ctx.schedule_timer(self.sglb_config.remap_interval_us, SGLB_REMAP_TIMER_ID);
        }
    }

    fn check_and_plan_migration(
        &self,
        ctx: &MLContext,
        topo: &SpineTree<SGLBRouter>,
    ) -> Option<MigrationPlan> {
        let frag = compute_segment_fragmentation(ctx, topo);

        println!(
            "[MonkeyTreeSGLB] Fragmentation: max={}, threshold={}, fragmented_segments={}",
            frag.max_fragmentation,
            self.config.fragmentation_threshold,
            frag.fragmented_segments.len()
        );

        if frag.max_fragmentation <= self.config.fragmentation_threshold {
            return None;
        }

        print_segment_fragmentation_summary(&frag, ctx.time_us.get());
        println!("[MonkeyTreeSGLB] Threshold exceeded! Solving ILP...");

        let placements = ctx.placements.borrow();
        let hosts_per_leaf = topo.hosts_per_leaf;
        let num_tors = topo.num_leaves;

        let fragmented_segments: Vec<SegmentId> = frag.fragmented_segments.iter().copied().collect();

        let segments: HashMap<SegmentId, JobSegment> = frag.segments.iter()
            .map(|s| (s.id, s.clone()))
            .collect();

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
                eprintln!("[MonkeyTreeSGLB] ILP error: {}", e);
                return None;
            }
        };
        let ilp_duration = ilp_start.elapsed();

        println!("[MonkeyTreeSGLB] ILP solve time: {:.3}ms", ilp_duration.as_secs_f64() * 1000.0);

        if solution.status != SolveStatus::Optimal {
            println!("[MonkeyTreeSGLB] ILP status: {:?}", solution.status);
            return None;
        }

        if solution.num_moves == 0 {
            println!("[MonkeyTreeSGLB] No moves needed.");
            return None;
        }

        println!("[MonkeyTreeSGLB] ILP solved: {} moves", solution.num_moves);

        let placements_map: HashMap<JobId, crate::utils::DHashMap<WorkerId, usize>> =
            placements.iter().map(|(k, v)| (*k, v.clone())).collect();

        let migrations = compute_segment_migrations(&ilp_input, &solution, &placements_map, hosts_per_leaf);

        if migrations.is_empty() {
            return None;
        }

        let total_workers: usize = migrations.iter().map(|m| m.worker_to_host.len()).sum();
        println!(
            "[MonkeyTreeSGLB] Migration plan: {} jobs, {} workers",
            migrations.len(),
            total_workers
        );

        Some(MigrationPlan { jobs: migrations })
    }
}

impl<S, FS> SystemModule<SpineTree<SGLBRouter>, S, FS> for MonkeyTreeSGLB
where
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_init(
        &mut self,
        ctx: &MLContext,
        _topo: &SpineTree<SGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        println!(
            "[MonkeyTreeSGLB] Initialized: threshold={}, block_size={}, sglb_k={}, remap_interval={}us",
            self.config.fragmentation_threshold,
            self.config.block_size,
            self.sglb_config.k,
            self.sglb_config.remap_interval_us,
        );

        if self.sglb_config.enable_remapping && self.sglb_config.remap_interval_us > 0 {
            self.schedule_remap_timer(ctx);
            self.timer_started = true;
        }
    }

    fn on_job_scheduled(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        _job: &MLJob,
        _topo: &SpineTree<SGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
    }

    fn on_job_completed(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        _job: &MLJob,
        _topo: &SpineTree<SGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
    }

    fn on_reconfigure(
        &mut self,
        _now_us: u64,
        ctx: &MLContext,
        topo: &SpineTree<SGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) -> Option<MigrationPlan> {
        // Trigger immediate SGLB remap if periodic timer is disabled
        if self.sglb_config.enable_remapping && self.sglb_config.remap_interval_us == 0 {
            topo.router.borrow_mut().remap_ineligible_flows(topo);
        }

        self.check_and_plan_migration(ctx, topo)
    }

    fn on_migration_end(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        _job: &MLJob,
        _topo: &SpineTree<SGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // SGLB routing is reactive -- no route recomputation needed.
        // New flows after migration will be routed based on current link load.
    }

    fn on_timer(
        &mut self,
        _now_us: u64,
        ctx: &MLContext,
        timer_id: TimerId,
        topo: &SpineTree<SGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        if timer_id != SGLB_REMAP_TIMER_ID {
            return;
        }

        if self.sglb_config.enable_remapping {
            topo.router.borrow_mut().remap_ineligible_flows(topo);
        }

        self.schedule_remap_timer(ctx);
    }
}
