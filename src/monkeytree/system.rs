//! Base MonkeyTree System Module
//!
//! A system module that monitors cluster fragmentation and triggers ILP-based
//! migrations when fragmentation exceeds a threshold.
//!
//! This is generic over the router type and works with any `SpineTreeRouter`,
//! including ECMP where no system-directed routing is needed.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::time::Instant;

use crate::simulator::ml_simulator::MLContext;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::system::{SystemModule, MigrationPlan};
use crate::spine::{SpineTree, SpineTreeRouter};

use super::fragmentation::{
    compute_segment_fragmentation, SegmentId, JobSegment,
    print_segment_fragmentation_summary,
};
use super::ilp::{SegmentILPInput, solve_segment_migration_ilp, compute_segment_migrations, SolveStatus};

/// Configuration for the MonkeyTree system module
#[derive(Debug, Clone)]
pub struct MonkeyTreeConfig {
    /// Maximum fragmented jobs allowed per ToR before triggering migration
    pub fragmentation_threshold: usize,
    /// Block size for allocation (1 = individual workers, 8 = 8-GPU servers)
    /// When > 1, migrations happen at block granularity
    pub block_size: usize,
}

impl Default for MonkeyTreeConfig {
    fn default() -> Self {
        Self {
            fragmentation_threshold: 3,
            block_size: 1, // Default to individual worker allocation
        }
    }
}

impl MonkeyTreeConfig {
    /// Create a configuration for block-based GPU allocation
    pub fn with_gpu_blocks(fragmentation_threshold: usize) -> Self {
        Self {
            fragmentation_threshold,
            block_size: 8, // 8 GPUs per server
        }
    }
}

/// Base MonkeyTree system module - works with any router
///
/// On job events, checks fragmentation and returns migration plans when
/// the threshold is exceeded.
pub struct MonkeyTreeSystem<R: SpineTreeRouter> {
    config: MonkeyTreeConfig,
    _router_marker: PhantomData<R>,
}

impl<R: SpineTreeRouter> MonkeyTreeSystem<R> {
    pub fn new(config: MonkeyTreeConfig) -> Self {
        Self {
            config,
            _router_marker: PhantomData,
        }
    }
    
    pub fn with_threshold(threshold: usize) -> Self {
        Self::new(MonkeyTreeConfig {
            fragmentation_threshold: threshold,
            block_size: 1,
        })
    }
    
    /// Create with GPU block-based allocation (8 GPUs per server)
    pub fn with_gpu_blocks(threshold: usize) -> Self {
        Self::new(MonkeyTreeConfig::with_gpu_blocks(threshold))
    }
}

impl<R: SpineTreeRouter> Default for MonkeyTreeSystem<R> {
    fn default() -> Self {
        Self::new(MonkeyTreeConfig::default())
    }
}

impl<R: SpineTreeRouter> std::fmt::Debug for MonkeyTreeSystem<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MonkeyTreeSystem")
            .field("config", &self.config)
            .finish()
    }
}

impl<R, S, FS> SystemModule<SpineTree<R>, S, FS> for MonkeyTreeSystem<R>
where
    R: SpineTreeRouter,
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_init(
        &mut self,
        _ctx: &MLContext,
        _topo: &SpineTree<R>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // No initialization needed for base MonkeyTree
    }

    fn on_job_scheduled(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        _job: &MLJob,
        _topo: &SpineTree<R>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // Reconfiguration will be triggered automatically by the simulator
    }

    fn on_job_completed(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        _job: &MLJob,
        _topo: &SpineTree<R>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // Reconfiguration will be triggered automatically by the simulator
    }

    fn on_reconfigure(
        &mut self,
        _now_us: u64,
        ctx: &MLContext,
        topo: &SpineTree<R>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) -> Option<MigrationPlan> {
        // 1. Compute current segment-based fragmentation
        // For pipeline jobs, each stage is a separate segment
        let frag = compute_segment_fragmentation(ctx, topo);
        
        println!(
            "[MonkeyTree] Fragmentation check: max={}, threshold={}, fragmented_segments={}",
            frag.max_fragmentation,
            self.config.fragmentation_threshold,
            frag.fragmented_segments.len()
        );
        
        // 2. Check if above threshold
        if frag.max_fragmentation <= self.config.fragmentation_threshold {
            return None;
        }
        
        // Print detailed fragmentation state
        print_segment_fragmentation_summary(&frag, ctx.time_us.get());
        
        println!(
            "[MonkeyTree] Fragmentation threshold exceeded! Solving ILP..."
        );
        
        // 3. Build segment-based ILP input
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
        
        // Non-fragmented workers per ToR (from segments not in fragmented list)
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
        
        // 4. Solve segment-based ILP
        let ilp_start = Instant::now();
        let solution = match solve_segment_migration_ilp(&ilp_input) {
            Ok(sol) => sol,
            Err(e) => {
                eprintln!("[MonkeyTree] ILP error: {}", e);
                return None;
            }
        };
        let ilp_duration = ilp_start.elapsed();
        
        println!("[MonkeyTree] ILP solve time: {:.3}ms", ilp_duration.as_secs_f64() * 1000.0);
        
        if solution.status != SolveStatus::Optimal {
            println!("[MonkeyTree] ILP status: {:?}", solution.status);
            return None;
        }
        
        if solution.num_moves == 0 {
            println!("[MonkeyTree] No moves needed according to ILP.");
            return None;
        }
        
        println!(
            "[MonkeyTree] ILP solved: {} moves required",
            solution.num_moves
        );
        
        // 5. Convert to migration plan
        let placements_map: HashMap<JobId, crate::utils::DHashMap<crate::simulator::ml_worker::WorkerId, usize>> = 
            placements.iter().map(|(k, v)| (*k, v.clone())).collect();
        
        let migrations = compute_segment_migrations(
            &ilp_input,
            &solution,
            &placements_map,
            hosts_per_leaf,
        );
        
        if migrations.is_empty() {
            return None;
        }
        
        let total_workers_moved: usize = migrations.iter()
            .map(|m| m.worker_to_host.len())
            .sum();
        
        println!(
            "[MonkeyTree] Migration plan: {} jobs, {} workers to move",
            migrations.len(),
            total_workers_moved
        );
        
        Some(MigrationPlan { jobs: migrations })
    }
}

