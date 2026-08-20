//! Pipeline parallel collective communication pattern.
//!
//! Pipeline parallelism divides a model into sequential stages, where each stage
//! processes microbatches and passes activations/gradients to adjacent stages.
//!
//! This module supports GPipe-style scheduling with optional Tensor Parallelism (TP)
//! and Data Parallelism (DP):
//! 1. All microbatches flow forward through all stages
//! 2. All microbatches flow backward through all stages  
//! 3. (Optional) DP AllReduce across DP replicas at the end
//!
//! Worker layout (TP-rank fastest, then DP-replica, then stage/PP slowest):
//! - With P stages, T TP size, and D DP replicas:
//!   - Total workers: P * T * D
//!   - Workers per stage: T * D
//!   - Stage (PP): worker_id / (T * D)
//!   - DP replica: (worker_id % (T * D)) / T
//!   - TP rank: worker_id % T
//!
//! Example: 2 stages, 4 TP, 2 DP (16 workers)
//! - Stage 0 (PP0):
//!   - DP replica 0: workers 0,1,2,3 (TP 0-3)
//!   - DP replica 1: workers 4,5,6,7 (TP 0-3)
//! - Stage 1 (PP1):
//!   - DP replica 0: workers 8,9,10,11 (TP 0-3)
//!   - DP replica 1: workers 12,13,14,15 (TP 0-3)
//!
//! Communication patterns:
//! - PP: Point-to-point from (stage s, dp, tp) to (stage s+1, dp, tp)
//! - DP: AllReduce across DP replicas (same stage, same TP rank)

use crate::simulator::WorkerEvent;
use crate::simulator::ml_worker::FlowKind;
use super::CollectiveEvents;

/// Specification for a single pipeline stage.
#[derive(Debug, Clone)]
pub struct StageSpec {
    /// Forward pass compute time in microseconds
    pub forward_us: u64,
    /// Backward pass compute time in microseconds
    pub backward_us: u64,
    /// Activation tensor size sent to next stage (bytes)
    pub activation_size_bytes: u64,
    /// Gradient tensor size (typically same as activation, sent to previous stage)
    pub gradient_size_bytes: u64,
}

impl Default for StageSpec {
    fn default() -> Self {
        Self {
            forward_us: 50000,
            backward_us: 100000,
            activation_size_bytes: 1_000_000_000,
            gradient_size_bytes: 1_000_000_000,
        }
    }
}

/// Pipeline schedule type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PipelineSchedule {
    /// GPipe: all forward passes, then all backward passes
    #[default]
    GPipe,
    // Future: OneFOneBe, InterleavedOneFOneB
}

/// Configuration for pipeline parallel job generation.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of pipeline stages
    pub num_stages: usize,
    /// Tensor parallel size (GPUs per TP group within each stage)
    pub tp_size: usize,
    /// Number of DP replicas (full pipeline replicas)
    pub dp_replicas: usize,
    /// Number of microbatches per iteration
    pub microbatches: usize,
    /// Pipeline schedule
    pub schedule: PipelineSchedule,
    /// Per-stage specifications (if shorter than num_stages, last entry is repeated)
    pub stages: Vec<StageSpec>,
}

impl PipelineConfig {
    /// Get the stage spec for a given stage index.
    pub fn get_stage_spec(&self, stage: usize) -> &StageSpec {
        if stage < self.stages.len() {
            &self.stages[stage]
        } else if !self.stages.is_empty() {
            self.stages.last().unwrap()
        } else {
            // Should not happen, but provide a default
            panic!("No stage specs provided")
        }
    }
    
    /// Total number of workers: stages * tp_size * dp_replicas
    pub fn num_workers(&self) -> usize {
        self.num_stages * self.tp_size * self.dp_replicas
    }
    
    /// Workers per stage (all TP ranks across all DP replicas)
    pub fn workers_per_stage(&self) -> usize {
        self.tp_size * self.dp_replicas
    }
    
    /// Workers per DP replica (one full pipeline)
    pub fn workers_per_replica(&self) -> usize {
        self.num_stages * self.tp_size
    }
    
    /// Get the stage index (PP) for a worker.
    /// Stage changes slowest in the worker ordering.
    pub fn worker_stage(&self, worker_id: usize) -> usize {
        worker_id / self.workers_per_stage()
    }
    
    /// Get the DP replica index for a worker.
    /// DP changes at medium rate (after TP, before PP).
    pub fn worker_dp_replica(&self, worker_id: usize) -> usize {
        (worker_id % self.workers_per_stage()) / self.tp_size
    }
    
    /// Get the TP rank for a worker (within its stage and DP replica).
    /// TP rank changes fastest in the worker ordering.
    pub fn worker_tp_rank(&self, worker_id: usize) -> usize {
        worker_id % self.tp_size
    }
    
    /// Get the worker ID for a given (dp_replica, stage, tp_rank).
    /// Layout: stage * (tp_size * dp_replicas) + dp_replica * tp_size + tp_rank
    pub fn worker_id(&self, dp_replica: usize, stage: usize, tp_rank: usize) -> usize {
        stage * self.workers_per_stage() + dp_replica * self.tp_size + tp_rank
    }
    
    /// Get all workers in a TP group (same dp_replica, same stage).
    pub fn tp_group(&self, dp_replica: usize, stage: usize) -> Vec<usize> {
        (0..self.tp_size)
            .map(|tp_rank| self.worker_id(dp_replica, stage, tp_rank))
            .collect()
    }
    
    /// Get all workers with the same (stage, tp_rank) across DP replicas.
    pub fn dp_group(&self, stage: usize, tp_rank: usize) -> Vec<usize> {
        (0..self.dp_replicas)
            .map(|dp_replica| self.worker_id(dp_replica, stage, tp_rank))
            .collect()
    }
}

/// Generates pipeline parallel events for all workers using GPipe schedule.
///
/// Supports full 3D parallelism: Pipeline Parallel (PP) + Tensor Parallel (TP) + Data Parallel (DP)
///
/// GPipe processes all microbatches through forward passes first, then all
/// through backward passes. This creates a "bubble" but is simple to implement.
///
/// Per forward/backward pass within a stage:
/// 1. Receive activations/gradients from adjacent stage (point-to-point by TP rank)
/// 2. Compute forward/backward
/// 3. (If TP > 1) TP AllReduce within the stage's TP group
/// 4. Send activations/gradients to adjacent stage
///
/// # Arguments
/// * `config` - Pipeline configuration
/// * `start_event_id` - First event ID to use for generated events
/// * `dependencies` - Event IDs that must complete before the pipeline starts
///
/// # Returns
/// A `CollectiveEvents` struct containing the generated events for each worker.
/// Note: Unlike other collectives, each worker may have DIFFERENT event counts.
pub fn generate_pipeline_gpipe_events(
    config: &PipelineConfig,
    start_event_id: usize,
    dependencies: Vec<usize>,
) -> CollectiveEvents {
    let num_workers = config.num_workers();
    let mut worker_events: Vec<Vec<WorkerEvent>> = vec![vec![]; num_workers];
    let mut completion_event_ids: Vec<Vec<usize>> = vec![vec![]; num_workers];
    
    let mut current_event_id = start_event_id;
    
    // We'll track dependencies per-worker as we build the DAG
    let mut worker_last_events: Vec<Vec<usize>> = vec![dependencies.clone(); num_workers];
    
    // Per-stage, per-dp_replica, per-tp_rank, per-microbatch tracking
    // forward_compute_done[dp][stage][tp][mb] = compute completion event IDs
    // forward_send_done[dp][stage][tp][mb] = send completion event IDs (or compute if last stage)
    let mut forward_compute_done: Vec<Vec<Vec<Vec<Vec<usize>>>>> = 
        vec![vec![vec![vec![vec![]; config.microbatches]; config.tp_size]; config.num_stages]; config.dp_replicas];
    let mut forward_send_done: Vec<Vec<Vec<Vec<Vec<usize>>>>> = 
        vec![vec![vec![vec![vec![]; config.microbatches]; config.tp_size]; config.num_stages]; config.dp_replicas];
    
    // === FORWARD PASS ===
    // New dependency structure: after compute N, both send N and recv N+1 can start in parallel.
    // Compute N+1 waits for both send N and recv N+1 to complete.
    for mb in 0..config.microbatches {
        for stage in 0..config.num_stages {
            let spec = config.get_stage_spec(stage);
            
            for dp in 0..config.dp_replicas {
                for tp in 0..config.tp_size {
                    let worker = config.worker_id(dp, stage, tp);
                    
                    // Dependencies for receive N+1: compute N must be done (allows parallel with send N)
                    let recv_deps = if mb == 0 {
                        dependencies.clone()
                    } else {
                        forward_compute_done[dp][stage][tp][mb - 1].clone()
                    };
                    
                    // Dependencies for compute N+1:
                    // 1. Send N must be done (if mb > 0)
                    // 2. Receive N+1 must be done (if stage > 0)
                    let mut compute_deps = if mb == 0 {
                        dependencies.clone()
                    } else {
                        forward_send_done[dp][stage][tp][mb - 1].clone()
                    };
                    
                    // Receive activation from corresponding TP rank in previous stage
                    if stage > 0 {
                        let src_worker = config.worker_id(dp, stage - 1, tp);
                        
                        let recv_id = current_event_id;
                        current_event_id += 1;
                        
                        // FlowReceive depends on previous mb's compute completion
                        // This allows recv N+1 to run in parallel with send N
                        worker_events[worker].push(WorkerEvent::new_flow_receive_with_kind(
                            recv_id,
                            src_worker,
                            spec.activation_size_bytes,
                            recv_deps,
                            FlowKind::Pipeline,
                        ));
                        
                        // Compute depends on both recv completion and previous send completion
                        compute_deps.push(recv_id);
                    }
                    
                    // Forward compute
                    let fwd_compute_id = current_event_id;
                    current_event_id += 1;
                    
                    worker_events[worker].push(WorkerEvent::new_compute(
                        fwd_compute_id,
                        spec.forward_us,
                        compute_deps,
                    ));
                    
                    // Record compute completion and send activation to next stage
                    let compute_deps_out = vec![fwd_compute_id];
                    
                    // Record compute completion (used for next mb's recv dependency)
                    forward_compute_done[dp][stage][tp][mb] = compute_deps_out.clone();
                    
                    // Send activation to next stage (if not last stage)
                    let send_completion = if stage < config.num_stages - 1 {
                        let dst_worker = config.worker_id(dp, stage + 1, tp);
                        
                        let send_id = current_event_id;
                        current_event_id += 1;
                        
                        worker_events[worker].push(WorkerEvent::new_flow_send_with_kind(
                            send_id,
                            dst_worker,
                            spec.activation_size_bytes,
                            compute_deps_out,
                            FlowKind::Pipeline,
                        ));
                        
                        vec![send_id]
                    } else {
                        compute_deps_out
                    };
                    
                    // Record send completion (used for next mb's compute dependency)
                    forward_send_done[dp][stage][tp][mb] = send_completion;
                }
            }
        }
    }
    
    // === BACKWARD PASS ===
    // GPipe schedule: ALL forward passes must complete before ANY backward pass starts.
    // Same parallel send/recv pattern as forward: after compute N, both send N and recv N+1 
    // can start in parallel. Compute N+1 waits for both send N and recv N+1 to complete.
    let mut backward_compute_done: Vec<Vec<Vec<Vec<Vec<usize>>>>> = 
        vec![vec![vec![vec![vec![]; config.microbatches]; config.tp_size]; config.num_stages]; config.dp_replicas];
    let mut backward_send_done: Vec<Vec<Vec<Vec<Vec<usize>>>>> = 
        vec![vec![vec![vec![vec![]; config.microbatches]; config.tp_size]; config.num_stages]; config.dp_replicas];
    
    let last_fwd_mb = config.microbatches - 1;
    
    for mb in 0..config.microbatches {
        for stage in (0..config.num_stages).rev() {
            let spec = config.get_stage_spec(stage);
            
            for dp in 0..config.dp_replicas {
                for tp in 0..config.tp_size {
                    let worker = config.worker_id(dp, stage, tp);
                    
                    // Dependencies for recv:
                    // - For mb == 0: wait for ALL forward passes (last forward mb's send completion)
                    // - For mb > 0: wait for backward compute N-1 (allows parallel with send N-1)
                    let recv_deps = if mb == 0 {
                        // GPipe: first backward must wait for all forwards to complete
                        forward_send_done[dp][stage][tp][last_fwd_mb].clone()
                    } else {
                        backward_compute_done[dp][stage][tp][mb - 1].clone()
                    };
                    
                    // Dependencies for compute:
                    // - For mb == 0: wait for ALL forward passes (last forward mb's send completion)
                    // - For mb > 0: wait for backward send N-1
                    // - Plus: receive completion (if not last stage)
                    let mut compute_deps = if mb == 0 {
                        // GPipe: first backward must wait for all forwards to complete
                        forward_send_done[dp][stage][tp][last_fwd_mb].clone()
                    } else {
                        backward_send_done[dp][stage][tp][mb - 1].clone()
                    };
                    
                    // Receive gradient from next stage
                    if stage < config.num_stages - 1 {
                        let src_worker = config.worker_id(dp, stage + 1, tp);
                        
                        let recv_id = current_event_id;
                        current_event_id += 1;
                        
                        // FlowReceive depends on previous mb's backward compute completion
                        // This allows recv N+1 to run in parallel with send N
                        worker_events[worker].push(WorkerEvent::new_flow_receive_with_kind(
                            recv_id,
                            src_worker,
                            spec.gradient_size_bytes,
                            recv_deps,
                            FlowKind::Pipeline,
                        ));
                        
                        // Compute depends on both recv completion and previous send completion
                        compute_deps.push(recv_id);
                    }
                    
                    // Backward compute
                    let bwd_compute_id = current_event_id;
                    current_event_id += 1;
                    
                    worker_events[worker].push(WorkerEvent::new_compute(
                        bwd_compute_id,
                        spec.backward_us,
                        compute_deps,
                    ));
                    
                    // Record compute completion and send gradient to previous stage
                    let compute_deps_out = vec![bwd_compute_id];
                    
                    // Record compute completion (used for next mb's recv dependency)
                    backward_compute_done[dp][stage][tp][mb] = compute_deps_out.clone();
                    
                    // Send gradient to previous stage (if not first stage)
                    let send_completion = if stage > 0 {
                        let dst_worker = config.worker_id(dp, stage - 1, tp);
                        
                        let send_id = current_event_id;
                        current_event_id += 1;
                        
                        worker_events[worker].push(WorkerEvent::new_flow_send_with_kind(
                            send_id,
                            dst_worker,
                            spec.gradient_size_bytes,
                            compute_deps_out,
                            FlowKind::Pipeline,
                        ));
                        
                        vec![send_id]
                    } else {
                        compute_deps_out
                    };
                    
                    // Record send completion (used for next mb's compute dependency)
                    backward_send_done[dp][stage][tp][mb] = send_completion;
                }
            }
        }
    }
    
    // Update worker_last_events to point to the last backward event
    let last_mb = config.microbatches - 1;
    for dp in 0..config.dp_replicas {
        for stage in 0..config.num_stages {
            for tp in 0..config.tp_size {
                let worker = config.worker_id(dp, stage, tp);
                if !backward_send_done[dp][stage][tp][last_mb].is_empty() {
                    worker_last_events[worker] = backward_send_done[dp][stage][tp][last_mb].clone();
                }
            }
        }
    }
    
    // Set completion events
    for worker in 0..num_workers {
        completion_event_ids[worker] = worker_last_events[worker].clone();
    }
    
    let events_per_worker = worker_events.iter().map(|e| e.len()).max().unwrap_or(0);
    
    CollectiveEvents {
        worker_events,
        events_per_worker,
        completion_event_ids,
    }
}

/// Generates pipeline parallel events with final DP AllReduce per stage.
///
/// This is the full PP + TP + DP pattern:
/// 1. Run GPipe forward/backward for all microbatches (with optional TP AllReduce per pass)
/// 2. Each (stage, tp_rank) group does AllReduce across DP replicas to sync gradients
///
/// # Arguments
/// * `config` - Pipeline configuration  
/// * `model_shard_size` - Size of model shard per stage for AllReduce (bytes)
/// * `start_event_id` - First event ID to use
/// * `dependencies` - Event IDs that must complete before pipeline starts
pub fn generate_pipeline_with_dp_allreduce(
    config: &PipelineConfig,
    model_shard_size: u64,
    start_event_id: usize,
    dependencies: Vec<usize>,
) -> CollectiveEvents {
    // First generate the pipeline events
    let mut pipeline = generate_pipeline_gpipe_events(config, start_event_id, dependencies);
    
    // If DP replicas > 1, add AllReduce across DP replicas
    if config.dp_replicas <= 1 {
        return pipeline;
    }
    
    // Find the next event ID
    let mut current_event_id = start_event_id + pipeline.worker_events.iter()
        .map(|e| e.len())
        .sum::<usize>();
    
    // AllReduce data size: 2 * (D-1) / D * model_shard_size
    let d = config.dp_replicas as f64;
    let allreduce_size = ((2.0 * (d - 1.0) / d) * model_shard_size as f64) as u64;
    
    // For each (stage, tp_rank) pair, create a ring AllReduce across DP replicas
    for stage in 0..config.num_stages {
        for tp in 0..config.tp_size {
            // Get the DP group: workers with same (stage, tp_rank) across DP replicas
            let dp_group = config.dp_group(stage, tp);
            
            // Create ring topology within this DP group
            for (i, &worker) in dp_group.iter().enumerate() {
                let next_replica = (i + 1) % config.dp_replicas;
                let prev_replica = if i == 0 { config.dp_replicas - 1 } else { i - 1 };
                
                let next_worker = dp_group[next_replica];
                let prev_worker = dp_group[prev_replica];
                
                // Dependencies: this worker's pipeline completion
                let deps = pipeline.completion_event_ids[worker].clone();
                
                // Send to next replica in ring
                let send_id = current_event_id;
                current_event_id += 1;
                
                pipeline.worker_events[worker].push(WorkerEvent::new_flow_send_with_kind(
                    send_id,
                    next_worker,
                    allreduce_size,
                    deps.clone(),
                    FlowKind::Ring,
                ));
                
                // Receive from previous replica in ring
                let recv_id = current_event_id;
                current_event_id += 1;
                
                pipeline.worker_events[worker].push(WorkerEvent::new_flow_receive_with_kind(
                    recv_id,
                    prev_worker,
                    allreduce_size,
                    deps,
                    FlowKind::Ring,
                ));
                
                // Update completion events
                pipeline.completion_event_ids[worker] = vec![send_id, recv_id];
            }
        }
    }
    
    // Update events_per_worker
    pipeline.events_per_worker = pipeline.worker_events.iter()
        .map(|e| e.len())
        .max()
        .unwrap_or(0);
    
    pipeline
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::ml_worker::WorkerEventKind;
    
    /// Create a simple config with tp_size=1 (no TP)
    fn make_simple_config(stages: usize, dp_replicas: usize, microbatches: usize) -> PipelineConfig {
        PipelineConfig {
            num_stages: stages,
            tp_size: 1,
            dp_replicas,
            microbatches,
            schedule: PipelineSchedule::GPipe,
            stages: vec![StageSpec::default()],
        }
    }
    
    /// Create a config with all 3 parallelism types
    fn make_3d_config(stages: usize, tp: usize, dp: usize, microbatches: usize) -> PipelineConfig {
        PipelineConfig {
            num_stages: stages,
            tp_size: tp,
            dp_replicas: dp,
            microbatches,
            schedule: PipelineSchedule::GPipe,
            stages: vec![StageSpec::default()],
        }
    }
    
    #[test]
    fn test_pipeline_worker_layout() {
        // 2 stages, 4 TP, 2 DP = 16 workers
        // New layout: TP fastest, then DP, then PP (stage) slowest
        // Stage 0: DP0 = workers 0-3, DP1 = workers 4-7
        // Stage 1: DP0 = workers 8-11, DP1 = workers 12-15
        let config = make_3d_config(2, 4, 2, 1);
        
        assert_eq!(config.num_workers(), 16);
        assert_eq!(config.workers_per_replica(), 8);
        assert_eq!(config.workers_per_stage(), 8);
        
        // Worker 0: stage=0, dp=0, tp=0
        assert_eq!(config.worker_stage(0), 0);
        assert_eq!(config.worker_dp_replica(0), 0);
        assert_eq!(config.worker_tp_rank(0), 0);
        
        // Worker 5: stage=0, dp=1, tp=1
        assert_eq!(config.worker_stage(5), 0);
        assert_eq!(config.worker_dp_replica(5), 1);
        assert_eq!(config.worker_tp_rank(5), 1);
        
        // Worker 8: stage=1, dp=0, tp=0 (start of second stage)
        assert_eq!(config.worker_stage(8), 1);
        assert_eq!(config.worker_dp_replica(8), 0);
        assert_eq!(config.worker_tp_rank(8), 0);
        
        // Worker 15: stage=1, dp=1, tp=3
        assert_eq!(config.worker_stage(15), 1);
        assert_eq!(config.worker_dp_replica(15), 1);
        assert_eq!(config.worker_tp_rank(15), 3);
        
        // Worker ID lookups: worker_id(dp, stage, tp)
        assert_eq!(config.worker_id(0, 0, 0), 0);   // stage 0, dp 0, tp 0
        assert_eq!(config.worker_id(0, 0, 3), 3);   // stage 0, dp 0, tp 3
        assert_eq!(config.worker_id(1, 0, 0), 4);   // stage 0, dp 1, tp 0
        assert_eq!(config.worker_id(0, 1, 0), 8);   // stage 1, dp 0, tp 0
        assert_eq!(config.worker_id(1, 1, 3), 15);  // stage 1, dp 1, tp 3
        
        // TP group: same dp, same stage
        assert_eq!(config.tp_group(0, 0), vec![0, 1, 2, 3]);    // stage 0, dp 0
        assert_eq!(config.tp_group(1, 0), vec![4, 5, 6, 7]);    // stage 0, dp 1
        assert_eq!(config.tp_group(0, 1), vec![8, 9, 10, 11]);  // stage 1, dp 0
        
        // DP group: same stage, same tp_rank across DP replicas
        assert_eq!(config.dp_group(0, 0), vec![0, 4]);   // stage 0, tp 0 -> dp 0 and dp 1
        assert_eq!(config.dp_group(1, 3), vec![11, 15]); // stage 1, tp 3 -> dp 0 and dp 1
    }
    
    #[test]
    fn test_simple_pipeline_no_tp_no_dp() {
        // 2 stages, 1 TP, 1 DP, 1 microbatch
        let config = make_simple_config(2, 1, 1);
        let result = generate_pipeline_gpipe_events(&config, 0, vec![]);
        
        assert_eq!(result.worker_events.len(), 2);
        
        // Stage 0: forward compute, send activation, receive gradient, backward compute
        let w0 = &result.worker_events[0];
        assert!(w0.len() >= 4, "Stage 0 should have at least 4 events, got {}", w0.len());
        
        // Stage 1: receive activation, forward compute, backward compute, send gradient  
        let w1 = &result.worker_events[1];
        assert!(w1.len() >= 4, "Stage 1 should have at least 4 events, got {}", w1.len());
        
        // Verify event types for stage 0
        assert_eq!(w0[0].kind, WorkerEventKind::Compute); // forward
        assert_eq!(w0[1].kind, WorkerEventKind::FlowSend); // send activation
        assert_eq!(w0[2].kind, WorkerEventKind::FlowReceive); // receive gradient
        assert_eq!(w0[3].kind, WorkerEventKind::Compute); // backward
    }
    
    #[test]
    fn test_pipeline_with_tp() {
        // 2 stages, 2 TP, 1 DP, 1 microbatch
        // TP AllReduce is NOT simulated (absorbed into compute time)
        // PP communication: same TP rank in adjacent stages communicate
        let config = make_3d_config(2, 2, 1, 1);
        let result = generate_pipeline_gpipe_events(&config, 0, vec![]);
        
        assert_eq!(result.worker_events.len(), 4);
        
        // Worker 0 (stage 0, tp 0) should send PP to worker 2 (stage 1, tp 0)
        let w0 = &result.worker_events[0];
        let pp_sends = w0.iter().filter(|e| {
            e.kind == WorkerEventKind::FlowSend && 
            e.flow_send.as_ref().map(|s| s.dst_worker == 2).unwrap_or(false)
        }).count();
        assert_eq!(pp_sends, 1, "Worker 0 should send 1 activation to worker 2 (PP)");
        
        // Worker 1 (stage 0, tp 1) should send PP to worker 3 (stage 1, tp 1)
        let w1 = &result.worker_events[1];
        let pp_sends_w1 = w1.iter().filter(|e| {
            e.kind == WorkerEventKind::FlowSend && 
            e.flow_send.as_ref().map(|s| s.dst_worker == 3).unwrap_or(false)
        }).count();
        assert_eq!(pp_sends_w1, 1, "Worker 1 should send 1 activation to worker 3 (PP)");
        
        // No TP AllReduce (workers 0 and 1 should NOT communicate with each other)
        let tp_sends = w0.iter().filter(|e| {
            e.kind == WorkerEventKind::FlowSend && 
            e.flow_send.as_ref().map(|s| s.dst_worker == 1).unwrap_or(false)
        }).count();
        assert_eq!(tp_sends, 0, "Worker 0 should NOT send to worker 1 (no TP AllReduce)");
    }
    
    #[test]
    fn test_pipeline_with_dp_allreduce() {
        // 2 stages, 1 TP, 2 DP, 1 microbatch
        // New layout: Stage 0 = workers 0,1 (DP 0,1), Stage 1 = workers 2,3 (DP 0,1)
        let config = make_simple_config(2, 2, 1);
        let result = generate_pipeline_with_dp_allreduce(&config, 1_000_000_000, 0, vec![]);
        
        assert_eq!(result.worker_events.len(), 4);
        
        // Each worker should end with DP AllReduce events (send + receive)
        for worker in 0..4 {
            let events = &result.worker_events[worker];
            let last_two: Vec<_> = events.iter().rev().take(2).collect();
            
            let has_send = last_two.iter().any(|e| e.kind == WorkerEventKind::FlowSend);
            let has_recv = last_two.iter().any(|e| e.kind == WorkerEventKind::FlowReceive);
            
            assert!(has_send, "Worker {} should have DP AllReduce send", worker);
            assert!(has_recv, "Worker {} should have DP AllReduce receive", worker);
        }
        
        // Worker 0 (stage=0, dp=0) should DP AllReduce with worker 1 (stage=0, dp=1)
        let w0 = &result.worker_events[0];
        let w0_send = w0.iter().rev().find(|e| e.kind == WorkerEventKind::FlowSend).unwrap();
        assert_eq!(w0_send.flow_send.as_ref().unwrap().dst_worker, 1);
    }
    
    #[test]
    fn test_pipeline_3d_parallelism() {
        // 2 stages, 2 TP, 2 DP, 2 microbatches = 8 workers
        // New layout: TP fastest, then DP, then PP (stage) slowest
        // Stage 0: DP0 = workers 0,1 (TP 0,1), DP1 = workers 2,3 (TP 0,1)
        // Stage 1: DP0 = workers 4,5 (TP 0,1), DP1 = workers 6,7 (TP 0,1)
        let config = make_3d_config(2, 2, 2, 2);
        let result = generate_pipeline_with_dp_allreduce(&config, 1_000_000_000, 0, vec![]);
        
        assert_eq!(result.worker_events.len(), 8);
        
        // Verify all workers have events
        for (worker, events) in result.worker_events.iter().enumerate() {
            assert!(!events.is_empty(), "Worker {} should have events", worker);
        }
        
        // Check PP: worker 0 (stage 0, dp 0, tp 0) sends to worker 4 (stage 1, dp 0, tp 0)
        let w0 = &result.worker_events[0];
        let pp_sends: Vec<_> = w0.iter().filter(|e| {
            e.kind == WorkerEventKind::FlowSend && 
            e.flow_send.as_ref().map(|s| s.dst_worker == 4).unwrap_or(false)
        }).collect();
        assert!(!pp_sends.is_empty(), "Worker 0 should send to worker 4 (PP)");
        
        // TP AllReduce is not simulated - no communication between workers 0 and 1
        let tp_sends: Vec<_> = w0.iter().filter(|e| {
            e.kind == WorkerEventKind::FlowSend && 
            e.flow_send.as_ref().map(|s| s.dst_worker == 1).unwrap_or(false)
        }).collect();
        assert!(tp_sends.is_empty(), "Worker 0 should NOT send to worker 1 (no TP AllReduce)");
        
        // Check DP: worker 0 sends to worker 2 (same stage+tp, different DP)
        let dp_sends: Vec<_> = w0.iter().filter(|e| {
            e.kind == WorkerEventKind::FlowSend && 
            e.flow_send.as_ref().map(|s| s.dst_worker == 2).unwrap_or(false)
        }).collect();
        assert!(!dp_sends.is_empty(), "Worker 0 should send to worker 2 (DP)");
    }
    
    #[test]
    fn test_pipeline_multiple_microbatches() {
        // 2 stages, 1 TP, 1 DP, 4 microbatches
        let config = make_simple_config(2, 1, 4);
        let result = generate_pipeline_gpipe_events(&config, 0, vec![]);
        
        // With 4 microbatches and no TP:
        // Stage 0: 4 * (forward + send) + 4 * (recv + backward) = 16 events
        // Stage 1: 4 * (recv + forward) + 4 * (backward + send) = 16 events
        assert_eq!(result.worker_events[0].len(), 16);
        assert_eq!(result.worker_events[1].len(), 16);
    }
}
