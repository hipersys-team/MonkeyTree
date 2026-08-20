//! High-level job builder using collective operations.
//!
//! This module provides `CollectiveJobBuilder`, a fluent API for constructing
//! ML jobs using high-level phases (compute, collectives) rather than
//! manually specifying individual send/receive events.

use crate::simulator::{MLJob, WorkerEvent};
use crate::simulator::ml_job::{JobId, MLJobBuilder};

use super::{CollectiveOp, JobPhase};
use super::alltoall::generate_alltoall_events;
use super::all_reduce::generate_allreduce_events;
use super::all_to_all_ring::generate_all_to_all_ring_events;
use super::strided_ring::generate_strided_ring_events;
use super::pipeline::{generate_pipeline_gpipe_events, generate_pipeline_with_dp_allreduce};

/// A builder for constructing ML jobs using high-level collective operations.
///
/// This builder allows you to specify a job as a sequence of phases (compute,
/// collective operations) and automatically generates the underlying worker
/// event DAGs.
///
/// # Example
///
/// ```rust,ignore
/// use monkeytree::collectives::{CollectiveJobBuilder, CollectiveOp};
///
/// let job = CollectiveJobBuilder::new(0, 0, 4, 100)
///     .with_name("Training iteration".to_string())
///     .add_compute(500)  // Forward/backward pass
///     .add_collective(CollectiveOp::AllReduce { model_size: 5_000_000_000 })
///     .build();
/// ```
pub struct CollectiveJobBuilder {
    job_id: JobId,
    submit_time_us: u64,
    num_workers: usize,
    total_iterations: usize,
    name: Option<String>,
    phases: Vec<JobPhase>,
    /// Pipeline stage info (set when a pipeline collective is added)
    pipeline_stages: Option<crate::simulator::ml_job::PipelineStageInfo>,
}

impl CollectiveJobBuilder {
    /// Creates a new collective job builder.
    ///
    /// # Arguments
    /// * `job_id` - Unique identifier for this job
    /// * `submit_time_us` - Time when the job is submitted
    /// * `num_workers` - Number of workers in the job
    /// * `total_iterations` - Number of iterations to run
    pub fn new(job_id: JobId, submit_time_us: u64, num_workers: usize, total_iterations: usize) -> Self {
        Self {
            job_id,
            submit_time_us,
            num_workers,
            total_iterations,
            name: None,
            phases: Vec::new(),
            pipeline_stages: None,
        }
    }
    
    /// Sets the name of the job.
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
    
    /// Adds a compute phase where all workers compute for the given duration.
    ///
    /// # Arguments
    /// * `duration_us` - Duration of the compute phase in milliseconds
    pub fn add_compute(mut self, duration_us: u64) -> Self {
        self.phases.push(JobPhase::Compute { duration_us });
        self
    }
    
    /// Adds a collective communication operation.
    ///
    /// # Arguments
    /// * `op` - The collective operation to perform
    pub fn add_collective(mut self, op: CollectiveOp) -> Self {
        // If this is a pipeline op, extract the pipeline stage info
        if let CollectiveOp::Pipeline { ref config, .. } = op {
            self.pipeline_stages = Some(crate::simulator::ml_job::PipelineStageInfo {
                num_stages: config.num_stages,
                workers_per_stage: config.workers_per_stage(),
                // Each stage has tp_size independent DP AllReduce rings (one per TP rank)
                rings_per_stage: config.tp_size,
            });
        }
        self.phases.push(JobPhase::Collective(op));
        self
    }
    
    /// Builds the final MLJob by expanding all phases into worker events.
    pub fn build(self) -> MLJob {
        // Generate events for each worker
        let worker_events = self.generate_all_worker_events();
        
        // Build the job using the standard MLJobBuilder
        let mut builder = MLJobBuilder::new(
            self.job_id,
            self.submit_time_us,
            self.num_workers,
            self.total_iterations,
        );
        
        if let Some(name) = self.name {
            builder = builder.with_name(name);
        }
        
        // Add each worker with their generated events
        for (worker_id, events) in worker_events.into_iter().enumerate() {
            builder = builder.add_worker_with_events(worker_id, events);
        }
        
        let mut job = builder.build();
        
        // Apply pipeline stage info if this is a pipeline job
        if let Some(pipeline_stages) = self.pipeline_stages {
            job = job.with_pipeline_stages(pipeline_stages);
        }
        
        job
    }
    
    /// Generates events for all workers by expanding phases.
    fn generate_all_worker_events(&self) -> Vec<Vec<WorkerEvent>> {
        let mut worker_events: Vec<Vec<WorkerEvent>> = vec![vec![]; self.num_workers];
        let mut next_event_id = 0usize;
        // Track the last event IDs per worker that the next phase depends on
        let mut last_phase_events: Vec<Vec<usize>> = vec![vec![]; self.num_workers];
        
        for phase in &self.phases {
            match phase {
                JobPhase::Compute { duration_us } => {
                    // Add a compute event to each worker, depending on previous phase
                    for worker_id in 0..self.num_workers {
                        let event_id = next_event_id;
                        next_event_id += 1;
                        
                        let deps = last_phase_events[worker_id].clone();
                        let event = WorkerEvent::new_compute(event_id, *duration_us, deps);
                        worker_events[worker_id].push(event);
                        
                        // This compute event is now the dependency for the next phase
                        last_phase_events[worker_id] = vec![event_id];
                    }
                }
                
                JobPhase::Collective(op) => {
                    match op {
                        CollectiveOp::AllReduce { model_size } => {
                            self.expand_collective(
                                |num_workers, start_id, _deps| {
                                    generate_allreduce_events(num_workers, *model_size, start_id, vec![])
                                },
                                &mut worker_events,
                                &mut next_event_id,
                                &mut last_phase_events,
                            );
                        }
                        CollectiveOp::AllToAll { total_data_size } => {
                            self.expand_collective(
                                |num_workers, start_id, _deps| {
                                    // Per-pair size = total / (n * (n-1))
                                    let per_pair_size = if num_workers > 1 {
                                        *total_data_size / (num_workers * (num_workers - 1)) as u64
                                    } else {
                                        0
                                    };
                                    generate_alltoall_events(num_workers, per_pair_size, start_id, vec![])
                                },
                                &mut worker_events,
                                &mut next_event_id,
                                &mut last_phase_events,
                            );
                        }
                        CollectiveOp::AllToAllRing { total_data_size } => {
                            self.expand_collective(
                                |num_workers, start_id, _deps| {
                                    // Per-pair size = total / (n * (n-1))
                                    let per_pair_size = if num_workers > 1 {
                                        *total_data_size / (num_workers * (num_workers - 1)) as u64
                                    } else {
                                        0
                                    };
                                    generate_all_to_all_ring_events(num_workers, per_pair_size, start_id, vec![])
                                },
                                &mut worker_events,
                                &mut next_event_id,
                                &mut last_phase_events,
                            );
                        }
                        CollectiveOp::StridedRing { stride, model_size } => {
                            let stride_val = *stride;
                            let model_size_val = *model_size;
                            self.expand_collective(
                                |num_workers, start_id, _deps| {
                                    generate_strided_ring_events(num_workers, stride_val, model_size_val, start_id, vec![])
                                },
                                &mut worker_events,
                                &mut next_event_id,
                                &mut last_phase_events,
                            );
                        }
                        CollectiveOp::Pipeline { config, model_shard_size } => {
                            // Pipeline is special: it generates different events per worker
                            // and the number of workers is determined by the config
                            let config = config.clone();
                            let model_shard = *model_shard_size;
                            self.expand_pipeline(
                                &config,
                                model_shard,
                                &mut worker_events,
                                &mut next_event_id,
                                &mut last_phase_events,
                            );
                        }
                    }
                }
            }
        }
        
        worker_events
    }
    
    /// Generic method to expand any collective into worker events.
    fn expand_collective<F>(
        &self,
        generate_fn: F,
        worker_events: &mut Vec<Vec<WorkerEvent>>,
        next_event_id: &mut usize,
        last_phase_events: &mut Vec<Vec<usize>>,
    )
    where
        F: FnOnce(usize, usize, Vec<usize>) -> super::CollectiveEvents,
    {
        // Generate events for all workers using the provided generator
        let collective = generate_fn(self.num_workers, *next_event_id, vec![]);
        
        // Update the next event ID counter
        *next_event_id += collective.events_per_worker * self.num_workers;
        
        // Add events to each worker with proper dependencies
        for worker_id in 0..self.num_workers {
            let phase_deps = last_phase_events[worker_id].clone();
            
            // Add the collective events, merging prior phase deps with internal deps.
            // Events that originally had empty dependencies get the phase deps.
            // Events with internal dependencies (e.g., round 2 depends on round 1)
            // keep those but also need phase deps for the first round.
            for mut event in collective.worker_events[worker_id].clone() {
                if event.dependencies.is_empty() {
                    // First round events: depend on the prior phase
                    event.dependencies = phase_deps.clone();
                }
                // Events with existing dependencies keep them (inter-round deps)
                // They were set up by the collective generator
                worker_events[worker_id].push(event);
            }
            
            // The collective's completion events become dependencies for next phase
            last_phase_events[worker_id] = collective.completion_event_ids[worker_id].clone();
        }
    }
    
    /// Expands a pipeline parallel operation into worker events.
    /// 
    /// Pipeline is special because:
    /// 1. Different workers have different event sequences (heterogeneous DAGs)
    /// 2. The number of workers is determined by the pipeline config
    fn expand_pipeline(
        &self,
        config: &super::pipeline::PipelineConfig,
        model_shard_size: u64,
        worker_events: &mut Vec<Vec<WorkerEvent>>,
        next_event_id: &mut usize,
        last_phase_events: &mut Vec<Vec<usize>>,
    ) {
        // Verify worker count matches
        assert_eq!(
            config.num_workers(),
            self.num_workers,
            "Pipeline config has {} workers but job has {}",
            config.num_workers(),
            self.num_workers
        );
        
        // Generate pipeline events
        let collective = if model_shard_size > 0 {
            generate_pipeline_with_dp_allreduce(config, model_shard_size, *next_event_id, vec![])
        } else {
            generate_pipeline_gpipe_events(config, *next_event_id, vec![])
        };
        
        // Update the next event ID counter (sum of all events across workers)
        let total_events: usize = collective.worker_events.iter().map(|e| e.len()).sum();
        *next_event_id += total_events;
        
        // Add events to each worker with proper dependencies
        for worker_id in 0..self.num_workers {
            let phase_deps = last_phase_events[worker_id].clone();
            
            for mut event in collective.worker_events[worker_id].clone() {
                if event.dependencies.is_empty() {
                    event.dependencies = phase_deps.clone();
                }
                worker_events[worker_id].push(event);
            }
            
            last_phase_events[worker_id] = collective.completion_event_ids[worker_id].clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::ml_worker::WorkerEventKind;
    
    #[test]
    fn test_simple_compute_only() {
        let job = CollectiveJobBuilder::new(0, 0, 2, 10)
            .with_name("Test job".to_string())
            .add_compute(500)
            .add_compute(300)
            .build();
        
        assert_eq!(job.num_workers, 2);
        assert_eq!(job.total_iterations, 10);
        assert_eq!(job.name, Some("Test job".to_string()));
        
        // Each worker should have 2 compute events
        for (_, worker) in &job.workers {
            assert_eq!(worker.template_events.len(), 2);
            assert!(worker.template_events[0].kind == WorkerEventKind::Compute);
            assert!(worker.template_events[1].kind == WorkerEventKind::Compute);
            
            // Second compute depends on first
            assert_eq!(worker.template_events[1].dependencies, vec![worker.template_events[0].id]);
        }
    }
    
    #[test]
    fn test_compute_allreduce_compute() {
        // Test with 4 workers, model_size = 5GB
        // Expected data per worker: 2 * (4-1)/4 * 5GB = 7.5GB
        let model_size = 5_000_000_000u64;
        let job = CollectiveJobBuilder::new(0, 0, 4, 10)
            .add_compute(800)
            .add_collective(CollectiveOp::AllReduce { model_size })
            .build();
        
        let expected_data = ((2.0 * 3.0 / 4.0) * model_size as f64) as u64;
        
        // Each worker should have:
        // - 1 compute event
        // - 2 collective events (1 send + 1 receive)
        // Total: 3 events per worker
        for (worker_id, worker) in &job.workers {
            assert_eq!(worker.template_events.len(), 3);
            
            // First event is compute
            assert!(worker.template_events[0].kind == WorkerEventKind::Compute);
            
            // Second event is send to next worker
            let send = &worker.template_events[1];
            assert!(send.kind == WorkerEventKind::FlowSend);
            let expected_dst = (*worker_id + 1) % 4;
            assert_eq!(send.flow_send.as_ref().unwrap().dst_worker, expected_dst);
            assert_eq!(send.flow_send.as_ref().unwrap().size_bytes, expected_data);
            
            // Third event is receive from previous worker
            let recv = &worker.template_events[2];
            assert!(recv.kind == WorkerEventKind::FlowReceive);
            let expected_src = if *worker_id == 0 { 3 } else { *worker_id - 1 };
            assert_eq!(recv.flow_receive.as_ref().unwrap().src_worker, expected_src);
            
            // Both collective events depend on the compute
            assert!(send.dependencies.contains(&worker.template_events[0].id));
            assert!(recv.dependencies.contains(&worker.template_events[0].id));
        }
    }
    
    #[test]
    fn test_compute_alltoall_compute() {
        // 3 workers: 3 * 2 = 6 flows, total = 6_000_000 -> per_pair = 1_000_000
        let job = CollectiveJobBuilder::new(0, 0, 3, 5)
            .add_compute(100)
            .add_collective(CollectiveOp::AllToAll { total_data_size: 6_000_000 })
            .add_compute(50)
            .build();
        
        // Each worker should have:
        // - 1 compute event
        // - 4 collective events (2 sends + 2 receives for 3 workers)
        // - 1 compute event
        // Total: 6 events per worker
        for (_, worker) in &job.workers {
            assert_eq!(worker.template_events.len(), 6);
            
            // First event is compute
            assert!(worker.template_events[0].kind == WorkerEventKind::Compute);
            
            // Middle events are sends/receives
            let mut sends = 0;
            let mut receives = 0;
            for i in 1..5 {
                match worker.template_events[i].kind {
                    WorkerEventKind::FlowSend => sends += 1,
                    WorkerEventKind::FlowReceive => receives += 1,
                    _ => panic!("Expected send or receive"),
                }
                // All collective events depend on the first compute
                assert!(worker.template_events[i].dependencies.contains(&worker.template_events[0].id));
            }
            assert_eq!(sends, 2);
            assert_eq!(receives, 2);
            
            // Last event is compute and depends on all collective events
            assert!(worker.template_events[5].kind == WorkerEventKind::Compute);
            assert_eq!(worker.template_events[5].dependencies.len(), 4);
        }
    }
    
    #[test]
    fn test_alltoall_two_workers() {
        // 2 workers: 2 * 1 = 2 flows, total = 10GB -> per_pair = 5GB
        let job = CollectiveJobBuilder::new(0, 0, 2, 1)
            .add_collective(CollectiveOp::AllToAll { total_data_size: 10_000_000_000 })
            .build();
        
        // Each worker: 1 send + 1 receive = 2 events
        for (worker_id, worker) in &job.workers {
            assert_eq!(worker.template_events.len(), 2);
            
            let other_worker = if *worker_id == 0 { 1 } else { 0 };
            
            // Check send destination
            let send = &worker.template_events[0];
            assert!(send.kind == WorkerEventKind::FlowSend);
            assert_eq!(send.flow_send.as_ref().unwrap().dst_worker, other_worker);
            // per_pair = 10GB / 2 = 5GB
            assert_eq!(send.flow_send.as_ref().unwrap().size_bytes, 5_000_000_000);
            
            // Check receive source
            let recv = &worker.template_events[1];
            assert!(recv.kind == WorkerEventKind::FlowReceive);
            assert_eq!(recv.flow_receive.as_ref().unwrap().src_worker, other_worker);
        }
    }
    
    #[test]
    fn test_allreduce_two_workers() {
        // 2 workers: data = 2 * (2-1)/2 * model_size = model_size
        let model_size = 1_000_000_000u64;
        let job = CollectiveJobBuilder::new(0, 0, 2, 1)
            .add_collective(CollectiveOp::AllReduce { model_size })
            .build();
        
        // Each worker: 1 send + 1 receive = 2 events
        for (worker_id, worker) in &job.workers {
            assert_eq!(worker.template_events.len(), 2);
            
            let other_worker = if *worker_id == 0 { 1 } else { 0 };
            
            // Check send destination
            let send = &worker.template_events[0];
            assert!(send.kind == WorkerEventKind::FlowSend);
            assert_eq!(send.flow_send.as_ref().unwrap().dst_worker, other_worker);
            // For 2 workers: 2 * 1/2 * model_size = model_size
            assert_eq!(send.flow_send.as_ref().unwrap().size_bytes, model_size);
            
            // Check receive source
            let recv = &worker.template_events[1];
            assert!(recv.kind == WorkerEventKind::FlowReceive);
            assert_eq!(recv.flow_receive.as_ref().unwrap().src_worker, other_worker);
        }
    }
}
