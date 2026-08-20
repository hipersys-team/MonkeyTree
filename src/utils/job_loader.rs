//! Job definition loader from YAML files.
//!
//! This module provides functionality to load job type definitions from YAML files
//! in the `jobs/` directory. Job definitions specify a sequence of steps that all
//! workers execute, using high-level collective operations.
//!
//! # YAML Schema
//!
//! ```yaml
//! name: "Job Name"
//! model_size_bytes: 5_000_000_000
//! steps:
//!   - !compute
//!       duration_us: 800
//!   - !all_reduce {}
//!   - !all_to_all
//!       total_size_bytes: 188_000_000
//!   - !all_to_all_ring
//!       total_size_bytes: 188_000_000
//!   - !strided_ring
//!       stride: 8
//!       data_size_bytes: 1_000_000_000
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};

use crate::simulator::MLJob;
use crate::collectives::{CollectiveJobBuilder, CollectiveOp, PipelineConfig, PipelineSchedule, StageSpec};

/// Parameters for a compute step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeParams {
    pub duration_us: u64,
}

/// Parameters for all-reduce (empty, uses job's model_size)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AllReduceParams {}

/// Parameters for all-to-all
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllToAllParams {
    /// Total data size in bytes across all n*(n-1) flows
    pub total_size_bytes: u64,
}

/// Parameters for all-to-all ring (phased version)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllToAllRingParams {
    /// Total data size in bytes across all n*(n-1) flows
    pub total_size_bytes: u64,
}

/// Parameters for strided ring AllReduce
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StridedRingParams {
    /// Distance between communicating workers (must divide num_workers)
    pub stride: usize,
    /// Model size in bytes (will be scaled by 2*(ring_size-1)/ring_size)
    pub model_size_bytes: u64,
}

/// Parameters for a single pipeline stage (YAML-friendly)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStageParams {
    /// Forward pass compute time in microseconds
    pub forward_us: u64,
    /// Backward pass compute time in microseconds
    pub backward_us: u64,
    /// Activation tensor size sent to next stage (bytes)
    #[serde(default = "default_activation_size")]
    pub activation_size_bytes: u64,
    /// Gradient tensor size (bytes), defaults to activation_size if not specified
    #[serde(default)]
    pub gradient_size_bytes: Option<u64>,
}

fn default_activation_size() -> u64 { 1_000_000_000 }

impl PipelineStageParams {
    fn to_stage_spec(&self) -> StageSpec {
        StageSpec {
            forward_us: self.forward_us,
            backward_us: self.backward_us,
            activation_size_bytes: self.activation_size_bytes,
            gradient_size_bytes: self.gradient_size_bytes.unwrap_or(self.activation_size_bytes),
        }
    }
}

/// Parameters for pipeline parallel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineParams {
    /// Number of pipeline stages
    pub num_stages: usize,
    /// Tensor parallel size (GPUs per TP group within each stage)
    #[serde(default = "default_tp_size")]
    pub tp_size: usize,
    /// Number of DP replicas (full pipeline replicas).
    /// If not specified, computed from num_workers at job creation time.
    #[serde(default)]
    pub dp_replicas: Option<usize>,
    /// Number of microbatches per iteration
    #[serde(default = "default_microbatches")]
    pub microbatches: usize,
    /// Per-stage specifications (if not provided, uses uniform stages)
    #[serde(default)]
    pub stages: Vec<PipelineStageParams>,
    /// Uniform forward time if stages not specified
    #[serde(default)]
    pub forward_us: Option<u64>,
    /// Uniform backward time if stages not specified  
    #[serde(default)]
    pub backward_us: Option<u64>,
    /// Uniform activation size if stages not specified
    #[serde(default)]
    pub activation_size_bytes: Option<u64>,
    /// Model shard size per stage for final DP AllReduce (0 to skip)
    #[serde(default)]
    pub model_shard_bytes: u64,
}

fn default_tp_size() -> usize { 1 }
fn default_microbatches() -> usize { 1 }

impl PipelineParams {
    /// Workers per DP replica (one full pipeline).
    pub fn workers_per_replica(&self) -> usize {
        self.num_stages * self.tp_size
    }
    
    /// Convert to PipelineConfig, computing dp_replicas from num_workers if not specified.
    /// 
    /// # Panics
    /// Panics if num_workers is not divisible by workers_per_replica.
    pub fn to_config_with_workers(&self, num_workers: usize) -> PipelineConfig {
        let workers_per_replica = self.workers_per_replica();
        
        let dp_replicas = match self.dp_replicas {
            Some(dp) => dp,
            None => {
                assert!(
                    num_workers % workers_per_replica == 0,
                    "num_workers ({}) must be divisible by workers_per_replica ({} = {} stages * {} tp)",
                    num_workers, workers_per_replica, self.num_stages, self.tp_size
                );
                num_workers / workers_per_replica
            }
        };
        
        // Validate that workers_per_stage is a multiple of 8 (GPU block size)
        // This ensures pipeline stages align with block-based scheduling
        let workers_per_stage = self.tp_size * dp_replicas;
        const GPU_BLOCK_SIZE: usize = 8;
        assert!(
            workers_per_stage % GPU_BLOCK_SIZE == 0,
            "workers_per_stage ({} = {} tp_size * {} dp_replicas) must be a multiple of {} (GPU block size). \
             For pipeline jobs, total GPUs must be a multiple of {} (num_stages * block_size).",
            workers_per_stage, self.tp_size, dp_replicas, GPU_BLOCK_SIZE,
            self.num_stages * GPU_BLOCK_SIZE
        );
        
        let stages = if !self.stages.is_empty() {
            self.stages.iter().map(|s| s.to_stage_spec()).collect()
        } else {
            vec![StageSpec {
                forward_us: self.forward_us.unwrap_or(50000),
                backward_us: self.backward_us.unwrap_or(100000),
                activation_size_bytes: self.activation_size_bytes.unwrap_or(1_000_000_000),
                gradient_size_bytes: self.activation_size_bytes.unwrap_or(1_000_000_000),
            }]
        };
        
        PipelineConfig {
            num_stages: self.num_stages,
            tp_size: self.tp_size,
            dp_replicas,
            microbatches: self.microbatches,
            schedule: PipelineSchedule::GPipe,
            stages,
        }
    }
    
    /// Get dp_replicas, defaulting to 1 if not specified (for backwards compat).
    pub fn dp_replicas_or_default(&self) -> usize {
        self.dp_replicas.unwrap_or(1)
    }
}

/// A step in a job definition, using adjacently-tagged format for YAML compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStep {
    /// Compute phase - all workers compute for the specified duration
    Compute(ComputeParams),
    /// AllReduce - ring-based allreduce with data = 2*(n-1)/n * model_size
    AllReduce(AllReduceParams),
    /// All-to-All - each worker sends to all other workers (simultaneous)
    AllToAll(AllToAllParams),
    /// All-to-All Ring - phased all-to-all using N-1 rounds of ring exchanges
    AllToAllRing(AllToAllRingParams),
    /// Strided Ring - multiple independent rings with specified stride
    StridedRing(StridedRingParams),
    /// Pipeline parallel - model split into sequential stages with microbatching
    Pipeline(PipelineParams),
}

/// A job type definition loaded from YAML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobDefinition {
    /// Human-readable name for this job type
    pub name: String,
    /// Model size in bytes - used for collective flow sizes and migrations
    pub model_size_bytes: u64,
    /// Sequence of steps that all workers execute each iteration
    pub steps: Vec<JobStep>,
}

impl JobDefinition {
    /// Load a job definition from a YAML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read job file {:?}: {}", path.as_ref(), e))?;
        
        Self::from_yaml(&content)
    }
    
    /// Load a job definition from a YAML string
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        serde_yaml::from_str(yaml)
            .map_err(|e| format!("Failed to parse job YAML: {}", e))
    }
    
    /// Get the compute time per iteration in microseconds.
    /// 
    /// For pipeline jobs, this accounts for GPipe pipelining:
    /// - Pipeline steps = num_stages + microbatches - 1
    /// - Compute time = pipeline_steps * (forward_us + backward_us)
    pub fn compute_time_us(&self) -> u64 {
        let mut total_us: u64 = 0;
        
        for step in &self.steps {
            match step {
                JobStep::Compute(params) => {
                    total_us += params.duration_us;
                }
                JobStep::Pipeline(params) => {
                    // GPipe pipelining: (S + M - 1) steps for forward, same for backward
                    // where S = num_stages, M = microbatches
                    let pipeline_steps = params.num_stages + params.microbatches - 1;
                    let forward_us = params.forward_us.unwrap_or(50000);
                    let backward_us = params.backward_us.unwrap_or(100000);
                    total_us += (pipeline_steps as u64) * (forward_us + backward_us);
                }
                _ => {}
            }
        }
        
        total_us
    }
    
    /// Compute the ideal network time per iteration in microseconds.
    /// 
    /// This accounts for all collective operations in the job:
    /// - AllReduce: 2 * (n-1)/n * model_size
    /// - AllToAll: total_size / n per worker
    /// - AllToAllRing: total_size / n per worker  
    /// - StridedRing: 2 * (ring_size-1)/ring_size * model_size, where ring_size = n/stride
    pub fn network_time_us(&self, num_workers: usize, bandwidth_bps: f64) -> u64 {
        let mut total_bytes: f64 = 0.0;
        
        for step in &self.steps {
            match step {
                JobStep::Compute(_) => {}
                JobStep::AllReduce(_) => {
                    // Ring allreduce: 2 * (n-1)/n * model_size
                    if num_workers > 1 {
                        total_bytes += 2.0 * (num_workers - 1) as f64 / num_workers as f64 
                            * self.model_size_bytes as f64;
                    }
                }
                JobStep::AllToAll(params) => {
                    // Each worker sends (n-1)/n of total data
                    if num_workers > 1 {
                        total_bytes += (num_workers - 1) as f64 / num_workers as f64 
                            * params.total_size_bytes as f64;
                    }
                }
                JobStep::AllToAllRing(params) => {
                    // Same as AllToAll in terms of data volume
                    if num_workers > 1 {
                        total_bytes += (num_workers - 1) as f64 / num_workers as f64 
                            * params.total_size_bytes as f64;
                    }
                }
                JobStep::StridedRing(params) => {
                    // Strided ring: ring_size = n/stride, data = 2*(ring_size-1)/ring_size * model_size
                    let ring_size = if params.stride > 0 { num_workers / params.stride } else { 1 };
                    if ring_size > 1 {
                        total_bytes += 2.0 * (ring_size - 1) as f64 / ring_size as f64 
                            * params.model_size_bytes as f64;
                    }
                }
                JobStep::Pipeline(params) => {
                    // Pipeline: activations/gradients flow through stages with pipelining
                    // 
                    // With GPipe pipelining, we have (S + M - 1) steps for each phase.
                    // The last stage doesn't send in forward, first stage doesn't send in backward.
                    // So we subtract 1 transfer from each phase:
                    //
                    // Network time = 2 * (S + M - 1 - 1) * time_per_transfer
                    //              = 2 * (S + M - 2) * time_per_transfer
                    let act_size = params.activation_size_bytes.unwrap_or(1_000_000_000) as f64;
                    let pipeline_steps = (params.num_stages + params.microbatches - 1) as f64;
                    
                    // Subtract 1 from each phase for the stage that doesn't communicate
                    // Forward: last stage doesn't send, Backward: first stage doesn't send
                    total_bytes += 2.0 * (pipeline_steps - 1.0) * act_size;
                    
                    // DP AllReduce if enabled (happens once per iteration after backward)
                    let workers_per_replica = params.workers_per_replica();
                    let dp_replicas = params.dp_replicas
                        .unwrap_or_else(|| num_workers / workers_per_replica);
                    if params.model_shard_bytes > 0 && dp_replicas > 1 {
                        let d = dp_replicas as f64;
                        total_bytes += 2.0 * (d - 1.0) / d * params.model_shard_bytes as f64;
                    }
                }
            }
        }
        
        // Convert bytes to time: time = bytes * 8 / bandwidth
        (total_bytes * 8.0 / bandwidth_bps * 1_000_000.0) as u64
    }
    
    /// Compute the ideal job duration in microseconds (no network congestion).
    pub fn ideal_duration_us(&self, num_workers: usize, num_iterations: usize, bandwidth_bps: f64) -> u64 {
        let compute_us = self.compute_time_us();
        let network_us = self.network_time_us(num_workers, bandwidth_bps);
        (compute_us + network_us) * num_iterations as u64
    }
    
    /// Build an MLJob from this definition using the CollectiveJobBuilder.
    pub fn build_job(
        &self,
        job_id: usize,
        num_workers: usize,
        num_iterations: usize,
    ) -> MLJob {
        let mut builder = CollectiveJobBuilder::new(job_id, 0, num_workers, num_iterations)
            .with_name(self.name.clone());
        
        // Track the maximum stride across all strided ring steps.
        // This determines the number of independent rings the job creates.
        let mut max_stride: usize = 1;
        // Track whether this job contains a pipeline phase
        let mut has_pipeline = false;
        
        for step in &self.steps {
            match step {
                JobStep::Compute(params) => {
                    builder = builder.add_compute(params.duration_us);
                }
                JobStep::AllReduce(_) => {
                    builder = builder.add_collective(CollectiveOp::AllReduce {
                        model_size: self.model_size_bytes,
                    });
                }
                JobStep::AllToAll(params) => {
                    builder = builder.add_collective(CollectiveOp::AllToAll {
                        total_data_size: params.total_size_bytes,
                    });
                }
                JobStep::AllToAllRing(params) => {
                    builder = builder.add_collective(CollectiveOp::AllToAllRing {
                        total_data_size: params.total_size_bytes,
                    });
                }
                JobStep::StridedRing(params) => {
                    max_stride = max_stride.max(params.stride);
                    builder = builder.add_collective(CollectiveOp::StridedRing {
                        stride: params.stride,
                        model_size: params.model_size_bytes,
                    });
                }
                JobStep::Pipeline(params) => {
                    let config = params.to_config_with_workers(num_workers);
                    // For pipeline, ring_count is set by the builder to num_stages * dp_replicas
                    // We still track max_stride for any non-pipeline phases
                    max_stride = max_stride.max(config.dp_replicas);
                    has_pipeline = true;
                    builder = builder.add_collective(CollectiveOp::Pipeline {
                        config,
                        model_shard_size: params.model_shard_bytes,
                    });
                }
            }
        }
        
        let job = builder.build();
        
        // For pipeline jobs, ring_count is already set correctly by the builder
        // For non-pipeline jobs (e.g., strided ring), set ring_count to max_stride
        if has_pipeline {
            job  // ring_count already set by CollectiveJobBuilder
        } else {
            job.with_ring_count(max_stride)
        }
    }
}

/// Registry of job definitions loaded from YAML files
#[derive(Debug, Default)]
pub struct JobRegistry {
    definitions: HashMap<String, JobDefinition>,
}

impl JobRegistry {
    /// Load all job definitions from the given directory (recursively).
    ///
    /// Files in the top-level directory get keys equal to their stem (e.g.
    /// `jobs/s1.yaml` → `"s1"`).  Files in subdirectories use the relative
    /// path from `jobs_dir` without the extension (e.g.
    /// `jobs/seq16k/gpt_120b_32gpu.yaml` → `"seq16k/gpt_120b_32gpu"`).
    pub fn load_all<P: AsRef<Path>>(jobs_dir: P) -> Result<Self, String> {
        let jobs_dir = jobs_dir.as_ref();
        
        if !jobs_dir.exists() {
            return Err(format!("Jobs directory does not exist: {:?}", jobs_dir));
        }
        
        let mut definitions = HashMap::new();
        Self::load_dir_recursive(jobs_dir, jobs_dir, &mut definitions)?;
        Ok(Self { definitions })
    }

    fn load_dir_recursive(
        root: &Path,
        dir: &Path,
        definitions: &mut HashMap<String, JobDefinition>,
    ) -> Result<(), String> {
        let entries = fs::read_dir(dir)
            .map_err(|e| format!("Failed to read directory {:?}: {}", dir, e))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();

            if path.is_dir() {
                Self::load_dir_recursive(root, &path, definitions)?;
            } else if path.extension().map(|e| e == "yaml" || e == "yml").unwrap_or(false) {
                let def = JobDefinition::from_file(&path)?;
                let rel = path.strip_prefix(root)
                    .map_err(|e| format!("strip_prefix failed: {}", e))?;
                let job_type = rel.with_extension("")
                    .to_str()
                    .ok_or_else(|| format!("Non-UTF8 path: {:?}", rel))?
                    .to_string();
                definitions.insert(job_type, def);
            }
        }
        Ok(())
    }
    
    /// Get a job definition by type name
    pub fn get(&self, job_type: &str) -> Option<&JobDefinition> {
        self.definitions.get(job_type)
    }
    
    /// Build an MLJob from a registered job type
    pub fn build_job(
        &self,
        job_type: &str,
        job_id: usize,
        num_workers: usize,
        num_iterations: usize,
    ) -> Result<MLJob, String> {
        let def = self.get(job_type)
            .ok_or_else(|| format!("Unknown job type: {}", job_type))?;
        Ok(def.build_job(job_id, num_workers, num_iterations))
    }
    
    /// List all registered job types
    pub fn job_types(&self) -> Vec<&str> {
        self.definitions.keys().map(|s| s.as_str()).collect()
    }
    
    /// Get the number of registered job types
    pub fn len(&self) -> usize {
        self.definitions.len()
    }
    
    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }
}

/// Default jobs directory path relative to the crate root
pub const DEFAULT_JOBS_DIR: &str = "jobs";

/// Load a job registry from the default jobs directory
pub fn load_default_registry() -> Result<JobRegistry, String> {
    JobRegistry::load_all(DEFAULT_JOBS_DIR)
}

/// Convenience function to build a job from the default registry
pub fn build_job_from_yaml(
    job_type: &str,
    job_id: usize,
    num_workers: usize,
    num_iterations: usize,
) -> Result<MLJob, String> {
    let registry = load_default_registry()?;
    registry.build_job(job_type, job_id, num_workers, num_iterations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::ml_worker::WorkerEventKind;
    
    const S1_YAML: &str = r#"
name: "S1"
model_size_bytes: 6875000000
steps:
  - !compute
    duration_us: 500
  - !all_reduce {}
"#;
    
    #[test]
    fn test_parse_s1_yaml() {
        let def = JobDefinition::from_yaml(S1_YAML).unwrap();
        assert_eq!(def.name, "S1");
        assert_eq!(def.model_size_bytes, 6_875_000_000);
        assert_eq!(def.steps.len(), 2);
        assert_eq!(def.compute_time_us(), 500);
    }
    
    #[test]
    fn test_build_job_from_yaml() {
        let def = JobDefinition::from_yaml(S1_YAML).unwrap();
        let job = def.build_job(0, 4, 10);
        
        assert_eq!(job.id, 0);
        assert_eq!(job.num_workers, 4);
        assert_eq!(job.total_iterations, 10);
        assert_eq!(job.name, Some("S1".to_string()));
        
        // Each worker should have 3 events: compute, send, receive
        for (_, worker) in &job.workers {
            assert_eq!(worker.template_events.len(), 3);
            
            // First is compute
            assert!(worker.template_events[0].kind == WorkerEventKind::Compute);
            
            // Second is send
            assert!(worker.template_events[1].kind == WorkerEventKind::FlowSend);
            
            // Third is receive
            assert!(worker.template_events[2].kind == WorkerEventKind::FlowReceive);
        }
    }
    
    #[test]
    fn test_allreduce_topology() {
        let def = JobDefinition::from_yaml(S1_YAML).unwrap();
        let job = def.build_job(0, 4, 10);
        
        // Expected data size: 2 * (4-1)/4 * 6.875GB = 1.5 * 6.875GB = 10.3125GB
        let expected_data = ((2.0 * 3.0 / 4.0) * 6_875_000_000.0) as u64;
        
        // Check ring topology: 0→1→2→3→0
        for (worker_id, worker) in &job.workers {
            let send = &worker.template_events[1];
            let recv = &worker.template_events[2];
            
            let expected_dst = (*worker_id + 1) % 4;
            let expected_src = if *worker_id == 0 { 3 } else { *worker_id - 1 };
            
            assert_eq!(send.flow_send.as_ref().unwrap().dst_worker, expected_dst);
            assert_eq!(recv.flow_receive.as_ref().unwrap().src_worker, expected_src);
            
            // Flow size should be 2*(n-1)/n * model_size
            assert_eq!(send.flow_send.as_ref().unwrap().size_bytes, expected_data);
        }
    }
    
    #[test]
    fn test_all_to_all_yaml() {
        let yaml = r#"
name: "AllToAll Test"
model_size_bytes: 1000000000
steps:
  - !compute
    duration_us: 100
  - !all_to_all
    total_size_bytes: 500000000
"#;
        
        let def = JobDefinition::from_yaml(yaml).unwrap();
        let job = def.build_job(0, 3, 5);
        
        // 3 workers: each has compute + 2 sends + 2 receives = 5 events
        for (_, worker) in &job.workers {
            assert_eq!(worker.template_events.len(), 5);
        }
    }
    
    #[test]
    fn test_strided_ring_yaml() {
        let yaml = r#"
name: "Strided Ring Test"
model_size_bytes: 1000000000
steps:
  - !compute
    duration_us: 50
  - !strided_ring
    stride: 2
    model_size_bytes: 500000000
"#;
        
        let def = JobDefinition::from_yaml(yaml).unwrap();
        let job = def.build_job(0, 8, 10);
        
        assert_eq!(job.name, Some("Strided Ring Test".to_string()));
        
        // 8 workers with stride 2: each has compute + 1 send + 1 receive = 3 events
        for (_, worker) in &job.workers {
            assert_eq!(worker.template_events.len(), 3);
        }
        
        // Check topology: worker 0 sends to 2, receives from 6
        let worker_0 = job.workers.get(&0).unwrap();
        let send = &worker_0.template_events[1];
        let recv = &worker_0.template_events[2];
        
        assert_eq!(send.flow_send.as_ref().unwrap().dst_worker, 2);
        assert_eq!(recv.flow_receive.as_ref().unwrap().src_worker, 6);
        // With 8 workers, stride=2: ring_size=4, so data = 2*(4-1)/4 * 500M = 750M
        assert_eq!(send.flow_send.as_ref().unwrap().size_bytes, 750_000_000);
        
        // Check topology: worker 3 sends to 5, receives from 1
        let worker_3 = job.workers.get(&3).unwrap();
        let send = &worker_3.template_events[1];
        let recv = &worker_3.template_events[2];
        
        assert_eq!(send.flow_send.as_ref().unwrap().dst_worker, 5);
        assert_eq!(recv.flow_receive.as_ref().unwrap().src_worker, 1);
    }
    
    #[test]
    fn test_pipeline_yaml() {
        let yaml = r#"
name: "Pipeline Test"
model_size_bytes: 8000000000
steps:
  - !pipeline
    num_stages: 2
    dp_replicas: 8
    microbatches: 2
    forward_us: 50000
    backward_us: 100000
    activation_size_bytes: 500000000
    model_shard_bytes: 4000000000
"#;
        
        let def = JobDefinition::from_yaml(yaml).unwrap();
        // 2 stages * 8 replicas = 16 workers (workers_per_stage = 8, which is block-aligned)
        let job = def.build_job(0, 16, 10);
        
        assert_eq!(job.name, Some("Pipeline Test".to_string()));
        assert_eq!(job.num_workers, 16);
        
        // Stage 0 workers (0-7) should have different events than stage 1 workers (8-15)
        // Stage 0: forward compute, send activation, recv gradient, backward compute, DP allreduce
        // Stage 1: recv activation, forward compute, backward compute, send gradient, DP allreduce
        
        // Verify all workers have events
        for (_, worker) in &job.workers {
            assert!(!worker.template_events.is_empty(), "Worker should have events");
        }
        
        // With layout (TP fastest, then DP, then PP/stage slowest):
        // Stage 0: workers 0-7 (DP replicas 0-7)
        // Stage 1: workers 8-15 (DP replicas 0-7)
        
        // Check that worker 0 sends to worker 8 (stage 0 replica 0 -> stage 1 replica 0)
        let worker_0 = job.workers.get(&0).unwrap();
        let has_send_to_8 = worker_0.template_events.iter().any(|e| {
            e.flow_send.as_ref().map(|s| s.dst_worker == 8).unwrap_or(false)
        });
        assert!(has_send_to_8, "Worker 0 should send to worker 8 (pipeline send)");
        
        // Check that worker 0 has DP AllReduce with worker 1 (same stage, next dp replica)
        let has_send_to_1 = worker_0.template_events.iter().any(|e| {
            e.flow_send.as_ref().map(|s| s.dst_worker == 1).unwrap_or(false)
        });
        assert!(has_send_to_1, "Worker 0 should have DP AllReduce with worker 1");
    }
}
