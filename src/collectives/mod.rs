//! Collective communication operations for distributed ML training.
//!
//! This module provides high-level collective communication primitives that
//! automatically generate the underlying send/receive event DAGs. Users can
//! specify collectives like All-to-All or AllReduce without manually coding
//! the individual point-to-point flows.
//!
//! # Example
//!
//! ```rust,ignore
//! use monkeytree::collectives::{CollectiveJobBuilder, CollectiveOp};
//!
//! let job = CollectiveJobBuilder::new(job_id, 0, num_workers, iterations)
//!     .add_compute(800)  // all workers compute for 800ms
//!     .add_collective(CollectiveOp::AllReduce { model_size: 5_000_000_000 })
//!     .build();
//! ```

pub mod alltoall;
pub mod all_reduce;
pub mod all_to_all_ring;
pub mod strided_ring;
pub mod pipeline;
pub mod builder;

pub use alltoall::generate_alltoall_events;
pub use all_reduce::generate_allreduce_events;
pub use all_to_all_ring::generate_all_to_all_ring_events;
pub use strided_ring::generate_strided_ring_events;
pub use pipeline::{generate_pipeline_gpipe_events, generate_pipeline_with_dp_allreduce, PipelineConfig, PipelineSchedule, StageSpec};
pub use builder::CollectiveJobBuilder;


/// Specification for a collective communication operation.
#[derive(Debug, Clone)]
pub enum CollectiveOp {
    /// AllReduce: combines values from all workers and distributes result back.
    /// Uses ring topology with data size = 2 * (n-1) / n * model_size
    AllReduce {
        /// Model size in bytes (will be scaled by 2*(n-1)/n for actual transfer)
        model_size: u64,
    },
    
    /// All-to-All: each worker sends data to every other worker.
    /// Total data is divided by n*(n-1) to get per-flow size.
    /// All flows happen simultaneously.
    AllToAll {
        /// Total data size in bytes across all n*(n-1) flows
        total_data_size: u64,
    },
    
    /// All-to-All Ring: phased all-to-all using N-1 rounds of ring exchanges.
    /// In round r (1 to N-1), worker i sends to (i + r) % N.
    /// Each round completes before the next begins, creating a dependency chain.
    /// Total data is divided by n*(n-1) to get per-flow size.
    AllToAllRing {
        /// Total data size in bytes across all n*(n-1) flows
        total_data_size: u64,
    },
    
    /// Strided ring AllReduce: creates B independent rings where workers stride apart communicate.
    /// Worker i sends to (i + stride) % N, receives from (i - stride + N) % N.
    /// Creates `stride` rings, each with N/stride workers.
    /// Data size per worker = 2 * (ring_size - 1) / ring_size * model_size
    StridedRing {
        /// Distance between communicating workers (must divide num_workers)
        stride: usize,
        /// Model size in bytes (will be scaled by AllReduce formula for ring_size)
        model_size: u64,
    },
    
    /// Pipeline parallel: model divided into sequential stages with microbatch pipelining.
    /// Workers are divided into pipeline stages, with optional DP replicas per stage.
    /// Communication: point-to-point activations/gradients between adjacent stages.
    Pipeline {
        /// Pipeline configuration
        config: PipelineConfig,
        /// Model shard size per stage for final DP AllReduce (0 to skip AllReduce)
        model_shard_size: u64,
    },
}

/// Result of generating events for a collective operation.
/// Contains the events for each worker and metadata about the collective.
#[derive(Debug, Clone)]
pub struct CollectiveEvents {
    /// Events for each worker, indexed by worker ID (0..num_workers)
    pub worker_events: Vec<Vec<crate::simulator::WorkerEvent>>,
    /// Number of events generated per worker
    pub events_per_worker: usize,
    /// IDs of the final events that must complete before the collective is done
    /// (one set per worker)
    pub completion_event_ids: Vec<Vec<usize>>,
}

/// Phase in a job definition - either compute or a collective
#[derive(Debug, Clone)]
pub enum JobPhase {
    /// A compute phase where all workers compute for the given duration
    Compute { duration_us: u64 },
    /// A collective communication operation
    Collective(CollectiveOp),
}
