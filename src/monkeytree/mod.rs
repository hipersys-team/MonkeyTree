//! MonkeyTree System Module
//!
//! MonkeyTree monitors cluster fragmentation and triggers worker migrations
//! to consolidate jobs onto fewer ToRs when fragmentation exceeds a threshold.
//!
//! ## Variants
//!
//! - [`MonkeyTreeSystem`]: Base module that works with any router (e.g., ECMP)
//! - [`MonkeyTreeCrux`]: Combined module with Crux system-directed routing
//! - [`MonkeyTreePerfect`]: Combined module with perfect routing via edge coloring

pub mod fragmentation;
pub mod ilp;
pub mod system;
pub mod crux;
pub mod perfect_matching;
pub mod perfect;
pub mod sglb;
pub mod fifo;
pub mod rail_fragmentation;
pub mod rail_system;
pub mod rail_crux;
pub mod rail_perfect;

pub use fragmentation::{
    // Segment-based fragmentation (pipeline-aware)
    SegmentId, JobSegment, SegmentFragmentation, ToRSegmentStats,
    build_segments_from_context, compute_segment_fragmentation,
    print_segment_fragmentation_summary,
};

// Legacy job-based fragmentation (deprecated)
#[allow(deprecated)]
pub use fragmentation::{
    ClusterFragmentation, ToRFragmentationStats, compute_cluster_fragmentation, print_fragmentation_summary,
};

pub use ilp::{
    SolveStatus,
    // Segment-based ILP
    SegmentILPInput, SegmentILPSolution, solve_segment_migration_ilp, compute_segment_migrations,
};

// Legacy job-based ILP (deprecated)
#[allow(deprecated)]
pub use ilp::{ILPInput, ILPSolution, solve_migration_ilp, compute_migrations};
pub use system::{MonkeyTreeConfig, MonkeyTreeSystem};
pub use crux::MonkeyTreeCrux;
pub use perfect::{SpinePerfectRouter, MonkeyTreePerfect, PerfectCore, FlowTemplateSpec, JobInfo};
pub use fifo::FifoPerfect;
pub use sglb::MonkeyTreeSGLB;

// Rail-optimized topology variants
pub use rail_fragmentation::compute_pod_fragmentation;
pub use rail_system::RailMonkeyTreeSystem;
pub use rail_crux::RailMonkeyTreeCrux;
pub use rail_perfect::RailMonkeyTreePerfect;

// Type aliases for common configurations
use crate::spine::SpineEcmpRouter;

/// MonkeyTree with ECMP routing (no system-directed routing)
pub type MonkeyTreeEcmp = MonkeyTreeSystem<SpineEcmpRouter>;
