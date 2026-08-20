use std::collections::HashMap;
use crate::simulator::ml_job::JobId;

/// Represents a communication phase within a job iteration
#[derive(Debug, Clone)]
pub struct CommunicationPhase {
    /// Duration of this phase in milliseconds
    pub duration_us: u64,
    /// Bandwidth demand during this phase in bytes/ms
    pub bandwidth_demand: u64,
    /// Whether this is an "Up" phase (high bandwidth) or "Down" phase (low bandwidth)
    pub is_up_phase: bool,
}

/// Profile of an ML job's communication pattern
#[derive(Debug, Clone)]
pub struct JobProfile {
    /// Job identifier
    pub job_id: JobId,
    /// Total iteration time in milliseconds
    pub iteration_time_us: u64,
    /// Communication phases within each iteration
    pub communication_phases: Vec<CommunicationPhase>,
    /// Number of workers for this job
    pub num_workers: usize,
    /// Job name for debugging
    pub name: Option<String>,
}

/// Represents a unified circle for geometric abstraction
#[derive(Debug, Clone)]
pub struct UnifiedCircle {
    /// Job this circle represents
    pub job_id: JobId,
    /// Perimeter of the circle (LCM of iteration times)
    pub perimeter_us: u64,
    /// Number of job iterations that fit in this circle
    pub num_iterations: u64,
    /// Bandwidth demand as a function of angle (0 to 2π)
    /// Stored as discrete samples with angular resolution
    pub bandwidth_samples: Vec<u64>,
    /// Angular resolution in degrees
    pub angular_resolution_deg: f64,
}

/// Compatibility score between jobs on a link
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct CompatibilityScore {
    /// Score value (0.0 = incompatible, 1.0 = fully compatible)
    pub score: f64,
    /// Whether this score represents full compatibility
    pub is_fully_compatible: bool,
}

impl CompatibilityScore {
    pub fn new(score: f64) -> Self {
        Self {
            score: score.max(0.0).min(1.0),
            is_fully_compatible: score >= 1.0,
        }
    }
    
    pub fn incompatible() -> Self {
        Self { score: 0.0, is_fully_compatible: false }
    }
    
    pub fn fully_compatible() -> Self {
        Self { score: 1.0, is_fully_compatible: true }
    }
}

/// Time shift for a job to achieve compatibility
#[derive(Debug, Clone, PartialEq)]
pub struct TimeShift {
    /// Job identifier
    pub job_id: JobId,
    /// Time shift in milliseconds
    pub shift_us: u64,
    /// Rotation angle in radians (before converting to time shift)
    pub rotation_angle: f64,
}

/// Bipartite affinity graph edge
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AffinityEdge {
    /// Job identifier
    pub job_id: JobId,
    /// Link identifier (topology-specific)
    pub link_id: usize,
    /// Weight of the edge (time shift for this job on this link)
    pub weight_us: u64,
}

/// Placement candidate for job scheduling
#[derive(Debug, Clone)]
pub struct PlacementCandidate {
    /// Job identifier
    pub job_id: JobId,
    /// Mapping from worker index to host index
    pub worker_to_host: HashMap<usize, usize>,
    /// Overall compatibility score for this placement
    pub compatibility_score: Option<CompatibilityScore>,
}

/// Complete Cassini schedule for the cluster
#[derive(Debug, Clone)]
pub struct CassiniSchedule {
    /// Schedule version for tracking updates
    pub version: u64,
    /// Time shifts for each job
    pub time_shifts: HashMap<JobId, TimeShift>,
    /// Per-job iteration periods in milliseconds (used for enforcement)
    pub job_periods: HashMap<JobId, u64>,
    /// When this schedule was computed
    pub computed_at_us: u64,
}

impl CassiniSchedule {
    pub fn new() -> Self {
        Self {
            version: 0,
            time_shifts: HashMap::new(),
            job_periods: HashMap::new(),
            computed_at_us: 0,
        }
    }
    
    pub fn add_time_shift(&mut self, time_shift: TimeShift) {
        self.time_shifts.insert(time_shift.job_id, time_shift);
    }
    
    pub fn get_time_shift(&self, job_id: JobId) -> Option<&TimeShift> {
        self.time_shifts.get(&job_id)
    }
}