//! Job definition and construction functions.
//!
//! This module provides functions for fetching and building ML jobs.
//! Job definitions are loaded from YAML files in the `jobs/` directory.
//!
//! # Usage
//!
//! ```rust,ignore
//! use network_sim::utils::job_def::fetch_job;
//!
//! let job = fetch_job(0, "s1", 4, 100);
//! ```

use crate::simulator::MLJob;
use super::job_loader::{JobRegistry, load_default_registry};
use std::sync::OnceLock;

/// Global job registry cache
static JOB_REGISTRY: OnceLock<JobRegistry> = OnceLock::new();

/// Get or initialize the global job registry
fn get_registry() -> &'static JobRegistry {
    JOB_REGISTRY.get_or_init(|| {
        load_default_registry().expect("Failed to load job registry from jobs/ directory")
    })
}

/// Fetch a job by type name, building it with the specified parameters.
///
/// Job definitions are loaded from YAML files in the `jobs/` directory.
///
/// # Arguments
/// * `job_id` - Unique identifier for this job instance
/// * `kind` - Job type name (e.g., "s1", "s2", "s3", "canon")
/// * `num_workers` - Number of workers for the job
/// * `num_iterations` - Number of training iterations
///
/// # Panics
/// Panics if the job type is not found in the registry.
pub fn fetch_job(job_id: usize, kind: &str, num_workers: usize, num_iterations: usize) -> MLJob {
    let registry = get_registry();
    
    registry.get(kind)
        .unwrap_or_else(|| panic!("Unknown job type: '{}'. Available types: {:?}", kind, registry.job_types()))
        .build_job(job_id, num_workers, num_iterations)
}

/// Get a job definition (name, model_size, compute_time) for a job type.
/// 
/// Returns (name, model_size_bytes, compute_time_us) or None if not found.
pub fn get_job_definition(kind: &str) -> Option<(String, u64, u64)> {
    let registry = get_registry();
    
    registry.get(kind).map(|def| {
        (def.name.clone(), def.model_size_bytes, def.compute_time_us())
    })
}

/// List all available job types.
pub fn list_job_types() -> Vec<String> {
    let registry = get_registry();
    let mut types: Vec<String> = registry.job_types().iter().map(|s| s.to_string()).collect();
    types.sort();
    types
}
