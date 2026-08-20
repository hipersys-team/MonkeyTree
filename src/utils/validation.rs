//! Validation utilities for job and cluster configurations.

use crate::simulator::ml_job::MLJob;
use crate::schedulers::DEFAULT_BLOCK_SIZE;

/// Validates that a job's worker count is compatible with block-based scheduling.
///
/// # Arguments
/// * `job` - The job to validate
/// * `block_size` - The block size (default 8 for GPU clusters)
///
/// # Returns
/// `Ok(())` if valid, `Err(message)` describing the problem otherwise
pub fn validate_job_for_block_scheduler(job: &MLJob, block_size: usize) -> Result<(), String> {
    if job.num_workers % block_size != 0 {
        return Err(format!(
            "Job {} requests {} workers, but must be a multiple of {} for block-based scheduling",
            job.id, job.num_workers, block_size
        ));
    }
    Ok(())
}

/// Validates multiple jobs for block-based scheduling.
///
/// # Returns
/// `Ok(())` if all jobs are valid, `Err(message)` listing all invalid jobs otherwise
pub fn validate_jobs_for_block_scheduler(jobs: &[MLJob], block_size: usize) -> Result<(), String> {
    let invalid: Vec<_> = jobs
        .iter()
        .filter(|j| j.num_workers % block_size != 0)
        .collect();

    if invalid.is_empty() {
        Ok(())
    } else {
        let details: Vec<String> = invalid
            .iter()
            .map(|j| format!("  Job {}: {} workers (not divisible by {})", j.id, j.num_workers, block_size))
            .collect();
        Err(format!(
            "The following jobs are invalid for block-based scheduling:\n{}",
            details.join("\n")
        ))
    }
}

/// Validates that a worker count is compatible with block-based scheduling.
pub fn validate_worker_count(num_workers: usize) -> Result<(), String> {
    validate_worker_count_with_block_size(num_workers, DEFAULT_BLOCK_SIZE)
}

/// Validates that a worker count is compatible with a specific block size.
pub fn validate_worker_count_with_block_size(num_workers: usize, block_size: usize) -> Result<(), String> {
    if num_workers % block_size != 0 {
        Err(format!(
            "{} workers is not a multiple of block size {}",
            num_workers, block_size
        ))
    } else {
        Ok(())
    }
}

/// Validates that a topology is compatible with block-based scheduling.
///
/// The number of hosts per ToR must be divisible by the block size.
pub fn validate_topology_for_blocks(hosts_per_tor: usize, block_size: usize) -> Result<(), String> {
    if hosts_per_tor % block_size != 0 {
        Err(format!(
            "hosts_per_tor ({}) must be divisible by block_size ({})",
            hosts_per_tor, block_size
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_worker_count() {
        assert!(validate_worker_count_with_block_size(8, 8).is_ok());
        assert!(validate_worker_count_with_block_size(16, 8).is_ok());
        assert!(validate_worker_count_with_block_size(24, 8).is_ok());
        
        assert!(validate_worker_count_with_block_size(10, 8).is_err());
        assert!(validate_worker_count_with_block_size(7, 8).is_err());
        assert!(validate_worker_count_with_block_size(1, 8).is_err());
    }

    #[test]
    fn test_validate_topology() {
        assert!(validate_topology_for_blocks(48, 8).is_ok());
        assert!(validate_topology_for_blocks(64, 8).is_ok());
        
        assert!(validate_topology_for_blocks(50, 8).is_err());
        assert!(validate_topology_for_blocks(7, 8).is_err());
    }
}
