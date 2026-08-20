//! Block-based scheduler for GPU clusters.
//!
//! This scheduler allocates hosts in contiguous blocks of a fixed size (default 8).
//! This models GPU clusters where each physical server has 8 GPUs, and jobs must
//! be allocated in units of complete servers.
//!
//! Key properties:
//! - Hosts are grouped into blocks: [0..8), [8..16), [16..24), etc.
//! - Jobs must request workers in multiples of `block_size`
//! - Allocation always assigns complete blocks
//! - Migrations also happen at block granularity

use std::collections::{HashMap, VecDeque};

use crate::network::topology::Topology;
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{JobId, MLJob};
use crate::system_modules::cassini::PlacementCandidate;

/// Block size for GPU allocation (8 GPUs per physical server)
pub const DEFAULT_BLOCK_SIZE: usize = 8;

// Debug flag for migration tracking. Enable to trace host assignments during scheduling.
const DEBUG_MIGRATION: bool = false;

/// A block-based job scheduler that allocates hosts in contiguous blocks.
///
/// This scheduler is designed for GPU clusters where:
/// - Each physical server has 8 GPUs
/// - Jobs must request workers in multiples of 8
/// - Allocation happens at server (block) granularity
#[derive(Debug)]
pub struct BlockScheduler {
    /// Queue of jobs waiting to be scheduled (FIFO order)
    job_queue: VecDeque<JobId>,
    /// Number of hosts per ToR switch
    hosts_per_tor: usize,
    /// Number of hosts per block (e.g., 8 GPUs per server)
    block_size: usize,
}

impl BlockScheduler {
    /// Creates a new BlockScheduler.
    ///
    /// # Arguments
    /// * `hosts_per_tor` - Number of hosts (GPUs) per ToR switch
    /// * `block_size` - Number of hosts per block (default 8)
    ///
    /// # Panics
    /// Panics if `hosts_per_tor` is not divisible by `block_size`
    pub fn new(hosts_per_tor: usize, block_size: usize) -> Self {
        assert!(hosts_per_tor > 0, "hosts_per_tor must be positive");
        assert!(block_size > 0, "block_size must be positive");
        assert!(
            hosts_per_tor % block_size == 0,
            "hosts_per_tor ({}) must be divisible by block_size ({})",
            hosts_per_tor,
            block_size
        );
        Self {
            job_queue: VecDeque::new(),
            hosts_per_tor,
            block_size,
        }
    }

    /// Creates a new BlockScheduler with default block size of 8.
    pub fn with_default_block_size(hosts_per_tor: usize) -> Self {
        Self::new(hosts_per_tor, DEFAULT_BLOCK_SIZE)
    }

    /// Returns the block size (hosts per block)
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the number of blocks per ToR
    pub fn blocks_per_tor(&self) -> usize {
        self.hosts_per_tor / self.block_size
    }

    /// Converts a block index to the starting host index
    #[inline]
    fn block_to_host(&self, block_idx: usize) -> usize {
        block_idx * self.block_size
    }

    /// Converts a host index to its block index
    #[inline]
    #[allow(dead_code)]
    fn host_to_block(&self, host_idx: usize) -> usize {
        host_idx / self.block_size
    }

    /// Returns the range of host indices for a given block
    #[inline]
    fn block_host_range(&self, block_idx: usize) -> std::ops::Range<usize> {
        let start = self.block_to_host(block_idx);
        start..start + self.block_size
    }

    /// Returns the ToR index for a given block
    #[inline]
    #[allow(dead_code)]
    fn block_to_tor(&self, block_idx: usize) -> usize {
        let host_start = self.block_to_host(block_idx);
        host_start / self.hosts_per_tor
    }

    /// Checks if a block is entirely free
    fn is_block_free(&self, block_idx: usize, host_busy: &[bool]) -> bool {
        let range = self.block_host_range(block_idx);
        if range.end > host_busy.len() {
            return false;
        }
        range.into_iter().all(|h| !host_busy[h])
    }

    /// Gets all free blocks, returning their block indices
    fn get_free_blocks(&self, host_busy: &[bool]) -> Vec<usize> {
        let num_blocks = host_busy.len() / self.block_size;
        (0..num_blocks)
            .filter(|&b| self.is_block_free(b, host_busy))
            .collect()
    }

    /// Gets free blocks within a specific ToR
    fn get_free_blocks_in_tor(&self, tor_idx: usize, host_busy: &[bool]) -> Vec<usize> {
        let blocks_per_tor = self.blocks_per_tor();
        let block_start = tor_idx * blocks_per_tor;
        let block_end = block_start + blocks_per_tor;
        
        (block_start..block_end)
            .filter(|&b| self.is_block_free(b, host_busy))
            .collect()
    }

    /// Computes free block capacity per ToR
    /// Returns vec of (tor_idx, free_block_count)
    fn compute_tor_block_capacities(&self, host_busy: &[bool]) -> Vec<(usize, usize)> {
        let num_blocks = host_busy.len() / self.block_size;
        let blocks_per_tor = self.blocks_per_tor();
        let num_tors = (num_blocks + blocks_per_tor - 1) / blocks_per_tor;
        
        let mut capacities = Vec::with_capacity(num_tors);
        for tor_idx in 0..num_tors {
            let free_blocks = self.get_free_blocks_in_tor(tor_idx, host_busy).len();
            if free_blocks > 0 {
                capacities.push((tor_idx, free_blocks));
            }
        }
        capacities
    }

    /// Attempts to place workers using block-based allocation.
    /// 
    /// # Arguments
    /// * `num_workers` - Must be divisible by block_size
    /// * `host_busy` - Current host allocation state
    ///
    /// # Returns
    /// Some(worker_to_host mapping) if successful, None if not enough capacity
    pub(crate) fn compute_placement(&self, num_workers: usize, host_busy: &[bool]) -> Option<HashMap<usize, usize>> {
        // Validate worker count is divisible by block size
        if num_workers % self.block_size != 0 {
            panic!(
                "Job requested {} workers, but must be a multiple of block_size ({})",
                num_workers, self.block_size
            );
        }

        let num_blocks_needed = num_workers / self.block_size;
        
        // Check total capacity
        let free_blocks = self.get_free_blocks(host_busy);
        if free_blocks.len() < num_blocks_needed {
            return None; // Not enough capacity
        }

        let tor_capacities = self.compute_tor_block_capacities(host_busy);
        if tor_capacities.is_empty() {
            return None;
        }

        let mut worker_to_host = HashMap::new();

        // Try to find a single ToR that can fit all blocks (best-fit)
        let mut fitting_tors: Vec<_> = tor_capacities
            .iter()
            .filter(|&&(_, free)| free >= num_blocks_needed)
            .copied()
            .collect();

        if !fitting_tors.is_empty() {
            // Sort by free capacity ascending (least free first = best fit)
            fitting_tors.sort_by_key(|&(_, free)| free);
            let (best_tor, _) = fitting_tors[0];

            // Place all blocks on this ToR
            let free_blocks = self.get_free_blocks_in_tor(best_tor, host_busy);
            let mut worker_id = 0;
            
            for &block_idx in free_blocks.iter().take(num_blocks_needed) {
                let host_range = self.block_host_range(block_idx);
                for host_idx in host_range {
                    worker_to_host.insert(worker_id, host_idx);
                    worker_id += 1;
                }
            }

            return Some(worker_to_host);
        }

        // No single ToR can fit - use greedy multi-ToR placement
        let mut assigned = host_busy.to_vec();
        let mut worker_id = 0;
        let mut remaining_blocks = num_blocks_needed;

        while remaining_blocks > 0 {
            let mut current_capacities = self.compute_tor_block_capacities(&assigned);
            if current_capacities.is_empty() {
                panic!("No ToRs available to place blocks");
            }

            // Sort by free capacity descending (largest first)
            current_capacities.sort_by_key(|&(_, free)| std::cmp::Reverse(free));
            let (best_tor, free_count) = current_capacities[0];

            let blocks_to_place = remaining_blocks.min(free_count);
            let free_blocks = self.get_free_blocks_in_tor(best_tor, &assigned);

            for &block_idx in free_blocks.iter().take(blocks_to_place) {
                let host_range = self.block_host_range(block_idx);
                for host_idx in host_range.clone() {
                    worker_to_host.insert(worker_id, host_idx);
                    assigned[host_idx] = true;
                    worker_id += 1;
                }
                remaining_blocks -= 1;
            }
        }

        Some(worker_to_host)
    }
}

impl Default for BlockScheduler {
    fn default() -> Self {
        // Default: 48 hosts per ToR (6 servers × 8 GPUs), block size 8
        Self::new(48, DEFAULT_BLOCK_SIZE)
    }
}

impl JobScheduler for BlockScheduler {
    fn try_schedule_job<T: Topology>(
        &mut self,
        job: &mut MLJob,
        _topology: &T,
        available_hosts: &[bool],
    ) -> bool {
        if DEBUG_MIGRATION {
            let busy_count = available_hosts.iter().filter(|&&b| b).count();
            let free_count = available_hosts.len() - busy_count;
            println!("DEBUG BlockScheduler::try_schedule_job job={} num_workers={} total_hosts={} busy={} free={}",
                job.id, job.num_workers, available_hosts.len(), busy_count, free_count);
            
            let busy_hosts: Vec<usize> = available_hosts.iter().enumerate()
                .filter(|(_, &b)| b)
                .map(|(i, _)| i)
                .collect();
            if busy_hosts.len() <= 100 {
                println!("DEBUG   busy_hosts={:?}", busy_hosts);
            } else {
                println!("DEBUG   busy_hosts (first 100)={:?}...", &busy_hosts[..100]);
            }
        }
        
        let placement = match self.compute_placement(job.num_workers, available_hosts) {
            Some(p) => p,
            None => {
                if DEBUG_MIGRATION {
                    println!("DEBUG   compute_placement returned None - insufficient capacity");
                }
                return false;
            }
        };

        if DEBUG_MIGRATION {
            let mut placement_hosts: Vec<usize> = placement.values().copied().collect();
            placement_hosts.sort();
            println!("DEBUG   computed_placement={:?}", placement_hosts);
        }

        // VALIDATION: Ensure all assigned hosts are actually free
        for (worker_id, &host_index) in &placement {
            if host_index < available_hosts.len() && available_hosts[host_index] {
                if DEBUG_MIGRATION {
                    println!("DEBUG   BUG DETECTED: host {} is busy but was selected for placement!", host_index);
                }
                panic!(
                    "[BlockScheduler] BUG: Assigned busy host {} to job {} worker {}. \
                    compute_placement returned an invalid placement!",
                    host_index, job.id, worker_id
                );
            }
        }

        // Apply the placement to the job
        for (worker_id, &host_index) in &placement {
            if let Some(worker) = job.workers.get_mut(worker_id) {
                worker.host_index = host_index;
            }
            job.worker_to_host.insert(*worker_id, host_index);
        }

        true
    }

    fn get_job_priority(&self, job: &MLJob) -> u64 {
        // FIFO: earlier submitted jobs have higher priority
        u64::MAX - job.submit_time_us
    }

    fn notify_job_completed(&mut self, _job_id: JobId, _completion_time_us: u64) {
        // BlockScheduler doesn't need to track completed jobs
    }

    fn get_next_job_to_schedule(&mut self) -> Option<JobId> {
        self.job_queue.front().copied()
    }

    fn enqueue_job(&mut self, job_id: JobId) {
        self.job_queue.push_back(job_id);
    }

    fn dequeue_job(&mut self) -> Option<JobId> {
        self.job_queue.pop_front()
    }
    
    fn has_queued_jobs(&self) -> bool {
        !self.job_queue.is_empty()
    }

    fn generate_placement_candidates<T: Topology>(
        &self,
        job: &MLJob,
        _topology: &T,
        available_hosts: &[bool],
        _max_candidates: usize,
    ) -> Vec<PlacementCandidate> {
        match self.compute_placement(job.num_workers, available_hosts) {
            Some(worker_to_host) => {
                vec![PlacementCandidate {
                    job_id: job.id,
                    worker_to_host,
                    compatibility_score: None,
                }]
            }
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_scheduler_creation() {
        let scheduler = BlockScheduler::new(48, 8);
        assert_eq!(scheduler.block_size(), 8);
        assert_eq!(scheduler.blocks_per_tor(), 6);
    }

    #[test]
    fn test_block_to_host_conversion() {
        let scheduler = BlockScheduler::new(48, 8);
        assert_eq!(scheduler.block_to_host(0), 0);
        assert_eq!(scheduler.block_to_host(1), 8);
        assert_eq!(scheduler.block_to_host(5), 40);
    }

    #[test]
    fn test_is_block_free() {
        let scheduler = BlockScheduler::new(48, 8);
        let mut host_busy = vec![false; 48];
        
        // Block 0 is free
        assert!(scheduler.is_block_free(0, &host_busy));
        
        // Mark one host in block 0 as busy
        host_busy[3] = true;
        assert!(!scheduler.is_block_free(0, &host_busy));
        
        // Block 1 is still free
        assert!(scheduler.is_block_free(1, &host_busy));
    }

    #[test]
    fn test_get_free_blocks() {
        let scheduler = BlockScheduler::new(48, 8);
        let mut host_busy = vec![false; 48];
        
        // All 6 blocks free
        let free = scheduler.get_free_blocks(&host_busy);
        assert_eq!(free.len(), 6);
        
        // Mark block 2 (hosts 16-23) as partially busy
        host_busy[20] = true;
        let free = scheduler.get_free_blocks(&host_busy);
        assert_eq!(free.len(), 5);
        assert!(!free.contains(&2));
    }

    #[test]
    fn test_single_tor_placement() {
        let scheduler = BlockScheduler::new(48, 8);
        let host_busy = vec![false; 48];
        
        // Request 16 workers (2 blocks)
        let placement = scheduler.compute_placement(16, &host_busy);
        assert!(placement.is_some());
        
        let mapping = placement.unwrap();
        assert_eq!(mapping.len(), 16);
        
        // Check workers are placed in contiguous blocks
        let hosts: Vec<usize> = (0..16).map(|w| mapping[&w]).collect();
        // First 8 workers should be in block 0, next 8 in block 1
        for i in 0..8 {
            assert!(hosts[i] < 8, "Worker {} should be in block 0", i);
        }
        for i in 8..16 {
            assert!(hosts[i] >= 8 && hosts[i] < 16, "Worker {} should be in block 1", i);
        }
    }

    #[test]
    #[should_panic(expected = "must be a multiple of block_size")]
    fn test_invalid_worker_count() {
        let scheduler = BlockScheduler::new(48, 8);
        let host_busy = vec![false; 48];
        
        // Request 10 workers (not a multiple of 8)
        scheduler.compute_placement(10, &host_busy);
    }

    #[test]
    fn test_multi_tor_placement() {
        let scheduler = BlockScheduler::new(48, 8);
        let mut host_busy = vec![false; 96]; // 2 ToRs
        
        // Fill first ToR except 1 block
        for i in 0..40 {
            host_busy[i] = true;
        }
        
        // Request 16 workers (2 blocks) - should span ToRs
        let placement = scheduler.compute_placement(16, &host_busy);
        assert!(placement.is_some());
        
        let mapping = placement.unwrap();
        assert_eq!(mapping.len(), 16);
    }

    #[test]
    fn test_insufficient_capacity() {
        let scheduler = BlockScheduler::new(48, 8);
        let mut host_busy = vec![true; 48]; // All busy
        
        // Make only 1 block free (8 hosts)
        for i in 0..8 {
            host_busy[i] = false;
        }
        
        // Request 16 workers (2 blocks) - should fail
        let placement = scheduler.compute_placement(16, &host_busy);
        assert!(placement.is_none());
    }
}
