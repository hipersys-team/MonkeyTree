//! Block-based scheduler with multiple placement candidate generation for Cassini.
//!
//! This scheduler extends BlockScheduler to generate multiple placement candidates
//! by varying decisions at each stage of the scheduling process:
//! - Which ToR to place the job on (when multiple ToRs can fit)
//! - Which blocks within a ToR to use (first-fit vs last-fit)
//! - For multi-ToR placement, the order of ToR selection
//!
//! This enables Cassini to evaluate compatibility across different placements
//! and select the one with the best network interleaving.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::network::topology::Topology;
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{JobId, MLJob};
use crate::system_modules::cassini::PlacementCandidate;

/// Maximum number of placement candidates to generate
const MAX_CANDIDATES: usize = 10;

/// Block-based job scheduler that generates multiple placement candidates for Cassini.
///
/// Unlike BlockScheduler which only returns the "optimal" placement,
/// this scheduler explores variations in ToR selection and block ordering
/// to give Cassini options to evaluate for network compatibility.
#[derive(Debug)]
pub struct CassiniBlockScheduler {
    /// Queue of jobs waiting to be scheduled (FIFO order)
    job_queue: VecDeque<JobId>,
    /// Number of hosts per ToR switch
    hosts_per_tor: usize,
    /// Number of hosts per block (e.g., 8 GPUs per server)
    block_size: usize,
}

impl CassiniBlockScheduler {
    /// Creates a new CassiniBlockScheduler.
    ///
    /// # Arguments
    /// * `hosts_per_tor` - Number of hosts (GPUs) per ToR switch
    /// * `block_size` - Number of hosts per block (default 8)
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

    /// Returns the range of host indices for a given block
    #[inline]
    fn block_host_range(&self, block_idx: usize) -> std::ops::Range<usize> {
        let start = self.block_to_host(block_idx);
        start..start + self.block_size
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

    /// Creates a worker-to-host mapping from a list of blocks
    fn blocks_to_placement(&self, job_id: JobId, blocks: &[usize]) -> PlacementCandidate {
        let mut worker_to_host = HashMap::new();
        let mut worker_id = 0;
        
        for &block_idx in blocks {
            for host_idx in self.block_host_range(block_idx) {
                worker_to_host.insert(worker_id, host_idx);
                worker_id += 1;
            }
        }
        
        PlacementCandidate {
            job_id,
            worker_to_host,
            compatibility_score: None,
        }
    }

    /// Generates multiple placement candidates for single-ToR placement
    /// Varies: which ToR, and first-fit vs last-fit within each ToR
    fn generate_single_tor_candidates(
        &self,
        job_id: JobId,
        num_blocks_needed: usize,
        fitting_tors: &[(usize, usize)], // (tor_idx, free_block_count)
        host_busy: &[bool],
        max_candidates: usize,
    ) -> Vec<PlacementCandidate> {
        let mut candidates = Vec::new();
        let mut seen_placements: HashSet<Vec<usize>> = HashSet::new();
        
        // Try each fitting ToR
        for &(tor_idx, _free_count) in fitting_tors {
            if candidates.len() >= max_candidates {
                break;
            }
            
            let free_blocks = self.get_free_blocks_in_tor(tor_idx, host_busy);
            if free_blocks.len() < num_blocks_needed {
                continue;
            }
            
            // Variation 1: First-fit (use first N blocks)
            let first_fit: Vec<usize> = free_blocks.iter().take(num_blocks_needed).copied().collect();
            if !seen_placements.contains(&first_fit) {
                seen_placements.insert(first_fit.clone());
                candidates.push(self.blocks_to_placement(job_id, &first_fit));
            }
            
            if candidates.len() >= max_candidates {
                break;
            }
            
            // Variation 2: Last-fit (use last N blocks)
            let last_fit: Vec<usize> = free_blocks.iter().rev().take(num_blocks_needed).rev().copied().collect();
            if !seen_placements.contains(&last_fit) {
                seen_placements.insert(last_fit.clone());
                candidates.push(self.blocks_to_placement(job_id, &last_fit));
            }
            
            if candidates.len() >= max_candidates {
                break;
            }
            
            // Variation 3: Middle blocks (skip first block if possible)
            if free_blocks.len() > num_blocks_needed {
                let middle_fit: Vec<usize> = free_blocks.iter()
                    .skip(1)
                    .take(num_blocks_needed)
                    .copied()
                    .collect();
                if middle_fit.len() == num_blocks_needed && !seen_placements.contains(&middle_fit) {
                    seen_placements.insert(middle_fit.clone());
                    candidates.push(self.blocks_to_placement(job_id, &middle_fit));
                }
            }
        }
        
        candidates
    }

    /// Generates a multi-ToR placement with a specific ToR ordering
    fn generate_multi_tor_placement(
        &self,
        job_id: JobId,
        num_blocks_needed: usize,
        tor_order: &[(usize, usize)], // (tor_idx, free_count) in desired order
        host_busy: &[bool],
        use_last_fit: bool,
    ) -> Option<PlacementCandidate> {
        let mut selected_blocks = Vec::new();
        let mut assigned = host_busy.to_vec();
        let mut remaining = num_blocks_needed;
        
        for &(tor_idx, _) in tor_order {
            if remaining == 0 {
                break;
            }
            
            let mut free_blocks = self.get_free_blocks_in_tor(tor_idx, &assigned);
            if free_blocks.is_empty() {
                continue;
            }
            
            // Vary block selection within ToR
            if use_last_fit {
                free_blocks.reverse();
            }
            
            let to_take = remaining.min(free_blocks.len());
            for &block_idx in free_blocks.iter().take(to_take) {
                selected_blocks.push(block_idx);
                for host_idx in self.block_host_range(block_idx) {
                    assigned[host_idx] = true;
                }
                remaining -= 1;
            }
        }
        
        if remaining == 0 {
            // Sort blocks for consistent worker ordering
            selected_blocks.sort();
            Some(self.blocks_to_placement(job_id, &selected_blocks))
        } else {
            None
        }
    }

    /// Generates multiple placement candidates for multi-ToR placement
    /// Varies: ToR ordering (most capacity first vs round-robin) and block selection
    fn generate_multi_tor_candidates(
        &self,
        job_id: JobId,
        num_blocks_needed: usize,
        tor_capacities: &[(usize, usize)],
        host_busy: &[bool],
        max_candidates: usize,
    ) -> Vec<PlacementCandidate> {
        let mut candidates = Vec::new();
        let mut seen_placements: HashSet<Vec<usize>> = HashSet::new();
        
        // Strategy 1: Greedy (most capacity first) + first-fit
        let mut greedy_order = tor_capacities.to_vec();
        greedy_order.sort_by_key(|&(_, free)| std::cmp::Reverse(free));
        
        if let Some(candidate) = self.generate_multi_tor_placement(
            job_id, num_blocks_needed, &greedy_order, host_busy, false
        ) {
            let blocks: Vec<usize> = self.extract_blocks_from_candidate(&candidate);
            if !seen_placements.contains(&blocks) {
                seen_placements.insert(blocks);
                candidates.push(candidate);
            }
        }
        
        if candidates.len() >= max_candidates {
            return candidates;
        }
        
        // Strategy 2: Greedy + last-fit
        if let Some(candidate) = self.generate_multi_tor_placement(
            job_id, num_blocks_needed, &greedy_order, host_busy, true
        ) {
            let blocks = self.extract_blocks_from_candidate(&candidate);
            if !seen_placements.contains(&blocks) {
                seen_placements.insert(blocks);
                candidates.push(candidate);
            }
        }
        
        if candidates.len() >= max_candidates {
            return candidates;
        }
        
        // Strategy 3: Spread across ToRs (round-robin style)
        // Try to distribute blocks more evenly
        if let Some(candidate) = self.generate_spread_placement(
            job_id, num_blocks_needed, &tor_capacities, host_busy
        ) {
            let blocks = self.extract_blocks_from_candidate(&candidate);
            if !seen_placements.contains(&blocks) {
                seen_placements.insert(blocks);
                candidates.push(candidate);
            }
        }
        
        if candidates.len() >= max_candidates {
            return candidates;
        }
        
        // Strategy 4-N: Try different ToR orderings
        // Vary which ToR gets priority by rotating the order
        for rotation in 1..tor_capacities.len().min(max_candidates - candidates.len() + 1) {
            let mut rotated = tor_capacities.to_vec();
            rotated.rotate_left(rotation);
            
            if let Some(candidate) = self.generate_multi_tor_placement(
                job_id, num_blocks_needed, &rotated, host_busy, false
            ) {
                let blocks = self.extract_blocks_from_candidate(&candidate);
                if !seen_placements.contains(&blocks) {
                    seen_placements.insert(blocks);
                    candidates.push(candidate);
                    
                    if candidates.len() >= max_candidates {
                        break;
                    }
                }
            }
        }
        
        candidates
    }

    /// Generates a spread placement that distributes blocks across ToRs more evenly
    fn generate_spread_placement(
        &self,
        job_id: JobId,
        num_blocks_needed: usize,
        tor_capacities: &[(usize, usize)],
        host_busy: &[bool],
    ) -> Option<PlacementCandidate> {
        if tor_capacities.is_empty() {
            return None;
        }
        
        let mut selected_blocks = Vec::new();
        let mut assigned = host_busy.to_vec();
        let mut remaining = num_blocks_needed;
        
        // Round-robin: take one block from each ToR in turn
        let mut tor_free_blocks: Vec<(usize, Vec<usize>)> = tor_capacities
            .iter()
            .map(|&(tor_idx, _)| (tor_idx, self.get_free_blocks_in_tor(tor_idx, &assigned)))
            .filter(|(_, blocks)| !blocks.is_empty())
            .collect();
        
        while remaining > 0 && !tor_free_blocks.is_empty() {
            let mut made_progress = false;
            
            for (_, free_blocks) in tor_free_blocks.iter_mut() {
                if remaining == 0 {
                    break;
                }
                
                if let Some(block_idx) = free_blocks.pop() {
                    // Check if still free
                    if self.is_block_free(block_idx, &assigned) {
                        selected_blocks.push(block_idx);
                        for host_idx in self.block_host_range(block_idx) {
                            assigned[host_idx] = true;
                        }
                        remaining -= 1;
                        made_progress = true;
                    }
                }
            }
            
            if !made_progress {
                break;
            }
            
            // Remove exhausted ToRs
            tor_free_blocks.retain(|(_, blocks)| !blocks.is_empty());
        }
        
        if remaining == 0 {
            selected_blocks.sort();
            Some(self.blocks_to_placement(job_id, &selected_blocks))
        } else {
            None
        }
    }

    /// Extracts block indices from a placement candidate for deduplication
    fn extract_blocks_from_candidate(&self, candidate: &PlacementCandidate) -> Vec<usize> {
        let mut blocks: HashSet<usize> = HashSet::new();
        for &host_idx in candidate.worker_to_host.values() {
            blocks.insert(host_idx / self.block_size);
        }
        let mut blocks_vec: Vec<usize> = blocks.into_iter().collect();
        blocks_vec.sort();
        blocks_vec
    }

    /// Generates placement candidates without requiring a topology reference.
    /// This is the core implementation used by generate_placement_candidates.
    pub fn generate_candidates_for_job(
        &self,
        job_id: JobId,
        num_workers: usize,
        host_busy: &[bool],
        max_candidates: usize,
    ) -> Vec<PlacementCandidate> {
        let max_to_generate = max_candidates.min(MAX_CANDIDATES);
        
        if num_workers % self.block_size != 0 {
            return Vec::new();
        }

        let num_blocks_needed = num_workers / self.block_size;
        let free_blocks = self.get_free_blocks(host_busy);
        
        if free_blocks.len() < num_blocks_needed {
            return Vec::new();
        }

        let tor_capacities = self.compute_tor_block_capacities(host_busy);
        if tor_capacities.is_empty() {
            return Vec::new();
        }

        // Find ToRs that can fit the entire job
        let mut fitting_tors: Vec<_> = tor_capacities
            .iter()
            .filter(|&&(_, free)| free >= num_blocks_needed)
            .copied()
            .collect();

        if !fitting_tors.is_empty() {
            // Sort by free capacity (least free first = best fit)
            fitting_tors.sort_by_key(|&(_, free)| free);
            return self.generate_single_tor_candidates(
                job_id,
                num_blocks_needed,
                &fitting_tors,
                host_busy,
                max_to_generate,
            );
        }

        // Multi-ToR placement needed
        self.generate_multi_tor_candidates(
            job_id,
            num_blocks_needed,
            &tor_capacities,
            host_busy,
            max_to_generate,
        )
    }

    /// Computes the primary (optimal) placement, matching BlockScheduler behavior
    fn compute_placement(&self, num_workers: usize, host_busy: &[bool]) -> Option<HashMap<usize, usize>> {
        if num_workers % self.block_size != 0 {
            panic!(
                "Job requested {} workers, but must be a multiple of block_size ({})",
                num_workers, self.block_size
            );
        }

        let num_blocks_needed = num_workers / self.block_size;
        let free_blocks = self.get_free_blocks(host_busy);
        
        if free_blocks.len() < num_blocks_needed {
            return None;
        }

        let tor_capacities = self.compute_tor_block_capacities(host_busy);
        if tor_capacities.is_empty() {
            return None;
        }

        // Try single-ToR placement first
        let fitting_tors: Vec<_> = tor_capacities
            .iter()
            .filter(|&&(_, free)| free >= num_blocks_needed)
            .copied()
            .collect();

        if !fitting_tors.is_empty() {
            // Best-fit: ToR with least free capacity that can still fit
            let (best_tor, _) = *fitting_tors.iter()
                .min_by_key(|&&(_, free)| free)
                .unwrap();
            
            let free_blocks = self.get_free_blocks_in_tor(best_tor, host_busy);
            let mut worker_to_host = HashMap::new();
            let mut worker_id = 0;
            
            for &block_idx in free_blocks.iter().take(num_blocks_needed) {
                for host_idx in self.block_host_range(block_idx) {
                    worker_to_host.insert(worker_id, host_idx);
                    worker_id += 1;
                }
            }
            
            return Some(worker_to_host);
        }

        // Multi-ToR greedy placement
        let mut assigned = host_busy.to_vec();
        let mut worker_to_host = HashMap::new();
        let mut worker_id = 0;
        let mut remaining = num_blocks_needed;

        while remaining > 0 {
            let mut current_capacities = self.compute_tor_block_capacities(&assigned);
            if current_capacities.is_empty() {
                return None;
            }
            
            current_capacities.sort_by_key(|&(_, free)| std::cmp::Reverse(free));
            let (best_tor, free_count) = current_capacities[0];
            let blocks_to_place = remaining.min(free_count);
            
            let free_blocks = self.get_free_blocks_in_tor(best_tor, &assigned);
            for &block_idx in free_blocks.iter().take(blocks_to_place) {
                for host_idx in self.block_host_range(block_idx) {
                    worker_to_host.insert(worker_id, host_idx);
                    assigned[host_idx] = true;
                    worker_id += 1;
                }
                remaining -= 1;
            }
        }

        Some(worker_to_host)
    }
}

impl Default for CassiniBlockScheduler {
    fn default() -> Self {
        Self::new(48, 8)
    }
}

impl CassiniBlockScheduler {
    /// Scores a placement candidate based on link contention.
    /// 
    /// For each ToR used by the placement, we count how many OTHER jobs
    /// (represented by busy hosts not in this placement) share that ToR.
    /// Score per ToR = 1 / (1 + num_other_jobs_on_tor)
    /// 
    /// Returns the average score across all ToRs, plus a locality bonus
    /// for placements that span fewer ToRs.
    fn score_placement(&self, candidate: &PlacementCandidate, host_busy: &[bool]) -> f64 {
        use std::collections::HashSet;
        
        // Collect hosts used by this candidate
        let candidate_hosts: HashSet<usize> = candidate.worker_to_host.values().cloned().collect();
        
        // Group candidate's hosts by ToR
        let mut tors_used: HashSet<usize> = HashSet::new();
        for &host in candidate.worker_to_host.values() {
            let tor = host / self.hosts_per_tor;
            tors_used.insert(tor);
        }
        
        if tors_used.is_empty() {
            return 0.0;
        }
        
        // For each ToR used, count how many OTHER busy hosts are on it
        // We treat each "other busy host" as representing contention from another job
        let mut total_score = 0.0;
        for &tor in &tors_used {
            let tor_start = tor * self.hosts_per_tor;
            let tor_end = (tor_start + self.hosts_per_tor).min(host_busy.len());
            
            // Count other busy hosts on this ToR (not part of this candidate)
            let other_busy_count = (tor_start..tor_end)
                .filter(|&h| host_busy[h] && !candidate_hosts.contains(&h))
                .count();
            
            // Score = 1 / (1 + other_busy_count)
            // - No other busy hosts: score = 1.0 (full compatibility)
            // - 1 other busy host: score = 0.5
            // - 2 other busy hosts: score = 0.33
            // etc.
            let score = 1.0 / (1 + other_busy_count) as f64;
            total_score += score;
        }
        
        // Average score across ToRs used
        let avg_score = total_score / tors_used.len() as f64;
        
        // Add locality bonus: prefer placements using fewer ToRs
        // This rewards keeping workers together
        let max_tors = (candidate.worker_to_host.len() + self.hosts_per_tor - 1) / self.hosts_per_tor;
        let locality_bonus = if max_tors > 0 {
            (max_tors as f64 - tors_used.len() as f64) / max_tors as f64 * 0.1
        } else {
            0.0
        };
        
        avg_score + locality_bonus
    }
}

impl JobScheduler for CassiniBlockScheduler {
    fn try_schedule_job<T: Topology>(
        &mut self,
        job: &mut MLJob,
        _topology: &T,
        host_busy: &[bool],
    ) -> bool {
        // Generate multiple placement candidates
        let candidates = self.generate_candidates_for_job(job.id, job.num_workers, host_busy, MAX_CANDIDATES);
        
        if candidates.is_empty() {
            return false;
        }
        
        // Score each candidate and pick the best
        let best_candidate = candidates
            .into_iter()
            .map(|c| {
                let score = self.score_placement(&c, host_busy);
                (c, score)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(c, _)| c);
        
        let candidate = match best_candidate {
            Some(c) => c,
            None => return false,
        };

        // Validate placement - no busy hosts should be assigned
        for (&_, &host_index) in &candidate.worker_to_host {
            if host_index < host_busy.len() && host_busy[host_index] {
                panic!(
                    "[CassiniBlockScheduler] BUG: Assigned busy host {} to job {}",
                    host_index, job.id
                );
            }
        }

        // Apply the chosen placement
        for (&worker_id, &host_index) in &candidate.worker_to_host {
            if let Some(worker) = job.workers.get_mut(&worker_id) {
                worker.host_index = host_index;
            }
            job.worker_to_host.insert(worker_id, host_index);
        }

        true
    }

    fn get_job_priority(&self, job: &MLJob) -> u64 {
        u64::MAX - job.submit_time_us
    }

    fn notify_job_completed(&mut self, _job_id: JobId, _completion_time_us: u64) {}

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
        max_candidates: usize,
    ) -> Vec<PlacementCandidate> {
        self.generate_candidates_for_job(job.id, job.num_workers, available_hosts, max_candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = CassiniBlockScheduler::new(48, 8);
        assert_eq!(scheduler.block_size(), 8);
        assert_eq!(scheduler.blocks_per_tor(), 6);
    }

    #[test]
    fn test_single_tor_multiple_candidates() {
        let scheduler = CassiniBlockScheduler::new(48, 8);
        let host_busy = vec![false; 48]; // Single ToR, all free
        
        // 2 blocks needed (16 workers)
        let candidates = scheduler.generate_candidates_for_job(0, 16, &host_busy, 10);
        
        // Should generate multiple candidates (first-fit, last-fit, middle)
        assert!(candidates.len() >= 2, "Expected at least 2 candidates, got {}", candidates.len());
        
        // All candidates should have 16 workers
        for candidate in &candidates {
            assert_eq!(candidate.worker_to_host.len(), 16);
        }
    }

    #[test]
    fn test_multi_tor_multiple_candidates() {
        let scheduler = CassiniBlockScheduler::new(48, 8);
        let mut host_busy = vec![false; 96]; // 2 ToRs
        
        // Fill most of first ToR
        for i in 0..40 {
            host_busy[i] = true;
        }
        
        // Need 3 blocks (24 workers) - must span ToRs
        let candidates = scheduler.generate_candidates_for_job(0, 24, &host_busy, 10);
        
        assert!(!candidates.is_empty(), "Expected at least 1 candidate");
        
        for candidate in &candidates {
            assert_eq!(candidate.worker_to_host.len(), 24);
        }
    }

    #[test]
    fn test_candidates_are_unique() {
        let scheduler = CassiniBlockScheduler::new(48, 8);
        let host_busy = vec![false; 96]; // 2 ToRs, all free
        
        let candidates = scheduler.generate_candidates_for_job(0, 16, &host_busy, 10);
        
        // Check all candidates are unique
        let mut seen: HashSet<Vec<usize>> = HashSet::new();
        for candidate in &candidates {
            let mut hosts: Vec<usize> = candidate.worker_to_host.values().copied().collect();
            hosts.sort();
            assert!(seen.insert(hosts.clone()), "Duplicate placement found: {:?}", hosts);
        }
    }

    #[test]
    fn test_insufficient_capacity() {
        let scheduler = CassiniBlockScheduler::new(48, 8);
        let host_busy = vec![true; 48]; // All busy
        
        let candidates = scheduler.generate_candidates_for_job(0, 16, &host_busy, 10);
        
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_respects_max_candidates() {
        let scheduler = CassiniBlockScheduler::new(48, 8);
        let host_busy = vec![false; 192]; // 4 ToRs
        
        // Request only 3 candidates (1 block = 8 workers)
        let candidates = scheduler.generate_candidates_for_job(0, 8, &host_busy, 3);
        assert!(candidates.len() <= 3);
        
        // Request many candidates
        let candidates = scheduler.generate_candidates_for_job(0, 8, &host_busy, 10);
        assert!(candidates.len() <= 10);
    }
}
