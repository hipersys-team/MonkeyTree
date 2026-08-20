use std::collections::{HashMap, VecDeque};

use crate::network::topology::Topology;
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{JobId, MLJob};
use crate::system_modules::cassini::PlacementCandidate;

/// A cluster-aware job scheduler that places workers on ToRs using a best-fit strategy.
///
/// This scheduler processes jobs in FIFO order and will not proceed to the next job
/// until the current one is scheduled. The placement algorithm:
///
/// 1. Check if there is enough total capacity in the cluster for the job
/// 2. If yes, scan all ToRs to find ones that can fully accommodate the job
/// 3. Pick the ToR with the least free capacity that can fit the job (best-fit)
/// 4. If no single ToR can fit the job, greedily fill ToRs starting with the one
///    with the most free capacity, repeating until all workers are placed
#[derive(Debug)]
pub struct ClusterScheduler {
    /// Queue of jobs waiting to be scheduled (FIFO order)
    job_queue: VecDeque<JobId>,
    /// Number of hosts per ToR switch (used to compute ToR assignments)
    hosts_per_tor: usize,
}

impl ClusterScheduler {
    /// Creates a new ClusterScheduler with the specified hosts-per-ToR configuration.
    ///
    /// # Arguments
    /// * `hosts_per_tor` - Number of hosts connected to each ToR switch.
    ///   Host indices 0..hosts_per_tor belong to ToR 0,
    ///   hosts_per_tor..2*hosts_per_tor belong to ToR 1, etc.
    pub fn new(hosts_per_tor: usize) -> Self {
        assert!(hosts_per_tor > 0, "hosts_per_tor must be positive");
        Self {
            job_queue: VecDeque::new(),
            hosts_per_tor,
        }
    }

    /// Returns the range of host indices for a given ToR.
    #[inline]
    fn tor_host_range(&self, tor_index: usize) -> std::ops::Range<usize> {
        let start = tor_index * self.hosts_per_tor;
        let end = start + self.hosts_per_tor;
        start..end
    }

    /// Computes free capacity per ToR from the available_hosts bitmap.
    /// Returns a vec of (tor_index, free_count) for ToRs with at least one free host.
    fn compute_tor_capacities(&self, available_hosts: &[bool]) -> Vec<(usize, usize)> {
        let num_tors = (available_hosts.len() + self.hosts_per_tor - 1) / self.hosts_per_tor;
        let mut tor_capacities = Vec::with_capacity(num_tors);

        for tor_idx in 0..num_tors {
            let range = self.tor_host_range(tor_idx);
            let end = range.end.min(available_hosts.len());
            let free_count = (range.start..end)
                .filter(|&h| !available_hosts[h])
                .count();
            if free_count > 0 {
                tor_capacities.push((tor_idx, free_count));
            }
        }

        tor_capacities
    }

    /// Gets the free host indices for a specific ToR.
    fn get_free_hosts_in_tor(&self, tor_index: usize, available_hosts: &[bool]) -> Vec<usize> {
        let range = self.tor_host_range(tor_index);
        let end = range.end.min(available_hosts.len());
        (range.start..end)
            .filter(|&h| !available_hosts[h])
            .collect()
    }

    /// Attempts to place workers using the cluster scheduling algorithm.
    /// Returns Some(worker_to_host mapping) if successful, None if not enough capacity.
    fn compute_placement(&self, num_workers: usize, available_hosts: &[bool]) -> Option<HashMap<usize, usize>> {
        // First check total cluster capacity
        let total_free = available_hosts.iter().filter(|&&busy| !busy).count();
        if total_free < num_workers {
            return None; // Not enough total capacity
        }

        let tor_capacities = self.compute_tor_capacities(available_hosts);
        if tor_capacities.is_empty() {
            return None;
        }

        let mut worker_to_host = HashMap::new();
        let mut remaining_workers = num_workers;

        // Try to find a single ToR that can fit all workers (best-fit: least free capacity)
        let mut fitting_tors: Vec<_> = tor_capacities
            .iter()
            .filter(|&&(_, free)| free >= num_workers)
            .copied()
            .collect();

        if !fitting_tors.is_empty() {
            // Sort by free capacity ascending (least free first = best fit)
            fitting_tors.sort_by_key(|&(_, free)| free);
            let (best_tor, _) = fitting_tors[0];

            // Place all workers on this ToR
            let free_hosts = self.get_free_hosts_in_tor(best_tor, available_hosts);
            for (worker_id, &host_index) in free_hosts.iter().take(num_workers).enumerate() {
                worker_to_host.insert(worker_id, host_index);
            }

            return Some(worker_to_host);
        }

        // No single ToR can fit the job - use greedy multi-ToR placement
        // Track which hosts we've assigned (to handle the greedy iteration)
        let mut assigned_hosts: Vec<bool> = available_hosts.to_vec();
        let mut next_worker_id = 0;

        while remaining_workers > 0 {
            // Recompute capacities based on current assignments
            let mut current_capacities = self.compute_tor_capacities(&assigned_hosts);
            if current_capacities.is_empty() {
                // This shouldn't happen if we checked total capacity, but be safe
                panic!("No ToRs available to place workers");
            }

            // Sort by free capacity descending (largest free first)
            current_capacities.sort_by_key(|&(_, free)| std::cmp::Reverse(free));
            let (best_tor, free_count) = current_capacities[0];

            // Place as many workers as possible on this ToR
            let workers_to_place = remaining_workers.min(free_count);
            let free_hosts = self.get_free_hosts_in_tor(best_tor, &assigned_hosts);

            for &host_index in free_hosts.iter().take(workers_to_place) {
                worker_to_host.insert(next_worker_id, host_index);
                assigned_hosts[host_index] = true; // Mark as assigned
                next_worker_id += 1;
                remaining_workers -= 1;
            }
        }

        Some(worker_to_host)
    }
}

impl Default for ClusterScheduler {
    fn default() -> Self {
        // Default to 4 hosts per ToR (common configuration)
        Self::new(4)
    }
}

impl JobScheduler for ClusterScheduler {
    fn try_schedule_job<T: Topology>(
        &mut self,
        job: &mut MLJob,
        _topology: &T,
        available_hosts: &[bool],
    ) -> bool {
        let placement = match self.compute_placement(job.num_workers, available_hosts) {
            Some(p) => p,
            None => return false,
        };

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
        // ClusterScheduler doesn't need to track completed jobs
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
        // Generate a single candidate using our placement algorithm
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