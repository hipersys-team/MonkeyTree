//! Pod-aware block scheduler for fat-tree topologies.
//!
//! Prioritizes scheduling within a single pod:
//! 1. Find the pod with the smallest number of free slots that can still fit the job → best-fit
//! 2. If no single pod can fit the job, pick the pod with the most free slots,
//!    fill as many workers as possible there, then repeat for remaining workers.
//!
//! Within each pod, delegates to block-level allocation (same as BlockScheduler).

use std::collections::{HashMap, VecDeque};

use crate::network::topology::Topology;
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{JobId, MLJob};
use crate::system_modules::cassini::PlacementCandidate;

use super::block::BlockScheduler;

#[derive(Debug)]
pub struct FatTreeBlockScheduler {
    job_queue: VecDeque<JobId>,
    hosts_per_tor: usize,
    tors_per_pod: usize,
    block_size: usize,
    /// Inner BlockScheduler used for within-pod placement decisions.
    inner: BlockScheduler,
}

impl FatTreeBlockScheduler {
    pub fn new(hosts_per_tor: usize, tors_per_pod: usize, block_size: usize) -> Self {
        assert!(hosts_per_tor > 0 && tors_per_pod > 0 && block_size > 0);
        assert!(
            hosts_per_tor % block_size == 0,
            "hosts_per_tor ({}) must be divisible by block_size ({})",
            hosts_per_tor, block_size
        );
        Self {
            job_queue: VecDeque::new(),
            hosts_per_tor,
            tors_per_pod,
            block_size,
            inner: BlockScheduler::new(hosts_per_tor, block_size),
        }
    }

    fn hosts_per_pod(&self) -> usize {
        self.hosts_per_tor * self.tors_per_pod
    }

    fn blocks_per_tor(&self) -> usize {
        self.hosts_per_tor / self.block_size
    }

    fn block_host_range(&self, block_idx: usize) -> std::ops::Range<usize> {
        let start = block_idx * self.block_size;
        start..start + self.block_size
    }

    fn is_block_free(&self, block_idx: usize, host_busy: &[bool]) -> bool {
        let range = self.block_host_range(block_idx);
        if range.end > host_busy.len() {
            return false;
        }
        range.into_iter().all(|h| !host_busy[h])
    }

    fn free_blocks_in_pod(&self, pod: usize, host_busy: &[bool]) -> Vec<usize> {
        let blocks_per_pod = self.tors_per_pod * self.blocks_per_tor();
        let block_start = pod * blocks_per_pod;
        let block_end = block_start + blocks_per_pod;
        (block_start..block_end)
            .filter(|&b| self.is_block_free(b, host_busy))
            .collect()
    }

    /// Extracts a pod's slice of `host_busy` (length = hosts_per_pod).
    fn pod_slice(&self, pod: usize, host_busy: &[bool]) -> Vec<bool> {
        let hosts_per_pod = self.hosts_per_pod();
        let start = pod * hosts_per_pod;
        let end = start + hosts_per_pod;
        host_busy[start..end].to_vec()
    }

    /// Places `num_workers` within a single pod using the inner BlockScheduler.
    /// Returns a mapping of (worker_id in [worker_id_base, worker_id_base+num_workers))
    /// to global host indices, and marks those hosts busy in `assigned`.
    fn place_in_pod(
        &self,
        pod: usize,
        num_workers: usize,
        assigned: &mut [bool],
        worker_to_host: &mut HashMap<usize, usize>,
        worker_id_base: &mut usize,
    ) -> bool {
        let hosts_per_pod = self.hosts_per_pod();
        let pod_offset = pod * hosts_per_pod;
        let pod_busy = self.pod_slice(pod, assigned);

        let pod_placement = match self.inner.compute_placement(num_workers, &pod_busy) {
            Some(p) => p,
            None => return false,
        };

        for (local_worker, &local_host) in pod_placement.iter() {
            let global_host = pod_offset + local_host;
            worker_to_host.insert(*worker_id_base + local_worker, global_host);
            assigned[global_host] = true;
        }
        *worker_id_base += num_workers;
        true
    }

    fn compute_placement(&self, num_workers: usize, host_busy: &[bool]) -> Option<HashMap<usize, usize>> {
        if num_workers % self.block_size != 0 {
            panic!(
                "Job requested {} workers, must be a multiple of block_size ({})",
                num_workers, self.block_size
            );
        }

        let num_blocks_needed = num_workers / self.block_size;
        let hosts_per_pod = self.hosts_per_pod();
        let num_pods = host_busy.len() / hosts_per_pod;

        let pod_free_counts: Vec<(usize, usize)> = (0..num_pods)
            .map(|p| (p, self.free_blocks_in_pod(p, host_busy).len()))
            .collect();

        let total_free: usize = pod_free_counts.iter().map(|(_, n)| n).sum();
        if total_free < num_blocks_needed {
            return None;
        }

        let mut worker_to_host = HashMap::new();
        let mut assigned = host_busy.to_vec();
        let mut worker_id_base = 0;

        // Phase 1: try best-fit at the pod level — smallest pod that can fit the whole job.
        let mut fitting: Vec<(usize, usize)> = pod_free_counts
            .iter()
            .filter(|(_, free)| *free >= num_blocks_needed)
            .copied()
            .collect();

        if !fitting.is_empty() {
            fitting.sort_by_key(|&(_, free)| free);
            let best_pod = fitting[0].0;
            // Delegate within-pod placement to BlockScheduler (ToR-level best-fit + greedy).
            let ok = self.place_in_pod(
                best_pod,
                num_workers,
                &mut assigned,
                &mut worker_to_host,
                &mut worker_id_base,
            );
            assert!(ok, "BlockScheduler failed to place in a pod that was reported as fitting");
            return Some(worker_to_host);
        }

        // Phase 2: no single pod fits — pick pod with most free blocks, place as many workers
        // there as possible (in multiples of block_size), then repeat for the remainder.
        let mut remaining_workers = num_workers;

        while remaining_workers > 0 {
            let mut current_counts: Vec<(usize, usize)> = (0..num_pods)
                .map(|p| (p, self.free_blocks_in_pod(p, &assigned).len()))
                .filter(|(_, free)| *free > 0)
                .collect();

            if current_counts.is_empty() {
                panic!("Ran out of capacity during multi-pod placement");
            }

            current_counts.sort_by_key(|&(_, free)| std::cmp::Reverse(free));
            let (best_pod, free_blocks) = current_counts[0];

            let free_workers_here = free_blocks * self.block_size;
            let to_place = remaining_workers.min(free_workers_here);

            let ok = self.place_in_pod(
                best_pod,
                to_place,
                &mut assigned,
                &mut worker_to_host,
                &mut worker_id_base,
            );
            assert!(ok, "BlockScheduler failed to place {} workers in pod {} with {} free blocks", to_place, best_pod, free_blocks);

            remaining_workers -= to_place;
        }

        Some(worker_to_host)
    }
}

impl JobScheduler for FatTreeBlockScheduler {
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

        for (worker_id, &host_index) in &placement {
            if host_index < available_hosts.len() && available_hosts[host_index] {
                panic!(
                    "[FatTreeBlockScheduler] BUG: Assigned busy host {} to job {} worker {}",
                    host_index, job.id, worker_id
                );
            }
        }

        for (worker_id, &host_index) in &placement {
            if let Some(worker) = job.workers.get_mut(worker_id) {
                worker.host_index = host_index;
            }
            job.worker_to_host.insert(*worker_id, host_index);
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
