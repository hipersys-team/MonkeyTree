use std::collections::{HashMap, VecDeque};

use crate::network::topology::Topology;
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{JobId, MLJob};
use crate::system_modules::cassini::PlacementCandidate;

/// Pod-aware block scheduler for rail-optimized topologies.
///
/// Prioritizes scheduling within a single pod:
/// 1. Find the pod with the fewest free blocks that can still fit the job (best-fit).
/// 2. If no single pod can fit the job, pick the pod with the most free blocks,
///    fill it entirely, then repeat the full algorithm for remaining workers.
///
/// All allocation happens at block (server) granularity.
#[derive(Debug)]
pub struct RailBlockScheduler {
    job_queue: VecDeque<JobId>,
    blocks_per_pod: usize,
    block_size: usize,
}

impl RailBlockScheduler {
    pub fn new(blocks_per_pod: usize, block_size: usize) -> Self {
        assert!(blocks_per_pod > 0 && block_size > 0);
        Self {
            job_queue: VecDeque::new(),
            blocks_per_pod,
            block_size,
        }
    }

    fn gpus_per_pod(&self) -> usize {
        self.blocks_per_pod * self.block_size
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
        let block_start = pod * self.blocks_per_pod;
        let block_end = block_start + self.blocks_per_pod;
        (block_start..block_end)
            .filter(|&b| self.is_block_free(b, host_busy))
            .collect()
    }

    pub(crate) fn compute_placement(
        &self,
        num_workers: usize,
        host_busy: &[bool],
    ) -> Option<HashMap<usize, usize>> {
        if num_workers % self.block_size != 0 {
            panic!(
                "Job requested {} workers, must be a multiple of block_size ({})",
                num_workers, self.block_size
            );
        }

        let num_blocks_needed = num_workers / self.block_size;
        let gpus_per_pod = self.gpus_per_pod();
        let num_pods = host_busy.len() / gpus_per_pod;

        let pod_free_counts: Vec<(usize, usize)> = (0..num_pods)
            .map(|p| (p, self.free_blocks_in_pod(p, host_busy).len()))
            .collect();

        let total_free: usize = pod_free_counts.iter().map(|(_, n)| n).sum();
        if total_free < num_blocks_needed {
            return None;
        }

        let mut worker_to_host = HashMap::new();
        let mut assigned = host_busy.to_vec();
        let mut worker_id: usize = 0;
        let mut remaining_blocks = num_blocks_needed;

        while remaining_blocks > 0 {
            let current_counts: Vec<(usize, usize)> = (0..num_pods)
                .map(|p| (p, self.free_blocks_in_pod(p, &assigned).len()))
                .collect();

            // Best-fit: smallest pod that can still fit all remaining blocks
            let mut fitting: Vec<(usize, usize)> = current_counts
                .iter()
                .filter(|(_, free)| *free >= remaining_blocks)
                .copied()
                .collect();

            if !fitting.is_empty() {
                fitting.sort_by_key(|&(_, free)| free);
                let best_pod = fitting[0].0;
                let free_blocks = self.free_blocks_in_pod(best_pod, &assigned);
                for &block_idx in free_blocks.iter().take(remaining_blocks) {
                    for host_idx in self.block_host_range(block_idx) {
                        worker_to_host.insert(worker_id, host_idx);
                        assigned[host_idx] = true;
                        worker_id += 1;
                    }
                }
                remaining_blocks = 0;
            } else {
                // No pod can fit remaining — fill the pod with the most free blocks
                let mut by_free: Vec<(usize, usize)> = current_counts
                    .iter()
                    .filter(|(_, free)| *free > 0)
                    .copied()
                    .collect();

                if by_free.is_empty() {
                    panic!("Ran out of capacity during multi-pod placement");
                }

                by_free.sort_by_key(|&(_, free)| std::cmp::Reverse(free));
                let (best_pod, free_count) = by_free[0];
                let free_blocks = self.free_blocks_in_pod(best_pod, &assigned);

                for &block_idx in free_blocks.iter().take(free_count) {
                    for host_idx in self.block_host_range(block_idx) {
                        worker_to_host.insert(worker_id, host_idx);
                        assigned[host_idx] = true;
                        worker_id += 1;
                    }
                    remaining_blocks -= 1;
                }
            }
        }

        Some(worker_to_host)
    }
}

impl JobScheduler for RailBlockScheduler {
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
                    "[RailBlockScheduler] BUG: Assigned busy host {} to job {} worker {}",
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
