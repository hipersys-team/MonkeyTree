use std::collections::{HashMap, VecDeque};

use crate::network::topology::Topology;
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{JobId, MLJob};
use crate::system_modules::cassini::PlacementCandidate;
use super::block::DEFAULT_BLOCK_SIZE;

/// A FIFO block scheduler that places workers on the first available blocks.
///
/// Unlike `BlockScheduler` which uses best-fit (locality-aware) placement,
/// this scheduler simply scans blocks in linear order and assigns the first
/// free blocks it finds, regardless of ToR affinity.
///
/// This serves as a baseline: FIFO ordering + first-fit placement with no
/// topology awareness beyond block alignment.
#[derive(Debug)]
pub struct FifoBlockScheduler {
    job_queue: VecDeque<JobId>,
    preloaded_placements: HashMap<JobId, Vec<usize>>,
    block_size: usize,
}

impl FifoBlockScheduler {
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        Self {
            job_queue: VecDeque::new(),
            preloaded_placements: HashMap::new(),
            block_size,
        }
    }

    pub fn with_default_block_size() -> Self {
        Self::new(DEFAULT_BLOCK_SIZE)
    }

    pub fn set_preloaded_placement(&mut self, job_id: JobId, hosts: Vec<usize>) {
        self.preloaded_placements.insert(job_id, hosts);
    }

    fn is_block_free(&self, block_idx: usize, host_busy: &[bool]) -> bool {
        let start = block_idx * self.block_size;
        let end = start + self.block_size;
        if end > host_busy.len() {
            return false;
        }
        (start..end).all(|h| !host_busy[h])
    }

    /// First-fit placement: scan blocks in order and grab the first available ones.
    fn compute_placement(&self, num_workers: usize, host_busy: &[bool]) -> Option<HashMap<usize, usize>> {
        if num_workers % self.block_size != 0 {
            panic!(
                "Job requested {} workers, but must be a multiple of block_size ({})",
                num_workers, self.block_size
            );
        }

        let num_blocks_needed = num_workers / self.block_size;
        let total_blocks = host_busy.len() / self.block_size;

        let mut worker_to_host = HashMap::new();
        let mut worker_id = 0;
        let mut blocks_found = 0;

        for block_idx in 0..total_blocks {
            if blocks_found >= num_blocks_needed {
                break;
            }
            if self.is_block_free(block_idx, host_busy) {
                let start = block_idx * self.block_size;
                for offset in 0..self.block_size {
                    worker_to_host.insert(worker_id, start + offset);
                    worker_id += 1;
                }
                blocks_found += 1;
            }
        }

        if blocks_found >= num_blocks_needed {
            Some(worker_to_host)
        } else {
            None
        }
    }
}

impl JobScheduler for FifoBlockScheduler {
    fn try_schedule_job<T: Topology>(
        &mut self,
        job: &mut MLJob,
        _topology: &T,
        available_hosts: &[bool],
    ) -> bool {
        // Handle preloaded placements
        if let Some(hosts) = self.preloaded_placements.get(&job.id) {
            if hosts.len() != job.num_workers {
                return false;
            }
            if hosts.iter().any(|&h| h >= available_hosts.len() || available_hosts[h]) {
                return false;
            }
            for (worker_id, &host_index) in hosts.iter().enumerate() {
                if let Some(worker) = job.workers.get_mut(&worker_id) {
                    worker.host_index = host_index;
                }
                job.worker_to_host.insert(worker_id, host_index);
            }
            return true;
        }

        let placement = match self.compute_placement(job.num_workers, available_hosts) {
            Some(p) => p,
            None => return false,
        };

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
