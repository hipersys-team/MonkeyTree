use std::collections::{HashMap, VecDeque};

use crate::network::topology::Topology;
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{JobId, MLJob};
use crate::system_modules::cassini::PlacementCandidate;

use super::rail_block::RailBlockScheduler;

/// Combines fixed placements for preloaded jobs with pod-aware
/// block scheduling (via `RailBlockScheduler`) for dynamically arriving jobs.
#[derive(Debug)]
pub struct PreloadedRailBlockScheduler {
    job_queue: VecDeque<JobId>,
    preloaded_placements: HashMap<JobId, Vec<usize>>,
    inner: RailBlockScheduler,
}

impl PreloadedRailBlockScheduler {
    pub fn new(blocks_per_pod: usize, block_size: usize) -> Self {
        Self {
            job_queue: VecDeque::new(),
            preloaded_placements: HashMap::new(),
            inner: RailBlockScheduler::new(blocks_per_pod, block_size),
        }
    }

    pub fn set_preloaded_placement(&mut self, job_id: JobId, hosts: Vec<usize>) {
        self.preloaded_placements.insert(job_id, hosts);
    }
}

impl JobScheduler for PreloadedRailBlockScheduler {
    fn try_schedule_job<T: Topology>(
        &mut self,
        job: &mut MLJob,
        topology: &T,
        available_hosts: &[bool],
    ) -> bool {
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
            true
        } else {
            self.inner.try_schedule_job(job, topology, available_hosts)
        }
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
        topology: &T,
        available_hosts: &[bool],
        max_candidates: usize,
    ) -> Vec<PlacementCandidate> {
        if let Some(hosts) = self.preloaded_placements.get(&job.id) {
            if hosts.len() != job.num_workers {
                return Vec::new();
            }
            if hosts.iter().any(|&h| h >= available_hosts.len() || available_hosts[h]) {
                return Vec::new();
            }
            let mut worker_to_host = HashMap::new();
            for (worker_id, &host_index) in hosts.iter().enumerate() {
                worker_to_host.insert(worker_id, host_index);
            }
            vec![PlacementCandidate {
                job_id: job.id,
                worker_to_host,
                compatibility_score: None,
            }]
        } else {
            self.inner.generate_placement_candidates(job, topology, available_hosts, max_candidates)
        }
    }
}
