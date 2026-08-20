use std::collections::VecDeque;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::job_scheduler::JobScheduler;
use crate::network::topology::Topology;
use crate::system_modules::cassini::PlacementCandidate;
use std::collections::HashMap;

/// Simple job scheduler that generates multiple placement candidates
/// Cassini will evaluate and select the best placement
#[derive(Debug)]
pub struct SimpleScheduler {
    /// Queue of jobs waiting to be scheduled
    job_queue: VecDeque<JobId>,
}

impl SimpleScheduler {
    pub fn new() -> Self {
        Self {
            job_queue: VecDeque::new(),
        }
    }
    
    /// Generates multiple placement candidates for a job
    /// Returns up to max_candidates different placement options
    /// TODO: make this the actual implementation
    pub fn generate_placement_candidates<T: Topology>(
        &self, 
        job: &MLJob, 
        _topology: &T, 
        available_hosts: &[bool],
        max_candidates: usize
    ) -> Vec<PlacementCandidate> {
        let mut candidates = Vec::new();
        
        // Count available hosts
        let num_available = available_hosts.iter().filter(|&&busy| !busy).count();
        if num_available < job.num_workers {
            return candidates; // Not enough resources
        }
        
        // Collect available host indices
        let available_host_indices: Vec<usize> = available_hosts
            .iter()
            .enumerate()
            .filter_map(|(idx, &busy)| if !busy { Some(idx) } else { None })
            .collect();
        
        // Strategy 1: First-fit placement
        if let Some(candidate) = self.generate_first_fit_placement(job, &available_host_indices) {
            candidates.push(candidate);
        }
        
        // Strategy 2: Random placements (multiple attempts)
        for _ in 0..std::cmp::min(max_candidates - 1, 5) {
            if let Some(candidate) = self.generate_random_placement(job, &available_host_indices) {
                // Check if this placement is different from existing ones
                if !candidates.iter().any(|c| self.placements_equal(&c.worker_to_host, &candidate.worker_to_host)) {
                    candidates.push(candidate);
                }
            }
        }
        
        // Strategy 3: Packed placement (fill hosts sequentially)
        if candidates.len() < max_candidates {
            if let Some(candidate) = self.generate_packed_placement(job, &available_host_indices) {
                if !candidates.iter().any(|c| self.placements_equal(&c.worker_to_host, &candidate.worker_to_host)) {
                    candidates.push(candidate);
                }
            }
        }
        
        // Strategy 4: Spread placement (distribute evenly)
        if candidates.len() < max_candidates {
            if let Some(candidate) = self.generate_spread_placement(job, &available_host_indices) {
                if !candidates.iter().any(|c| self.placements_equal(&c.worker_to_host, &candidate.worker_to_host)) {
                    candidates.push(candidate);
                }
            }
        }
        
        candidates
    }
    
    fn generate_first_fit_placement(&self, job: &MLJob, available_hosts: &[usize]) -> Option<PlacementCandidate> {
        if available_hosts.len() < job.num_workers {
            return None;
        }
        
        let mut worker_to_host = HashMap::new();
        for worker_id in 0..job.num_workers {
            worker_to_host.insert(worker_id, available_hosts[worker_id]);
        }
        
        Some(PlacementCandidate {
            job_id: job.id,
            worker_to_host,
            compatibility_score: None, // Cassini will compute this
        })
    }
    
    fn generate_random_placement(&self, job: &MLJob, available_hosts: &[usize]) -> Option<PlacementCandidate> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        
        if available_hosts.len() < job.num_workers {
            return None;
        }
        
        let mut rng = thread_rng();
        let mut shuffled_hosts = available_hosts.to_vec();
        shuffled_hosts.shuffle(&mut rng);
        
        let mut worker_to_host = HashMap::new();
        for worker_id in 0..job.num_workers {
            worker_to_host.insert(worker_id, shuffled_hosts[worker_id]);
        }
        
        Some(PlacementCandidate {
            job_id: job.id,
            worker_to_host,
            compatibility_score: None,
        })
    }
    
    fn generate_packed_placement(&self, job: &MLJob, available_hosts: &[usize]) -> Option<PlacementCandidate> {
        if available_hosts.len() < job.num_workers {
            return None;
        }
        
        // Try to pack workers on the lowest numbered hosts
        let mut worker_to_host = HashMap::new();
        let mut sorted_hosts = available_hosts.to_vec();
        sorted_hosts.sort_unstable();
        
        for worker_id in 0..job.num_workers {
            worker_to_host.insert(worker_id, sorted_hosts[worker_id]);
        }
        
        Some(PlacementCandidate {
            job_id: job.id,
            worker_to_host,
            compatibility_score: None,
        })
    }
    
    fn generate_spread_placement(&self, job: &MLJob, available_hosts: &[usize]) -> Option<PlacementCandidate> {
        if available_hosts.len() < job.num_workers {
            return None;
        }
        
        // Try to spread workers evenly across available hosts
        let mut worker_to_host = HashMap::new();
        let step = available_hosts.len() / job.num_workers;
        let step = step.max(1);
        
        for worker_id in 0..job.num_workers {
            let host_idx = (worker_id * step) % available_hosts.len();
            worker_to_host.insert(worker_id, available_hosts[host_idx]);
        }
        
        Some(PlacementCandidate {
            job_id: job.id,
            worker_to_host,
            compatibility_score: None,
        })
    }
    
    fn placements_equal(&self, placement1: &HashMap<usize, usize>, placement2: &HashMap<usize, usize>) -> bool {
        if placement1.len() != placement2.len() {
            return false;
        }
        
        for (worker, host1) in placement1 {
            if let Some(host2) = placement2.get(worker) {
                if host1 != host2 {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        true
    }
}

impl Default for SimpleScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl JobScheduler for SimpleScheduler {
    /// For SimpleScheduler, this just selects the first placement candidate
    /// In practice, Cassini should be called to evaluate candidates before this
    fn try_schedule_job<T: Topology>(&mut self, job: &mut MLJob, topology: &T, available_hosts: &[bool]) -> bool {
        let candidates = self.generate_placement_candidates(job, topology, available_hosts, 1);
        
        if let Some(candidate) = candidates.first() {
            // Apply the placement to the job
            for (worker_id, &host_index) in &candidate.worker_to_host {
                if let Some(worker) = job.workers.get_mut(worker_id) {
                    worker.host_index = host_index;
                }
                job.worker_to_host.insert(*worker_id, host_index);
            }
            true
        } else {
            false
        }
    }
    
    fn get_job_priority(&self, job: &MLJob) -> u64 {
        u64::MAX - job.submit_time_us
    }
    
    fn notify_job_completed(&mut self, _job_id: JobId, _completion_time_us: u64) {
        // No special action needed
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
        topology: &T,
        available_hosts: &[bool],
        max_candidates: usize,
    ) -> Vec<PlacementCandidate> {
        // Delegate to the inherent helper
        SimpleScheduler::generate_placement_candidates(self, job, topology, available_hosts, max_candidates)
    }
}