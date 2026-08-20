use std::collections::VecDeque;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::system_modules::cassini::PlacementCandidate;
use crate::network::topology::Topology;

/// Trait for job scheduling strategies
pub trait JobScheduler {
    /// Attempts to schedule a job on the available hosts
    /// Returns true if the job was successfully scheduled, false if it should remain queued
    fn try_schedule_job<T: Topology>(&mut self, job: &mut MLJob, topology: &T, available_hosts: &[bool]) -> bool;
    
    /// Returns the priority of a job for scheduling (higher values = higher priority)
    fn get_job_priority(&self, job: &MLJob) -> u64;
    
    /// Notifies the scheduler that a job has completed
    /// This allows the scheduler to update internal state and try scheduling queued jobs
    fn notify_job_completed(&mut self, job_id: JobId, completion_time_us: u64);
    
    /// Returns the next job ID that should be attempted for scheduling, if any
    fn get_next_job_to_schedule(&mut self) -> Option<JobId>;
    
    /// Adds a job to the scheduling queue
    fn enqueue_job(&mut self, job_id: JobId);
    
    /// Removes a job from the scheduling queue
    fn dequeue_job(&mut self) -> Option<JobId>;
    
    /// Returns true if there are jobs waiting in the queue
    fn has_queued_jobs(&self) -> bool;

    /// Generates up to `max_candidates` placement candidates for a job without mutating it.
    /// Implementations that do not search placements (e.g., Snapshot) should return an empty list.
    fn generate_placement_candidates<T: Topology>(
        &self,
        job: &MLJob,
        topology: &T,
        available_hosts: &[bool],
        max_candidates: usize,
    ) -> Vec<PlacementCandidate>;
}

/// A simple FIFO (First-In-First-Out) job scheduler
#[derive(Debug)]
pub struct FifoScheduler {
    /// Queue of jobs waiting to be scheduled
    job_queue: VecDeque<JobId>,
}

impl FifoScheduler {
    /// Creates a new FIFO scheduler
    pub fn new() -> Self {
        Self {
            job_queue: VecDeque::new(),
        }
    }
    
    /// Adds a job to the scheduling queue
    pub fn enqueue_job(&mut self, job_id: JobId) {
        self.job_queue.push_back(job_id);
    }
    
    /// Removes a job from the scheduling queue
    pub fn dequeue_job(&mut self) -> Option<JobId> {
        self.job_queue.pop_front()
    }
    
    /// Returns the number of jobs in the queue
    pub fn queue_length(&self) -> usize {
        self.job_queue.len()
    }
    
    /// Checks if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.job_queue.is_empty()
    }
    
    /// Peeks at the next job in the queue without removing it
    pub fn peek_next_job(&self) -> Option<JobId> {
        self.job_queue.front().copied()
    }
}

impl Default for FifoScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl JobScheduler for FifoScheduler {
    /// Tries to schedule a job using a simple first-fit strategy
    /// For FIFO, we simply try to assign the job to the first available hosts
    fn try_schedule_job<T: Topology>(&mut self, job: &mut MLJob, _topology: &T, available_hosts: &[bool]) -> bool {
        // Count available hosts (false = available, true = busy)
        let num_available = available_hosts.iter().filter(|&&busy| !busy).count();
        

        
        // Check if we have enough hosts for this job
        if num_available < job.num_workers {
            return false; // Not enough resources
        }
        
        // Find the first num_workers available hosts
        let mut assigned_hosts = Vec::new();
        for (host_index, &busy) in available_hosts.iter().enumerate() {
            if !busy { // host is available
                assigned_hosts.push(host_index);
                if assigned_hosts.len() == job.num_workers {
                    break;
                }
            }
        }
        
        // If we couldn't find enough hosts, something went wrong
        if assigned_hosts.len() != job.num_workers {
            return false;
        }
        
        // Update worker host assignments
        for (worker_id, &host_index) in assigned_hosts.iter().enumerate() {
            if let Some(worker) = job.workers.get_mut(&worker_id) {
                worker.host_index = host_index;
            }
            job.worker_to_host.insert(worker_id, host_index);
        }
        
        true
    }
    
    /// For FIFO, priority is simply the submit time (earlier jobs have higher priority)
    /// We negate the submit time so that earlier times result in higher priority values
    fn get_job_priority(&self, job: &MLJob) -> u64 {
        u64::MAX - job.submit_time_us
    }
    
    /// Notifies the scheduler that a job has completed
    /// For FIFO, we don't need to do anything special here since jobs are queued independently
    fn notify_job_completed(&mut self, _job_id: JobId, _completion_time_us: u64) {
        // FIFO scheduler doesn't need to track completed jobs
        // The main benefit is triggering attempts to schedule queued jobs
    }
    
    /// Returns the next job ID that should be attempted for scheduling, if any
    fn get_next_job_to_schedule(&mut self) -> Option<JobId> {
        self.peek_next_job()
    }
    
    /// Adds a job to the scheduling queue
    fn enqueue_job(&mut self, job_id: JobId) {
        self.job_queue.push_back(job_id);
    }
    
    /// Removes a job from the scheduling queue
    fn dequeue_job(&mut self) -> Option<JobId> {
        self.job_queue.pop_front()
    }
    
    /// Returns true if there are jobs waiting in the queue
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
        // Mirror FIFO's first-fit behavior as a single candidate, if possible
        let num_available = available_hosts.iter().filter(|&&busy| !busy).count();
        if num_available < job.num_workers { return Vec::new(); }

        let mut assigned_hosts = Vec::with_capacity(job.num_workers);
        for (host_index, &busy) in available_hosts.iter().enumerate() {
            if !busy {
                assigned_hosts.push(host_index);
                if assigned_hosts.len() == job.num_workers { break; }
            }
        }
        if assigned_hosts.len() != job.num_workers { return Vec::new(); }

        let mut worker_to_host = std::collections::HashMap::new();
        for (worker_id, &host_index) in assigned_hosts.iter().enumerate() {
            worker_to_host.insert(worker_id, host_index);
        }
        vec![PlacementCandidate { job_id: job.id, worker_to_host, compatibility_score: None }]
    }
}