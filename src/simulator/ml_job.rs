use crate::utils::DHashMap;
use crate::simulator::ml_worker::{MLWorker, WorkerId, WorkerEvent, WorkerEventKind};

/// Unique identifier for ML jobs
pub type JobId = usize;

/// Pipeline stage information for jobs with pipeline parallelism.
/// This allows monkeytree to treat each stage as an independent unit for
/// fragmentation analysis and migration optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineStageInfo {
    /// Number of pipeline stages
    pub num_stages: usize,
    /// Workers per stage (= tp_size * dp_replicas)
    pub workers_per_stage: usize,
    /// Number of independent DP rings per stage
    pub rings_per_stage: usize,
}

impl PipelineStageInfo {
    /// Get the stage index for a given worker ID.
    /// Workers are laid out as: stage * workers_per_stage + local_worker_id
    pub fn worker_stage(&self, worker_id: WorkerId) -> usize {
        worker_id / self.workers_per_stage
    }
    
    /// Get the worker IDs for a given stage.
    pub fn stage_workers(&self, stage: usize) -> std::ops::Range<WorkerId> {
        let start = stage * self.workers_per_stage;
        let end = start + self.workers_per_stage;
        start..end
    }
}

/// State of an ML job during execution
#[derive(Debug, Clone, PartialEq)]
pub enum JobState {
    /// Job is waiting to be scheduled
    Queued,
    /// Job has been scheduled and workers are assigned
    Scheduled,
    /// Job is currently running
    Running,
    /// Job has completed successfully
    Completed,
    /// Job has failed
    Failed,
}

/// Definition of an ML training job
#[derive(Debug)]
pub struct MLJob {
    /// Unique identifier for this job
    pub id: JobId,
    /// Current state of the job
    pub state: JobState,
    /// Time when the job was submitted (in milliseconds)
    pub submit_time_us: u64,
    /// Time when the job was scheduled (in milliseconds)
    pub schedule_time_us: Option<u64>,
    /// Time when the job started running (in milliseconds)
    pub start_time_us: Option<u64>,
    /// Time when the job completed (in milliseconds)
    pub completion_time_us: Option<u64>,
    /// Number of workers required for this job
    pub num_workers: usize,
    /// Total number of iterations for this job
    pub total_iterations: usize,
    /// Workers assigned to this job (worker_id -> worker)
    pub workers: DHashMap<WorkerId, MLWorker>,
    /// Mapping from worker_id to host_index in the topology
    pub worker_to_host: DHashMap<WorkerId, usize>,
    /// Optional name/description for the job
    pub name: Option<String>,
    /// Mapping from (src_worker, stable send template id) -> job-local flow index
    pub send_template_to_flow_idx: DHashMap<(WorkerId, usize), usize>,
    /// Next job-local flow index to assign
    pub next_flow_idx: usize,
    /// Number of independent communication rings this job creates.
    /// For strided ring with stride S, this equals S.
    /// For regular AllReduce or AllToAll, this is 1.
    /// For pipeline jobs, this is the total across all stages (num_stages * rings_per_stage).
    pub ring_count: usize,
    /// Pipeline stage information (if this is a pipeline parallel job).
    /// When set, monkeytree treats each stage as an independent unit for optimization.
    pub pipeline_stages: Option<PipelineStageInfo>,
}

impl MLJob {
    /// Creates a new ML job
    pub fn new(id: JobId, submit_time_us: u64, num_workers: usize, total_iterations: usize) -> Self {
        Self {
            id,
            state: JobState::Queued,
            submit_time_us,
            schedule_time_us: None,
            start_time_us: None,
            completion_time_us: None,
            num_workers,
            total_iterations,
            workers: DHashMap::default(),
            worker_to_host: DHashMap::default(),
            name: None,
            send_template_to_flow_idx: DHashMap::default(),
            next_flow_idx: 0,
            ring_count: 1, // Default: single ring
            pipeline_stages: None,
        }
    }
    
    /// Sets the name/description for this job
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
    
    /// Sets the number of independent rings this job creates
    pub fn with_ring_count(mut self, ring_count: usize) -> Self {
        self.ring_count = ring_count;
        self
    }
    
    /// Sets the pipeline stage information for pipeline parallel jobs.
    /// This also updates ring_count to be num_stages * rings_per_stage.
    pub fn with_pipeline_stages(mut self, info: PipelineStageInfo) -> Self {
        self.ring_count = info.num_stages * info.rings_per_stage;
        self.pipeline_stages = Some(info);
        self
    }
    
    /// Assigns a worker to a specific host in the topology
    pub fn assign_worker(&mut self, worker_id: WorkerId, host_index: usize) {
        let worker = MLWorker::new(self.id, worker_id, host_index, self.total_iterations);
        self.workers.insert(worker_id, worker);
        self.worker_to_host.insert(worker_id, host_index);
    }
    
    /// Adds an event template to a specific worker
    pub fn add_worker_event_template(&mut self, worker_id: WorkerId, event: WorkerEvent) -> Result<(), String> {
        if let Some(worker) = self.workers.get_mut(&worker_id) {
            worker.add_event_template(event);
            Ok(())
        } else {
            Err(format!("Worker {} not found in job {}", worker_id, self.id))
        }
    }
    
    /// Checks if the job is ready to start (all workers assigned)
    pub fn is_ready_to_start(&self) -> bool {
        self.workers.len() == self.num_workers && self.state == JobState::Scheduled
    }
    
    /// Checks if all workers have completed all their iterations
    pub fn is_completed(&self) -> bool {
        self.workers.values().all(|worker| worker.is_completely_finished())
    }
    
    /// Gets the list of assigned host indices
    pub fn get_assigned_hosts(&self) -> Vec<usize> {
        self.worker_to_host.values().cloned().collect()
    }
    
    /// Gets a worker by ID
    pub fn get_worker(&self, worker_id: WorkerId) -> Option<&MLWorker> {
        self.workers.get(&worker_id)
    }
    
    /// Gets a mutable reference to a worker by ID
    pub fn get_worker_mut(&mut self, worker_id: WorkerId) -> Option<&mut MLWorker> {
        self.workers.get_mut(&worker_id)
    }
    
    /// Gets the host index for a worker
    pub fn get_worker_host(&self, worker_id: WorkerId) -> Option<usize> {
        self.worker_to_host.get(&worker_id).copied()
    }
    
    /// Marks the job as scheduled
    pub fn mark_scheduled(&mut self, schedule_time_us: u64) {
        self.state = JobState::Scheduled;
        self.schedule_time_us = Some(schedule_time_us);
    }
    
    /// Marks the job as running
    pub fn mark_running(&mut self, start_time_us: u64) {
        self.state = JobState::Running;
        self.start_time_us = Some(start_time_us);
    }
    
    /// Marks the job as completed
    pub fn mark_completed(&mut self, completion_time_us: u64) {
        self.state = JobState::Completed;
        self.completion_time_us = Some(completion_time_us);
    }
    
    /// Marks the job as failed
    pub fn mark_failed(&mut self, failure_time_us: u64) {
        self.state = JobState::Failed;
        self.completion_time_us = Some(failure_time_us);
    }
    
    /// Rebuilds the worker_to_host mapping from the workers' host_index fields.
    /// Called after rank reassignment to ensure consistency.
    pub fn rebuild_worker_to_host(&mut self) {
        self.worker_to_host.clear();
        for (&wid, worker) in self.workers.iter() {
            self.worker_to_host.insert(wid, worker.host_index);
        }
    }
    
    /// Rebuilds the send_template_to_flow_idx mapping.
    /// Called after rank reassignment since WorkerIds have changed.
    pub fn rebuild_flow_indices(&mut self) {
        self.send_template_to_flow_idx.clear();
        let mut worker_ids: Vec<WorkerId> = self.workers.keys().copied().collect();
        worker_ids.sort_unstable();
        let mut idx = 0usize;
        for wid in worker_ids {
            let worker = self.workers.get(&wid).unwrap();
            for ev in &worker.template_events {
                if ev.kind == WorkerEventKind::FlowSend {
                    self.send_template_to_flow_idx.insert((wid, ev.template_id), idx);
                    idx += 1;
                }
            }
        }
        self.next_flow_idx = idx;
    }
}

/// Builder for creating ML jobs with specific worker configurations
pub struct MLJobBuilder {
    job: MLJob,
    next_worker_id: WorkerId,
    total_iterations: usize,
}

impl MLJobBuilder {
    /// Creates a new job builder
    pub fn new(job_id: JobId, submit_time_us: u64, num_workers: usize, total_iterations: usize) -> Self {
        Self {
            job: MLJob::new(job_id, submit_time_us, num_workers, total_iterations),
            next_worker_id: 0,
            total_iterations,
        }
    }
    
    /// Sets the job name
    pub fn with_name(mut self, name: String) -> Self {
        self.job.name = Some(name);
        self
    }
    
    /// Adds a worker with a specific set of event templates
    /// Note: Host assignment will be done by the scheduler when the job is scheduled
    pub fn add_worker_with_events(mut self, _host_index: usize, events: Vec<WorkerEvent>) -> Self {
        let worker_id = self.next_worker_id;
        self.next_worker_id += 1;
        
        // Create a temporary worker just to store the event templates
        // The scheduler will properly assign it to a host later
        let mut temp_worker = MLWorker::new(self.job.id, worker_id, 0, self.total_iterations); // temporary host index
        for event in events {
            temp_worker.add_event_template(event);
        }
        
        // Store the worker with event templates but no final host assignment yet
        self.job.workers.insert(worker_id, temp_worker);
        
        self
    }
    
    /// Builds the final job
    pub fn build(self) -> MLJob {
        let mut job = self.job;
        // Pre-assign global per-job flow indices in a deterministic order:
        // ascending worker_id, then in each worker by template order.
        let mut worker_ids: Vec<WorkerId> = job.workers.keys().copied().collect();
        worker_ids.sort_unstable();
        let mut idx = 0usize;
        for wid in worker_ids {
            let worker = job.workers.get(&wid).unwrap();
            for ev in &worker.template_events {
                if ev.kind == WorkerEventKind::FlowSend {
                    job.send_template_to_flow_idx.insert((wid, ev.template_id), idx);
                    idx += 1;
                }
            }
        }
        job.next_flow_idx = idx;
        job
    }
} 