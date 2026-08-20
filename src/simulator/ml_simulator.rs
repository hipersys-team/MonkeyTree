use std::collections::{BinaryHeap, HashMap};
use crate::utils::{DHashMap, DHashSet};
use crate::utils::compatibility::WorkerDescription;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::cmp::Ordering;
use crate::network::Simulator;
use crate::network::alloc::BandwidthAllocator;
use crate::network::EventKind;
use crate::network::topology::Topology;
use crate::network::flow::FlowId;
use crate::simulator::ml_job::{MLJob, JobId, JobState};
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::flow_scheduler::{FlowScheduler, QueuedFlow};
use crate::simulator::ml_worker::{WorkerId, WorkerEventKind, WorkerNotifyResult};
use crate::simulator::system::{SystemModule, MigrationPlan, WorkerMigrationInfo, PendingJobMigration, migration_flow_idx, TimerId};

// Debug flag for migration tracking. Enable to trace host assignments during migrations.
const DEBUG_MIGRATION: bool = false;

// Flow scheduling is now driven by polling the flow scheduler in the main loop.

/// Types of ML simulation events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MLEventKind {
    /// A new ML job arrives
    JobArrival,
    /// A computation event completes
    ComputeCompletion,
    /// A flow sending event needs to be processed
    FlowSendReady,
    /// A flow receiving event starts (worker told to start receiving)
    FlowReceiveReady,
    /// A flow send has completed (network simulator notification)
    FlowSendComplete,
    /// A job has completed and needs cleanup
    JobComplete,
    /// A network event has occurred
    NetworkEvent,
    /// System reconfiguration (routing, scheduling changes for future flows)
    Reconfigure,
    /// Migration phase begins
    MigrationBegin,
    /// Migration phase ends
    MigrationEnd,
    /// Wake the flow scheduler to poll for ready flows
    FlowSchedulerPoll,
    /// Restart a job after post-migration delay
    PostMigrationRestart,
    /// System module timer fired
    SystemTimer,
}

/// A pending timer request from a system module.
#[derive(Debug, Clone)]
pub struct TimerRequest {
    /// When the timer should fire (absolute time in microseconds)
    pub fire_at_us: u64,
    /// Identifier for the timer (passed back to on_timer)
    pub timer_id: TimerId,
}

/// Internal event structure for ML discrete event simulation
#[derive(Debug, Clone)]
struct MLEvent {
    /// Time when this event should be processed (in milliseconds)
    time_us: u64,
    /// Type of ML event
    kind: MLEventKind,
    /// Associated job ID (if applicable)
    job_id: Option<JobId>,
    /// Associated worker ID (if applicable) 
    worker_id: Option<WorkerId>,
    /// Associated worker event ID (if applicable)
    event_id: Option<usize>,
    /// Associated flow ID for network events
    flow_id: Option<FlowId>,
    /// Associated timer ID for SystemTimer events
    timer_id: Option<TimerId>,
}

impl MLEvent {
    fn key(&self) -> (u64, u8) { 
        (self.time_us, match self.kind {
            MLEventKind::JobArrival => 0,
            MLEventKind::ComputeCompletion => 1,
            MLEventKind::FlowSendReady => 2,
            MLEventKind::FlowReceiveReady => 3,
            MLEventKind::FlowSendComplete => 4,
            MLEventKind::JobComplete => 5,
            MLEventKind::NetworkEvent => 6,
            MLEventKind::Reconfigure => 7,
            MLEventKind::MigrationBegin => 8,
            MLEventKind::MigrationEnd => 9,
            MLEventKind::FlowSchedulerPoll => 10,
            MLEventKind::PostMigrationRestart => 11,
            MLEventKind::SystemTimer => 12,
        })
    }
}

// Event ordering for priority queue (min-heap)
impl Ord for MLEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        other.key().cmp(&self.key())
    }
}

impl PartialOrd for MLEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for MLEvent {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for MLEvent {}

/// Shared simulation context that can be passed to simulator components
/// to access high-level information such as the current simulation time.
///
/// For now it only tracks time, but additional fields can be added as
/// necessary (e.g., references to global statistics, configuration, etc.).
#[derive(Clone, Debug)]
pub struct MLContext {
    /// Handle to the current simulation time in milliseconds. The inner
    /// `Cell` allows mutation while the `Rc` enables cheap sharing across
    /// components without complex lifetime management.
    pub time_us: Rc<Cell<u64>>, // public so simulator can update directly
    pub waiting_flows: Rc<RefCell<DHashMap<FlowId, (JobId, usize, usize, WorkerId, WorkerId, usize, usize)>>>,
    /// Job iteration progress: job_id -> (total_iterations, completed_iterations)
    pub job_iterations: Rc<RefCell<DHashMap<JobId, (usize, usize)>>>,
    /// Active jobs info mirrored from simulator for read-only access in modules
    pub active_jobs: Rc<RefCell<DHashMap<JobId, ActiveJobInfo>>>,
    /// Current placements: job_id -> (worker_id -> host_index)
    pub placements: Rc<RefCell<DHashMap<JobId, DHashMap<WorkerId, usize>>>>,
    /// Inverse placement map for fast lookup: host_index -> (job_id, worker_id)
    pub host_to_worker: Rc<RefCell<Vec<Option<(JobId, WorkerId)>>>>,
    /// Per-worker distinct destinations this worker sends to: (job_id, worker_id) -> [dst_worker_id]
    pub worker_send_neighbors: Rc<RefCell<DHashMap<(JobId, WorkerId), Vec<WorkerId>>>>,
    /// Per-worker distinct sources this worker receives from: (job_id, worker_id) -> [src_worker_id]
    pub worker_recv_neighbors: Rc<RefCell<DHashMap<(JobId, WorkerId), Vec<WorkerId>>>>,
    /// Per-worker network send progress: (sent_total_bytes_across_all_iters, per_iter_total_bytes)
    /// per_iter_total_bytes is the sum of FlowSend sizes in one iteration for that worker
    pub worker_send_progress: Rc<RefCell<DHashMap<(JobId, WorkerId), (u64, u64)>>>,
    /// Per-job count of completed flows since last active-set change
    pub flow_completions_per_job: Rc<RefCell<DHashMap<JobId, u64>>>,
    /// Pending timer requests from system modules.
    /// System modules push timer requests here; the simulator drains and schedules them.
    pub pending_timers: Rc<RefCell<Vec<TimerRequest>>>,
}

impl MLContext {
    /// Schedule a timer to fire after `delay_us` microseconds.
    /// The `timer_id` will be passed to the system module's `on_timer` callback.
    pub fn schedule_timer(&self, delay_us: u64, timer_id: TimerId) {
        let fire_at = self.time_us.get() + delay_us;
        self.pending_timers.borrow_mut().push(TimerRequest {
            fire_at_us: fire_at,
            timer_id,
        });
    }

    /// Schedule a timer to fire at an absolute time.
    pub fn schedule_timer_at(&self, fire_at_us: u64, timer_id: TimerId) {
        self.pending_timers.borrow_mut().push(TimerRequest {
            fire_at_us,
            timer_id,
        });
    }
}

/// Read-only snapshot of an active job for system modules
#[derive(Clone, Debug)]
pub struct ActiveJobInfo {
    pub job_id: JobId,
    pub name: Option<String>,
    pub submit_time_us: u64,
    pub num_workers: usize,
    pub total_iterations: usize,
    /// Number of independent rings this job creates (stride for strided ring, 1 otherwise)
    pub ring_count: usize,
    /// Per-worker summarized templates for compatibility and planning
    pub worker_descriptions: Vec<WorkerDescription>,
    /// Pipeline stage information (if this is a pipeline parallel job)
    pub pipeline_stages: Option<crate::simulator::ml_job::PipelineStageInfo>,
}

/// Main ML training simulator that orchestrates jobs, workers, and network simulation
pub struct MLSimulator<T: Topology, S: JobScheduler, FS: FlowScheduler, M: SystemModule<T, S, FS>, A: BandwidthAllocator> {
    /// Current simulation time in milliseconds
    pub now_us: u64,
    /// Shared simulation context (exposed to workers and other components)
    pub context: MLContext,
    /// Underlying network simulator
    network_sim: Simulator<T, A>,
    /// Job scheduler
    scheduler: S,
    /// Flow scheduler
    flow_scheduler: FS,
    /// Counter for generating globally unique event IDs
    next_event_id: usize,
    /// All jobs (active and completed)
    jobs: DHashMap<JobId, MLJob>,
    /// Queue of pending ML events
    event_queue: BinaryHeap<MLEvent>,
    /// Track which hosts are currently busy (true = busy, false = available)
    host_busy: Vec<bool>,
    // Track active jobs
    active_jobs: DHashSet<JobId>,
    // Track which hosts are currently occupied by which job
    host_assignment: Vec<Option<JobId>>,
    /// Mapping from worker to running compute events and their completion times
    running_compute: DHashMap<(JobId, WorkerId), (usize, u64)>, // (event_id, completion_time)
    // Mapping from flow_id to (job_id, src_worker, dst_worker, event_id)
    waiting_flows: Rc<RefCell<DHashMap<FlowId, (JobId, usize, usize, WorkerId, WorkerId, usize, usize)>>>,
    /// Send events waiting for their corresponding receive events to start: (job_id, src_worker, dst_worker) -> (event_id)
    pending_sends: DHashMap<(JobId, WorkerId, WorkerId), usize>,
    /// Receive events waiting for their corresponding sends to start: (job_id, src_worker, dst_worker) -> (event_id)
    pending_receives: DHashMap<(JobId, WorkerId, WorkerId), usize>,
    /// Mapping from flow ID to job ID
    flow_to_job: DHashMap<FlowId, JobId>,
    /// Pluggable system module
    system_module: M,
    /// A reconfigure has been requested but not yet processed
    reconfigure_requested: bool,
    /// A reconfigure was deferred because migrations were in progress
    reconfigure_deferred_for_migration: bool,
    /// Per-job iteration barrier: (current_iter_idx, set of workers that reported completion for this iter)
    worker_iter_barrier: DHashMap<JobId, (usize, DHashSet<WorkerId>)>,
    // // --- instrumentation (disabled to save memory) ---
    // /// Total FlowSchedulerPoll events processed
    // instr_polls_processed: u64,
    // /// Map from timestamp to count of FlowSchedulerPoll processed at that time (to detect duplicates)
    // instr_polls_at_time: DHashMap<u64, u64>,
    /// Deduplicate pending FlowSchedulerPoll: only one at the earliest time is queued
    next_poll_scheduled: Option<u64>,
    /// Migration state
    migrating: bool,
    pending_migration: Option<MigrationPlan>,
    migration_complete_enqueued: bool,
    /// Active migration flows: FlowId -> WorkerMigrationInfo
    active_migration_flows: DHashMap<FlowId, WorkerMigrationInfo>,
    /// Jobs currently paused due to ongoing migration (per-job tracking)
    /// A job is removed from this set when all its migration flows complete
    jobs_migrating: DHashSet<JobId>,
    /// Pending migration flows per job: JobId -> set of FlowIds for that job's migration
    migration_flows_by_job: DHashMap<JobId, DHashSet<FlowId>>,
    /// Per-job pending migration state for barrier-synchronized migrations
    pending_job_migrations: DHashMap<JobId, PendingJobMigration>,
    /// Reverse dependency index: job_id -> set of jobs waiting for this job to reach its barrier
    /// (so they can migrate to hosts this job currently occupies)
    migration_dependents: DHashMap<JobId, DHashSet<JobId>>,
    /// Delay (in microseconds) before restarting a job after migration completes.
    /// Default is 0 (no delay). Set using `set_post_migration_delay_us`.
    post_migration_delay_us: u64,
    /// Scratch buffer for running job IDs (reused to avoid allocation)
    scratch_job_ids: Vec<JobId>,
    /// Scratch buffer for worker IDs (reused to avoid allocation)
    scratch_worker_ids: Vec<WorkerId>,
}

impl<T: Topology, S: JobScheduler, FS: FlowScheduler, M: SystemModule<T, S, FS>, A: BandwidthAllocator> MLSimulator<T, S, FS, M, A> {
    /// Build a read-only snapshot of a job for the shared context
    fn make_active_job_snapshot(job: &MLJob) -> ActiveJobInfo {
        // Build worker descriptions in stable worker_id order
        let mut worker_ids: Vec<crate::simulator::ml_worker::WorkerId> = job.workers.keys().copied().collect();
        worker_ids.sort_unstable();
        let worker_descriptions = worker_ids
            .iter()
            .map(|wid| WorkerDescription::from_worker(&job.workers[wid]))
            .collect();
        ActiveJobInfo {
            job_id: job.id,
            name: job.name.clone(),
            submit_time_us: job.submit_time_us,
            num_workers: job.num_workers,
            total_iterations: job.total_iterations,
            ring_count: job.ring_count,
            worker_descriptions,
            pipeline_stages: job.pipeline_stages,
        }
    }

    /// Applies a validated migration plan atomically to simulator and context state.
    fn apply_migration_plan(&mut self, plan: &crate::simulator::system::MigrationPlan) {
        // Validate: collect all host changes and ensure no conflicts
        let mut new_host_assignment = self.host_assignment.clone();
        let mut new_host_busy = self.host_busy.clone();
        // Build merged per-job mappings: start from current mapping and overlay requested changes
        let mut merged_maps: DHashMap<JobId, DHashMap<WorkerId, usize>> = DHashMap::default();
        // First, free all hosts for affected jobs in the scratch copies
        for jm in &plan.jobs {
            // Free current hosts
            if let Some(job) = self.jobs.get(&jm.job_id) {
                for &h in job.worker_to_host.values() {
                    if h < new_host_busy.len() { new_host_busy[h] = false; new_host_assignment[h] = None; }
                }
                // Merge: begin with current mapping
                let mut merged = job.worker_to_host.clone();
                // Overlay the entries provided in the plan (only workers that move)
                for (&wid, &host) in jm.worker_to_host.iter() {
                    merged.insert(wid, host);
                }
                merged_maps.insert(jm.job_id, merged);
            }
        }
        // Then, allocate new hosts and check bounds/conflicts
        for jm in &plan.jobs {
            // Use merged mapping (unchanged workers remain in place)
            let merged = merged_maps.get(&jm.job_id).expect("merged map must exist");
            // Sanity-check coverage against job size
            let expected_workers = self.jobs.get(&jm.job_id).map(|j| j.num_workers).unwrap_or(0);
            if merged.len() != expected_workers {
                panic!("Merged migration mapping for job {} has {} workers, expected {}", jm.job_id, merged.len(), expected_workers);
            }
            for (&_wid, &host) in merged.iter() {
                if host >= new_host_busy.len() { panic!("Migration assigns out-of-range host {}", host); }
                if new_host_busy[host] { panic!("Migration host {} already occupied", host); }
                new_host_busy[host] = true;
                new_host_assignment[host] = Some(jm.job_id);
            }
        }
        // Commit: update per-job, workers, and context
        for jm in &plan.jobs {
            if let Some(job) = self.jobs.get_mut(&jm.job_id) {
                // Update job mapping
                // Clear previous inverse mapping entries for this job
                {
                    let mut h2w = self.context.host_to_worker.borrow_mut();
                    for &old_host in job.worker_to_host.values() {
                        debug_assert!(old_host < h2w.len(), "Old host index {} out of range {}", old_host, h2w.len());
                        h2w[old_host] = None;
                    }
                }
                // Commit merged mapping
                let merged = merged_maps.get(&jm.job_id).expect("merged map must exist");
                job.worker_to_host = merged.clone();
                // Update workers' host_index
                for (wid, host) in job.worker_to_host.iter() {
                    if let Some(w) = job.workers.get_mut(wid) { w.host_index = *host; }
                }
                // Mirror to context
                self.context.placements.borrow_mut().insert(jm.job_id, job.worker_to_host.clone());
                // Update inverse placement for new mapping
                {
                    let mut h2w = self.context.host_to_worker.borrow_mut();
                    for (&wid, &host) in job.worker_to_host.iter() {
                        debug_assert!(host < h2w.len(), "New host index {} out of range {}", host, h2w.len());
                        h2w[host] = Some((jm.job_id, wid));
                    }
                }
            }
        }
        // Commit host occupancy
        self.host_busy = new_host_busy;
        self.host_assignment = new_host_assignment;
        // Active jobs set and iteration/barriers remain unchanged.
    }
    /// Creates a new ML simulator with the given network topology and job scheduler
    pub fn new(topology: T, scheduler: S, flow_scheduler: FS, system_module: M, mut allocator: A) -> Self {
        let num_hosts = topology.total_hosts();
        let time_cell = Rc::new(Cell::new(0));
        let waiting_flows_cell = Rc::new(RefCell::new(DHashMap::default()));
        let job_iters_cell = Rc::new(RefCell::new(DHashMap::default()));
        let active_jobs_cell = Rc::new(RefCell::new(DHashMap::default()));
        let placements_cell = Rc::new(RefCell::new(DHashMap::default()));
        let host_to_worker_cell = Rc::new(RefCell::new(vec![None; num_hosts]));
        let worker_send_neighbors_cell = Rc::new(RefCell::new(DHashMap::default()));
        let worker_recv_neighbors_cell = Rc::new(RefCell::new(DHashMap::default()));
        let worker_send_progress_cell = Rc::new(RefCell::new(DHashMap::default()));
        let flow_completions_cell = Rc::new(RefCell::new(DHashMap::default()));
        let pending_timers_cell = Rc::new(RefCell::new(Vec::new()));
        let context = MLContext { 
            time_us: time_cell.clone(),
            waiting_flows: waiting_flows_cell.clone(),
            job_iterations: job_iters_cell.clone(),
            active_jobs: active_jobs_cell.clone(),
            placements: placements_cell.clone(),
            host_to_worker: host_to_worker_cell.clone(),
            worker_send_neighbors: worker_send_neighbors_cell.clone(),
            worker_recv_neighbors: worker_recv_neighbors_cell.clone(),
            worker_send_progress: worker_send_progress_cell.clone(),
            flow_completions_per_job: flow_completions_cell.clone(),
            pending_timers: pending_timers_cell,
        };
        allocator.set_context(&context);
        topology.set_context(&context);
        let mut this = Self {
            now_us: 0,
            context,
            network_sim: Simulator::new(topology, allocator),
            scheduler,
            flow_scheduler,
            next_event_id: 0,
            jobs: DHashMap::default(),
            event_queue: BinaryHeap::new(),
            host_busy: vec![false; num_hosts],
            active_jobs: DHashSet::default(),
            host_assignment: vec![None; num_hosts],
            running_compute: DHashMap::default(),
            waiting_flows: waiting_flows_cell,
            // note: job_iterations stored inside context
            pending_sends: DHashMap::default(),
            pending_receives: DHashMap::default(),
            flow_to_job: DHashMap::default(),
            system_module,
            reconfigure_requested: false,
            reconfigure_deferred_for_migration: false,
            worker_iter_barrier: DHashMap::default(),
            // instr_polls_processed: 0,
            // instr_polls_at_time: DHashMap::default(),
            next_poll_scheduled: None,
            migrating: false,
            pending_migration: None,
            migration_complete_enqueued: false,
            active_migration_flows: DHashMap::default(),
            jobs_migrating: DHashSet::default(),
            migration_flows_by_job: DHashMap::default(),
            pending_job_migrations: DHashMap::default(),
            migration_dependents: DHashMap::default(),
            post_migration_delay_us: 10000000,
            scratch_job_ids: Vec::new(),
            scratch_worker_ids: Vec::new(),
        };
        // System module initialization
        let topo_ref = this.network_sim.topology();
        this.system_module.on_init(&this.context, topo_ref, &mut this.scheduler, &mut this.flow_scheduler);
        // Drain any timer requests from on_init
        this.drain_pending_timers();
        this
    }

    /// Sets the delay (in microseconds) that jobs wait before restarting after migration completes.
    /// Default is 0 (no delay).
    pub fn set_post_migration_delay_us(&mut self, delay_us: u64) {
        self.post_migration_delay_us = delay_us;
    }

    /// Sets the post-migration delay as a multiple of a job's iteration time.
    /// Since iteration time varies by job, this uses a fixed estimate in microseconds.
    /// `num_iterations` is the number of iteration equivalents to wait.
    /// `iteration_time_us` is the estimated iteration time in microseconds.
    pub fn set_post_migration_delay_iterations(&mut self, num_iterations: u64, iteration_time_us: u64) {
        self.post_migration_delay_us = num_iterations * iteration_time_us;
    }

    /// Called when a worker reports IterationCompleted or JobCompleted.
    /// Aggregates per-job barriers and updates context.job_iterations when all workers complete an iteration.
    /// When all workers complete the iteration, triggers the start of the next iteration for all workers.
    fn mark_worker_iteration_complete(&mut self, job_id: JobId, worker_id: WorkerId, iter_idx: usize) {
        use std::collections::hash_map::Entry;
        let barrier_satisfied = match self.worker_iter_barrier.entry(job_id) {
            Entry::Occupied(mut e) => {
                let (cur_iter, set) = e.get_mut();
                if *cur_iter != iter_idx { *cur_iter = iter_idx; set.clear(); }
                set.insert(worker_id);
                let total_workers = self.jobs[&job_id].workers.len();
                if set.len() == total_workers {
                    let mut map = self.context.job_iterations.borrow_mut();
                    if let Some((total, completed)) = map.get_mut(&job_id) {
                        *completed = completed.saturating_add(1).min(*total);
                    }
                    *cur_iter += 1;
                    set.clear();
                    true // All workers done with this iteration
                } else {
                    false
                }
            }
            Entry::Vacant(e) => {
                let mut set = DHashSet::default();
                set.insert(worker_id);
                e.insert((iter_idx, set));
                // Check if this is a single-worker job
                let total_workers = self.jobs[&job_id].workers.len();
                if total_workers == 1 {
                    let mut map = self.context.job_iterations.borrow_mut();
                    if let Some((total, completed)) = map.get_mut(&job_id) {
                        *completed = completed.saturating_add(1).min(*total);
                    }
                    true // Single worker job, barrier immediately satisfied
                } else {
                    false
                }
            }
        };
        
        // If barrier satisfied, check for pending migrations or start next iteration
        if barrier_satisfied {
            // Check if this job has a pending migration
            if self.pending_job_migrations.contains_key(&job_id) {
                // Handle migration barrier - don't start next iteration yet
                self.handle_job_reached_migration_barrier(job_id);
            } else {
                // No migration pending - start next iteration for all workers
                // First pass: collect worker info without holding mutable borrow of self
                let worker_info: Vec<(WorkerId, usize)> = if let Some(job) = self.jobs.get(&job_id) {
                    job.workers.iter()
                        .filter(|(_, w)| w.current_iteration < w.total_iterations)
                        .map(|(&wid, w)| (wid, w.template_events.len()))
                        .collect()
                } else {
                    Vec::new()
                };
                
                // Second pass: generate event IDs (requires &mut self)
                let worker_event_ids: Vec<(WorkerId, Vec<usize>)> = worker_info.into_iter()
                    .map(|(wid, template_count)| (wid, self.next_event_ids(template_count)))
                    .collect();
                
                // Third pass: apply to workers
                if let Some(job) = self.jobs.get_mut(&job_id) {
                    for (wid, event_ids) in worker_event_ids {
                        if let Some(worker) = job.workers.get_mut(&wid) {
                            worker.start_next_iteration(event_ids);
                        }
                    }
                }
            }
        }
    }
    
    /// Adds a new ML job arrival at the specified time
    pub fn add_job_arrival(&mut self, arrival_time_us: u64, mut job: MLJob) -> JobId {
        let job_id = job.id;
        // Update the job's submit_time to match the arrival time
        job.submit_time_us = arrival_time_us;
        self.jobs.insert(job_id, job);
        
        self.event_queue.push(MLEvent {
            time_us: arrival_time_us,
            kind: MLEventKind::JobArrival,
            job_id: Some(job_id),
            worker_id: None,
            event_id: None,
            flow_id: None,
            timer_id: None,
        });
        
        job_id
    }
    
    /// Advances the simulation by processing the next event
    /// Returns Some(event_kind) if an event was processed, None if simulation is complete
    pub fn advance_next_step(&mut self) -> Option<MLEventKind> {
        // Before processing any event, poll the flow scheduler for flows ready now.
        self.poll_and_install_ready_flows();
        
        let network_time = self.network_sim.peek_time();
        let ml_time = self.event_queue.peek().map(|e| e.time_us).unwrap_or(u64::MAX);

        //self.dump_bandwidth();
        
        if ml_time < network_time {
            //println!("Processing ML event at time {}", ml_time);
            let ml_event = self.event_queue.pop().unwrap();
            self.now_us = ml_event.time_us;
            // propagate time to shared context
            self.context.time_us.set(self.now_us);
            let kind = ml_event.kind;
            
            match kind {
                MLEventKind::JobArrival => self.handle_job_arrival(ml_event.job_id.unwrap()),
                MLEventKind::ComputeCompletion => {
                    self.handle_compute_completion(
                        ml_event.job_id.unwrap(),
                        ml_event.worker_id.unwrap(),
                        ml_event.event_id.unwrap()
                    );
                }
                MLEventKind::FlowSendReady => {
                    self.handle_flow_send_ready(
                        ml_event.job_id.unwrap(),
                        ml_event.worker_id.unwrap(),
                        ml_event.event_id.unwrap()
                    );
                }
                MLEventKind::FlowReceiveReady => {
                    self.handle_flow_receive_ready(
                        ml_event.job_id.unwrap(),
                        ml_event.worker_id.unwrap(),
                        ml_event.event_id.unwrap()
                    );
                }
                MLEventKind::FlowSendComplete => {
                    self.handle_flow_send_complete(ml_event.flow_id.unwrap());
                }
                MLEventKind::JobComplete => {
                    self.handle_job_completion(ml_event.job_id.unwrap());
                }
                MLEventKind::Reconfigure => self.handle_reconfigure(),
                MLEventKind::MigrationBegin => self.handle_migration_begin(),
                MLEventKind::MigrationEnd => self.handle_migration_end(),
                MLEventKind::FlowSchedulerPoll => {
                    // // instrumentation (disabled to save memory):
                    // self.instr_polls_processed += 1;
                    // let cnt = self.instr_polls_at_time.entry(self.now_us).or_insert(0);
                    // *cnt += 1;
                    // clear dedup guard if this poll is the scheduled one
                    if self.next_poll_scheduled == Some(self.now_us) {
                        self.next_poll_scheduled = None;
                    }
                    self.poll_and_install_ready_flows();
                }
                MLEventKind::PostMigrationRestart => {
                    if let Some(job_id) = ml_event.job_id {
                        self.restart_job_after_migration(job_id);
                    }
                }
                MLEventKind::SystemTimer => {
                    if let Some(timer_id) = ml_event.timer_id {
                        self.handle_system_timer(timer_id);
                    }
                }
                MLEventKind::NetworkEvent => {
                    // NetworkEvent should not appear in ML event queue - it's handled separately
                    panic!("NetworkEvent should not be in ML event queue");
                }
            }
            
            // Try to schedule new work for all running jobs
            self.schedule_ready_work();
            
            Some(kind)
        } else {
            // Process the network event.
            //println!("Processing network event at time {}", network_time);
            if let Some((event_kind, flow_id)) = self.network_sim.advance_next_step() {
                self.now_us = self.network_sim.now_us;
                // propagate time to shared context
                self.context.time_us.set(self.now_us);

                if event_kind == EventKind::Completion {
                    // Schedule flow completion event
                    self.event_queue.push(MLEvent {
                        time_us: self.now_us,
                        kind: MLEventKind::FlowSendComplete,
                        job_id: None,
                        worker_id: None,
                        event_id: None,
                        flow_id: Some(flow_id),
                        timer_id: None,
                    });
                }
                Some(MLEventKind::NetworkEvent)
            } else {
                // Simulation completed; print instrumentation summaries if available
                // self.try_print_delay_scheduler_stats();
                // self.print_poll_duplicates_summary();
                None // Simulation complete
            }
        }
    }
    
    /// Returns the current simulation time
    pub fn current_time_us(&self) -> u64 {
        self.now_us
    }
    
    /// Returns statistics about completed jobs
    pub fn get_job_statistics(&self) -> Vec<(JobId, u64, u64, Option<u64>)> {
        self.jobs.values()
            .filter(|job| job.state == JobState::Completed)
            .map(|job| (
                job.id,
                job.submit_time_us,
                job.start_time_us.unwrap_or(0),
                job.completion_time_us
            ))
            .collect()
    }
    
    /// Returns all jobs for debugging purposes
    pub fn get_all_jobs(&self) -> &DHashMap<JobId, MLJob> {
        &self.jobs
    }
    
    /// Prints the state of all actively running jobs for debugging
    pub fn dump_cluster_state(&self) {
        println!("{} ClusterState", self.now_us);
        
        for (host_index, job_id) in self.host_assignment.iter().enumerate() {
            if let Some(job_id) = job_id {
                println!("{}: {}", host_index, job_id);
            } else {
                println!("{}: -1", host_index);
            }
        }
    }
    
    /// Prints a human-readable summary of the entire cluster placement
    fn print_cluster_placement(&self) {
        println!("{} Placement", self.now_us);
        // Build per-host occupancy from context placements
        let placements = self.context.placements.borrow();
        let mut host_to_job: Vec<Option<JobId>> = vec![None; self.host_assignment.len()];
        for (jid, w2h) in placements.iter() {
            for (_wid, host) in w2h.iter() {
                if *host < host_to_job.len() {
                    host_to_job[*host] = Some(*jid);
                }
            }
        }
        for (host_index, job_id) in host_to_job.iter().enumerate() {
            if let Some(jid) = job_id {
                println!("  host {} -> job {}", host_index, jid);
            } else {
                println!("  host {} -> -", host_index);
            }
        }
        // Per-job worker placements from context
        let mut job_ids: Vec<JobId> = placements.keys().copied().collect();
        job_ids.sort_unstable();
        let active = self.context.active_jobs.borrow();
        for jid in job_ids {
            let maybe_name = active.get(&jid).and_then(|a| a.name.clone());
            if let Some(name) = maybe_name {
                println!("  job {} (\"{}\"):", jid, name);
            } else {
                println!("  job {}:", jid);
            }
            if let Some(w2h) = placements.get(&jid) {
                let mut pairs: Vec<(&WorkerId, &usize)> = w2h.iter().collect();
                pairs.sort_by_key(|(wid, _)| **wid);
                for (wid, host) in pairs {
                    println!("    worker {} -> host {}", wid, host);
                }
            }
        }
    }
    
    // --- Internal helper methods ---
    
    /// Generates a globally unique event ID
    fn next_event_id(&mut self) -> usize {
        let id = self.next_event_id;
        self.next_event_id += 1;
        id
    }
    
    /// Generates a batch of globally unique event IDs
    fn next_event_ids(&mut self, count: usize) -> Vec<usize> {
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(self.next_event_id());
        }
        ids
    }
    
    /// Handles a new job arrival
    fn handle_job_arrival(&mut self, job_id: JobId) {
        println!("{} JobArrival {}", self.now_us, job_id);
        
        // If migrations are in progress, don't try to schedule - just enqueue
        if !self.jobs_migrating.is_empty() {
            println!("{} JobQueued {} (migrations in progress)", self.now_us, job_id);
            self.scheduler.enqueue_job(job_id);
            return;
        }
        
        // FIFO enforcement: If there are already jobs in the queue, enqueue this one too
        // to maintain arrival order. Only try direct scheduling if queue is empty.
        if self.scheduler.has_queued_jobs() {
            println!("{} JobQueued {} (queue not empty)", self.now_us, job_id);
            self.scheduler.enqueue_job(job_id);
            // Try to schedule queued jobs in order
            self.try_schedule_queued_jobs();
            return;
        }
        
        // Attempt to schedule the job; keep mutable job borrow scope minimal
        let scheduled = {
            if let Some(job) = self.jobs.get_mut(&job_id) {
                if self.scheduler.try_schedule_job(job, self.network_sim.topology(), &self.host_busy) {
                    println!("{} JobScheduled {}", self.now_us, job_id);

                    // Job was scheduled successfully
                    job.mark_scheduled(self.now_us);

                    // DEBUG: Print hosts being claimed on job scheduling
                    if DEBUG_MIGRATION {
                        let mut hosts_to_claim: Vec<_> = job.get_assigned_hosts().iter().copied().collect();
                        hosts_to_claim.sort();
                        println!("{} DEBUG schedule_job job={} claiming_hosts={:?}", self.now_us, job_id, hosts_to_claim);
                    }

                    // Mark assigned hosts as busy
                    for &host_index in job.get_assigned_hosts().iter() {
                        // DEBUG: Check if host is already busy (collision!)
                        if DEBUG_MIGRATION && self.host_busy[host_index] {
                            println!("{} DEBUG   COLLISION: host {} already busy, owned by {:?}, but job {} claiming it!",
                                self.now_us, host_index, self.host_assignment[host_index], job_id);
                        }
                        self.host_busy[host_index] = true;
                        self.host_assignment[host_index] = Some(job_id);
                    }

                    // Start the job and initialize all workers
                    job.mark_running(self.now_us);
                    self.active_jobs.insert(job_id);
                    // Mirror into shared context as a read-only snapshot
                    let snapshot = Self::make_active_job_snapshot(&*job);
                    self.context.active_jobs.borrow_mut().insert(job_id, snapshot);
                    
                    // Initialize iteration tracking in context
                    self.context.job_iterations.borrow_mut().insert(job_id, (job.total_iterations, 0));
                    // Mirror placement into context
                    self.context.placements.borrow_mut().insert(job_id, job.worker_to_host.clone());
                    // Populate inverse placement map for this job's workers
                    {
                        let mut h2w = self.context.host_to_worker.borrow_mut();
                        for (&wid, &host) in job.worker_to_host.iter() {
                            debug_assert!(host < h2w.len(), "Assigned host index {} out of range {}", host, h2w.len());
                            h2w[host] = Some((job_id, wid));
                        }
                    }
                    // Populate per-worker neighbor maps for this job
                    {
                        let mut send_neighbors = self.context.worker_send_neighbors.borrow_mut();
                        let mut recv_neighbors = self.context.worker_recv_neighbors.borrow_mut();
                        for (&wid, worker) in job.workers.iter() {
                            send_neighbors.insert((job_id, wid), worker.get_send_neighbors());
                            recv_neighbors.insert((job_id, wid), worker.get_receive_neighbors());
                        }
                    }
                    // Initialize per-worker per-iteration total and zero sent
                    {
                        let mut progress = self.context.worker_send_progress.borrow_mut();
                        for (&wid, worker) in job.workers.iter() {
                            let per_iter_total: u64 = worker
                                .template_events
                                .iter()
                                .filter(|ev| ev.kind == WorkerEventKind::FlowSend)
                                .map(|ev| ev.flow_send.as_ref().map(|f| f.size_bytes).unwrap_or(0))
                                .sum();
                            progress.insert((job_id, wid), (0u64, per_iter_total));
                        }
                    }
                   
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };

        if !scheduled {
            // Job couldn't be scheduled, add it to the scheduler's queue
            self.scheduler.enqueue_job(job_id);
            return;
        }

        // Linearize ranks left-to-right in topology to minimize ring cross-ToR flows.
        // This ensures the ring visits all workers on one ToR before moving to the next,
        // producing at most 1 cross-ToR flow per ToR per ring.
        self.reassign_ranks_after_migration(job_id);
        // Rebuild worker_send_progress with potentially reassigned worker IDs
        {
            let job = &self.jobs[&job_id];
            let mut progress = self.context.worker_send_progress.borrow_mut();
            progress.retain(|(jid, _), _| *jid != job_id);
            for (&wid, worker) in job.workers.iter() {
                let per_iter_total: u64 = worker
                    .template_events
                    .iter()
                    .filter(|ev| ev.kind == WorkerEventKind::FlowSend)
                    .map(|ev| ev.flow_send.as_ref().map(|f| f.size_bytes).unwrap_or(0))
                    .sum();
                progress.insert((job_id, wid), (0u64, per_iter_total));
            }
        }

        // Print placement of the entire cluster now that scheduling occurred
        self.print_cluster_placement();

        // Reset per-job flow completion counters due to active-set change
        {
            let mut m = self.context.flow_completions_per_job.borrow_mut();
            m.clear();
            for jid in self.active_jobs.iter() { m.insert(*jid, 0); }
        }

        // Notify system module that a job has been scheduled (no active job borrow)
        let topo_ref = self.network_sim.topology();
        let job_ref = &self.jobs[&job_id];
        self.system_module.on_job_scheduled(
            self.now_us,
            &self.context,
            job_id,
            job_ref,
            topo_ref,
            &mut self.scheduler,
            &mut self.flow_scheduler,
        );
        // Drain any timer requests from on_job_scheduled
        self.drain_pending_timers();
        // Request reconfiguration to update routing/scheduling for the new job
        self.request_reconfigure();

        // Start all workers with first iteration
        let worker_ids: Vec<WorkerId> = {
            let job = &self.jobs[&job_id];
            job.workers.keys().copied().collect()
        };
        let template_counts: HashMap<WorkerId, usize> = {
            let job = &self.jobs[&job_id];
            worker_ids
                .iter()
                .map(|&worker_id| (worker_id, job.workers[&worker_id].template_events.len()))
                .collect()
        };

        // Pre-allocate event IDs for all workers
        let mut event_id_batches = HashMap::new();
        for &worker_id in &worker_ids {
            let template_count = template_counts[&worker_id];
            let event_ids = self.next_event_ids(template_count);
            event_id_batches.insert(worker_id, event_ids);
        }

        for worker_id in worker_ids {
            let job = self.jobs.get_mut(&job_id).unwrap();
            let worker = job.workers.get_mut(&worker_id).unwrap();
            // Pass shared context into the worker
            worker.set_context(self.context.clone());
            let event_ids = event_id_batches.remove(&worker_id).unwrap();
            worker.start(event_ids);
        }
    }
    
    /// Handles completion of a compute event
    fn handle_compute_completion(&mut self, job_id: JobId, worker_id: WorkerId, event_id: usize) {
        //println!("{} Compute {} {}", self.now_us, job_id, worker_id);
        // Remove from running compute tracking
        self.running_compute.remove(&(job_id, worker_id));
        
        let job = self.jobs.get_mut(&job_id)
            .unwrap_or_else(|| panic!("Job {} not found during compute completion", job_id));
        let worker = job.get_worker_mut(worker_id)
            .unwrap_or_else(|| panic!("Worker {} not found in job {} during compute completion", worker_id, job_id));
        
        match worker.notify_event_completed_ex(event_id) {
            WorkerNotifyResult::EventDone => {}
            WorkerNotifyResult::IterationCompleted { iteration_idx } => {
                self.mark_worker_iteration_complete(job_id, worker_id, iteration_idx);
            }
            WorkerNotifyResult::JobCompleted => {
                // Treat as completion of the worker's final iteration
                let iter_idx = self.jobs[&job_id].workers[&worker_id].current_iteration.saturating_sub(1);
                self.mark_worker_iteration_complete(job_id, worker_id, iter_idx);
            }
        }
    }
    
    /// Handles when a flow send event is ready to execute
    fn handle_flow_send_ready(&mut self, job_id: JobId, worker_id: WorkerId, event_id: usize) {
        //println!("{} FlowSend {} {}", self.now_us, job_id, worker_id);
        // First, get the necessary information without holding mutable borrows
        let (dst_worker, src_host, dst_host, size_bytes) = {
            let job = self.jobs.get(&job_id)
                .unwrap_or_else(|| panic!("Job {} not found during flow send", job_id));
            
            let worker = job.get_worker(worker_id)
                .unwrap_or_else(|| panic!("Worker {} not found in job {} during flow send", worker_id, job_id));
            
            // The event should be currently running
            let current_event = worker.get_running_event(event_id)
                .unwrap_or_else(|| panic!("Event {} not running for worker {} in job {} during flow send", event_id, worker_id, job_id));
            
            let flow_send = current_event.flow_send.as_ref()
                .unwrap_or_else(|| panic!("Current event {} for worker {} in job {} is not a flow send event", 
                                        event_id, worker_id, job_id));
            
            let src_host = worker.host_index;
            let dst_host = job.get_worker_host(flow_send.dst_worker)
                .unwrap_or_else(|| panic!("Destination worker {} not found in job {} during flow send", 
                                        flow_send.dst_worker, job_id));
            
            (flow_send.dst_worker, src_host, dst_host, flow_send.size_bytes)
        };
       
        // Check if the pending receive event exists
        let pending_key = (job_id, worker_id, dst_worker);
        if let Some(receive_event_id) = self.pending_receives.remove(&pending_key) {
            self.start_flow(job_id, worker_id, dst_worker, src_host, dst_host, size_bytes, event_id, receive_event_id);
        } else {
            self.pending_sends.insert(pending_key, event_id);
        }
    }
    
    /// Handles when a flow receive event becomes ready to start
    fn handle_flow_receive_ready(&mut self, job_id: JobId, worker_id: WorkerId, event_id: usize) {
        //println!("{} FlowReceive {} {}", self.now_us, job_id, worker_id);
        // Get the flow information from the receive event
        let (src_worker, src_host, dst_host, size_bytes) = {
            let job = self.jobs.get(&job_id)
                .unwrap_or_else(|| panic!("Job {} not found during flow receive ready", job_id));
            
            let worker = job.get_worker(worker_id)
                .unwrap_or_else(|| panic!("Worker {} not found in job {} during flow receive ready", worker_id, job_id));
            
            let receive_event = worker.get_running_event(event_id)
                .unwrap_or_else(|| panic!("Event {} not running for worker {} in job {} during flow receive ready", event_id, worker_id, job_id));
            
            let flow_receive = receive_event.flow_receive.as_ref()
                .unwrap_or_else(|| panic!("Event {} for worker {} in job {} is not a flow receive event", event_id, worker_id, job_id));

            let src_host = job.get_worker_host(flow_receive.src_worker)
                .unwrap_or_else(|| panic!("Source worker {} not found in job {} during flow receive ready", flow_receive.src_worker, job_id));
            
            let dst_host = worker.host_index;
            
            (flow_receive.src_worker, src_host, dst_host, flow_receive.size_bytes)
        };

        // Check if the pending send event exists
        let pending_key = (job_id, src_worker, worker_id);
        if let Some(send_event_id) = self.pending_sends.remove(&pending_key) {
            self.start_flow(job_id, src_worker, worker_id, src_host, dst_host, size_bytes, send_event_id, event_id);
        } else {
            self.pending_receives.insert(pending_key, event_id);
        }
    }

    fn start_flow(&mut self, job_id: JobId, src_worker: WorkerId, dst_worker: WorkerId, src_host: usize, dst_host: usize, size_bytes: u64, send_event_id: usize, receive_event_id: usize) {
        //println!("{} FlowScheduled {} {} {} {} {} {} {} {}", self.now_us, job_id, src_worker, dst_worker, src_host, dst_host, size_bytes, send_event_id, receive_event_id);
        // Determine the job-local stable flow index for this send template
        let job = self.jobs.get_mut(&job_id).expect("job must exist when starting flow");
        // We use the sender's running event's template id as a stable key
        let template_id = {
            let worker = job.get_worker(src_worker).unwrap();
            let ev = worker.get_running_event(send_event_id).unwrap();
            ev.template_id
        };
        let key = (src_worker, template_id);
        let job_flow_idx = job.send_template_to_flow_idx.get(&key)
            .copied()
            .unwrap_or_else(|| panic!("Missing preassigned flow index for (worker {}, template_id {}) in job {}", src_worker, template_id, job_id));

        // Determine current iteration index from the destination worker (both should be in same iteration)
        let iter_idx = {
            let worker = job.get_worker(dst_worker).unwrap();
            worker.current_iteration
        };

        // Enqueue into the ML flow scheduler. The scheduler may request a wake time.
        let next = self.flow_scheduler.enqueue_flow(self.now_us, QueuedFlow {
            job_id,
            job_flow_idx,
            iter_idx,
            src_worker,
            dst_worker,
            send_event_id,
            receive_event_id,
            src_host,
            dst_host,
            size_bytes,
        });
        if let Some(t) = next { 
            self.event_queue.push(MLEvent { time_us: t.max(self.now_us), kind: MLEventKind::FlowSchedulerPoll, job_id: None, worker_id: None, event_id: None, flow_id: None, timer_id: None });
        }
    }
    
    /// Handles when a flow send completes in the network
    fn handle_flow_send_complete(&mut self, flow_id: FlowId) {
        // First check if this is a migration flow
        if let Some(info) = self.active_migration_flows.remove(&flow_id) {
            println!(
                "{} MigrationFlowComplete job={} worker={} src={} -> dst={}",
                self.now_us, info.job_id, info.worker_id, info.src_host, info.dst_host
            );
            
            // Remove from per-job tracking
            let job_done = if let Some(flows) = self.migration_flows_by_job.get_mut(&info.job_id) {
                flows.remove(&flow_id);
                flows.is_empty()
            } else {
                false
            };
            
            // If this job's migration is complete, complete it and restart the job
            if job_done {
                let completed_job_id = info.job_id;
                self.complete_job_migration(completed_job_id);
            }
            return;
        }
        
        // Regular job flow completion
        let removed = {
            let mut map = self.waiting_flows.borrow_mut();
            map.remove(&flow_id)
        };
        if let Some((job_id, _job_flow_idx, _iter_idx, src_worker, dst_worker, send_event_id, receive_event_id)) = removed {
            // Remove the flow to job mapping
            self.flow_to_job.remove(&flow_id);
            // Increment per-job completed flow count
            {
                let mut m = self.context.flow_completions_per_job.borrow_mut();
                *m.entry(job_id).or_insert(0) += 1;
            }
            
            self.complete_flow_send(job_id, src_worker, send_event_id);
            self.complete_flow_receive(job_id, dst_worker, receive_event_id);
        } else {
            panic!("Flow {} not found during flow send complete", flow_id);
        }
    }

    /// Poll the flow scheduler and install any flows that are ready to start now.
    fn poll_and_install_ready_flows(&mut self) {
        let now = self.now_us.max(self.network_sim.now_us);
        let (ready, next_poll) = self.flow_scheduler.poll_ready(now);
        for f in ready {
            let flow_id = self.network_sim.add_flow_arrival(
                now,
                f.src_host,
                f.dst_host,
                f.size_bytes,
                f.job_flow_idx,
            );
            self.waiting_flows.borrow_mut().insert(flow_id, (f.job_id, f.job_flow_idx, f.iter_idx, f.src_worker, f.dst_worker, f.send_event_id, f.receive_event_id));
            self.flow_to_job.insert(flow_id, f.job_id);
        }
        if let Some(t) = next_poll {
            let wake = t.max(now);
            let should_push = match self.next_poll_scheduled { None => true, Some(cur) => wake < cur };
            if should_push {
                self.event_queue.push(MLEvent { time_us: wake, kind: MLEventKind::FlowSchedulerPoll, job_id: None, worker_id: None, event_id: None, flow_id: None, timer_id: None });
                self.next_poll_scheduled = Some(wake);
            }
        }
    }

    /// Request a reconfiguration event to update routing/scheduling for future flows.
    /// Active in-flight flows continue on their existing paths.
    pub fn request_reconfigure(&mut self) {
        if self.reconfigure_requested {
            return;
        }
        self.reconfigure_requested = true;
        self.event_queue.push(MLEvent {
            time_us: self.now_us,
            kind: MLEventKind::Reconfigure,
            job_id: None,
            worker_id: None,
            event_id: None,
            flow_id: None,
            timer_id: None,
        });
    }

    /// Handle reconfiguration: update routing and flow scheduling for future flows.
    /// Active in-flight flows continue on their existing paths.
    fn handle_reconfigure(&mut self) {
        self.reconfigure_requested = false;
        
        // Skip reconfiguration if migrations are in progress to avoid corrupting internal state.
        // A reconfigure will be triggered after migrations complete.
        if self.migrating || !self.jobs_migrating.is_empty() || !self.pending_job_migrations.is_empty() {
            println!("{} Reconfigure (deferred - migrations in progress)", self.now_us);
            self.reconfigure_deferred_for_migration = true;
            return;
        }
        
        println!("{} Reconfigure", self.now_us);
        
        let pending_plan: Option<MigrationPlan>;
        {
            let topo_ref = self.network_sim.topology();
            // Notify system module; it may reconfigure routing/scheduling and optionally request migration
            pending_plan = self.system_module.on_reconfigure(
                self.now_us,
                &self.context,
                topo_ref,
                &mut self.scheduler,
                &mut self.flow_scheduler,
            );
        }

        if let Some(plan) = pending_plan {
            // Set up per-job migration state and dependencies instead of starting immediately.
            // Migrations will start when jobs reach their iteration barriers.
            self.setup_pending_migrations(plan);
        }
        
        // Drain any timer requests from the system module
        self.drain_pending_timers();
    }

    /// Handles a system timer event by calling the system module's on_timer callback.
    fn handle_system_timer(&mut self, timer_id: TimerId) {
        let topo_ref = self.network_sim.topology();
        self.system_module.on_timer(
            self.now_us,
            &self.context,
            timer_id,
            topo_ref,
            &mut self.scheduler,
            &mut self.flow_scheduler,
        );
        
        // Drain any timer requests scheduled by the callback
        self.drain_pending_timers();
    }

    /// Drains pending timer requests from context and schedules them as events.
    fn drain_pending_timers(&mut self) {
        let timers: Vec<TimerRequest> = self.context.pending_timers.borrow_mut().drain(..).collect();
        for req in timers {
            self.event_queue.push(MLEvent {
                time_us: req.fire_at_us,
                kind: MLEventKind::SystemTimer,
                job_id: None,
                worker_id: None,
                event_id: None,
                flow_id: None,
                timer_id: Some(req.timer_id),
            });
        }
    }
    
    /// Sets up per-job migration state and dependency tracking.
    /// Migrations will start when all dependencies are satisfied and jobs reach their iteration barriers.
    fn setup_pending_migrations(&mut self, plan: MigrationPlan) {
        println!("{} SetupMigrations jobs={}", self.now_us, plan.jobs.len());
        
        // First, compute worker migrations for each job
        let all_migrations = self.compute_worker_migrations(&plan);
        
        // Group migrations by job
        let mut migrations_by_job: DHashMap<JobId, Vec<WorkerMigrationInfo>> = DHashMap::default();
        for mig in all_migrations {
            migrations_by_job.entry(mig.job_id).or_default().push(mig);
        }
        
        // For each job in the plan, compute which destination hosts it needs
        // and which other jobs currently occupy those hosts
        let mut job_dest_hosts: DHashMap<JobId, DHashSet<usize>> = DHashMap::default();
        for job_mig in &plan.jobs {
            let job_id = job_mig.job_id;
            let mut dest_hosts = DHashSet::default();
            for &host in job_mig.worker_to_host.values() {
                dest_hosts.insert(host);
            }
            job_dest_hosts.insert(job_id, dest_hosts);
        }
        
        // Build the set of jobs in the migration plan for fast lookup
        let migrating_jobs: DHashSet<JobId> = plan.jobs.iter().map(|j| j.job_id).collect();
        
        // For each job, find dependencies: other jobs that occupy destination hosts
        // and are also part of this migration (so they'll free those hosts)
        for job_mig in &plan.jobs {
            let job_id = job_mig.job_id;
            let dest_hosts = job_dest_hosts.get(&job_id).cloned().unwrap_or_default();
            
            // Find which jobs currently occupy the destination hosts
            let mut waiting_for = DHashSet::default();
            for &host in &dest_hosts {
                if let Some(occupying_job) = self.host_assignment[host] {
                    if occupying_job != job_id {
                        if migrating_jobs.contains(&occupying_job) {
                            // Depend on jobs that are also in the migration plan
                            // (they will free these hosts when they reach their barrier)
                            waiting_for.insert(occupying_job);
                        } else {
                            // CRITICAL: destination host is occupied by a job NOT in the migration plan.
                            // This is an invalid migration plan - we cannot move a job to a host
                            // that's occupied by a non-migrating job.
                            panic!(
                                "[Migration] INVALID PLAN: Job {} wants to migrate to host {}, \
                                but that host is occupied by job {} which is NOT part of the migration plan. \
                                The migration planner (FIFO/ILP) produced an invalid solution.",
                                job_id, host, occupying_job
                            );
                        }
                    }
                }
            }
            
            // Add reverse dependencies
            for &dep_job in &waiting_for {
                self.migration_dependents.entry(dep_job).or_default().insert(job_id);
            }
            
            // Create the pending migration state
            let moves = migrations_by_job.remove(&job_id).unwrap_or_default();
            self.pending_job_migrations.insert(job_id, PendingJobMigration {
                moves,
                waiting_for,
                at_barrier: false,
                flows_started: false,
            });
            
            println!("{} PendingMigration job={} moves={} waiting_for={:?}", 
                self.now_us, job_id, 
                self.pending_job_migrations[&job_id].moves.len(),
                self.pending_job_migrations[&job_id].waiting_for);
        }
        
        // Store the overall plan for reference
        self.pending_migration = Some(plan);
        self.migrating = true;
        
        // Check if any jobs are already at their iteration barrier and can start migrating
        let pending_jobs: Vec<JobId> = self.pending_job_migrations.keys().copied().collect();
        for job_id in pending_jobs {
            self.try_start_job_migration(job_id);
        }
    }
    
    /// Called when a job reaches its iteration barrier while a migration is pending.
    /// Updates the job's migration state and tries to start migrations for this job
    /// and any jobs that were waiting for it.
    fn handle_job_reached_migration_barrier(&mut self, job_id: JobId) {
        println!("{} JobReachedMigrationBarrier job={}", self.now_us, job_id);
        
        // Mark this job as at_barrier
        if let Some(state) = self.pending_job_migrations.get_mut(&job_id) {
            state.at_barrier = true;
        }
        
        // Try to start this job's migration (if dependencies satisfied)
        self.try_start_job_migration(job_id);
        
        // Check if any jobs were waiting for this job to reach its barrier
        if let Some(dependents) = self.migration_dependents.remove(&job_id) {
            for dep_job_id in dependents {
                // Remove this job from the dependent's waiting_for set
                if let Some(state) = self.pending_job_migrations.get_mut(&dep_job_id) {
                    state.waiting_for.remove(&job_id);
                }
                // Try to start the dependent's migration
                self.try_start_job_migration(dep_job_id);
            }
        }
    }
    
    /// Tries to start migration flows for a job if conditions are met:
    /// - Job is at its iteration barrier
    /// - All dependencies are satisfied (jobs it's waiting for have reached their barriers)
    /// - Flows haven't been started yet
    fn try_start_job_migration(&mut self, job_id: JobId) {
        let should_start = if let Some(state) = self.pending_job_migrations.get(&job_id) {
            state.at_barrier && state.waiting_for.is_empty() && !state.flows_started
        } else {
            false
        };
        
        if should_start {
            self.start_job_migration_flows(job_id);
        }
    }
    
    /// Starts migration flows for a specific job.
    /// Called when the job has reached its barrier and all dependencies are satisfied.
    fn start_job_migration_flows(&mut self, job_id: JobId) {
        println!("{} StartJobMigrationFlows job={}", self.now_us, job_id);
        
        // Mark flows as started
        let moves = if let Some(state) = self.pending_job_migrations.get_mut(&job_id) {
            state.flows_started = true;
            state.moves.clone()
        } else {
            return;
        };
        
        // Apply placement update for this job using the pending migration plan
        // Clone the mapping to avoid borrow conflict
        let new_worker_to_host = if let Some(plan) = &self.pending_migration {
            plan.jobs.iter()
                .find(|j| j.job_id == job_id)
                .map(|job_mig| job_mig.worker_to_host.clone())
        } else {
            None
        };
        
        if let Some(mapping) = new_worker_to_host {
            self.apply_single_job_migration(job_id, &mapping);
        }
        
        // Mark this job as actively migrating
        self.jobs_migrating.insert(job_id);
        
        // Notify flow scheduler about this job's migration
        self.flow_scheduler.on_migration_begin(self.now_us, &self.context, &[job_id]);
        
        // Create migration flows for each worker that moved
        let mut job_flows = DHashSet::default();
        for info in moves {
            if info.src_host == info.dst_host {
                // Worker didn't actually move, no migration flow needed
                continue;
            }
            if info.model_size_bytes == 0 {
                // No data to transfer
                continue;
            }
            
            println!(
                "{} MigrationFlow job={} worker={} src_host={} -> dst_host={} size={}",
                self.now_us, info.job_id, info.worker_id, info.src_host, info.dst_host, info.model_size_bytes
            );
            
            // Create a network flow from src_host to dst_host
            let flow_id = self.network_sim.add_flow_arrival(
                self.now_us,
                info.src_host,
                info.dst_host,
                info.model_size_bytes,
                0, // migration flows don't have a job_flow_idx
            );
            
            // Register the flow in waiting_flows so routers can look it up
            let mig_flow_idx = migration_flow_idx(info.worker_id);
            self.waiting_flows.borrow_mut().insert(
                flow_id,
                (info.job_id, mig_flow_idx, 0, info.worker_id, info.worker_id, usize::MAX, usize::MAX)
            );
            self.flow_to_job.insert(flow_id, info.job_id);
            
            // Track this flow
            job_flows.insert(flow_id);
            self.active_migration_flows.insert(flow_id, info);
        }
        
        if job_flows.is_empty() {
            // No flows to wait for - job can restart immediately
            println!("{} JobMigrationComplete job={} (no flows needed)", self.now_us, job_id);
            self.complete_job_migration(job_id);
        } else {
            self.migration_flows_by_job.insert(job_id, job_flows);
        }
    }
    
    /// Applies a migration for a single job, updating placements.
    /// 
    /// IMPORTANT: In swap migrations, multiple jobs may exchange hosts. We must NOT
    /// free a host that is a destination for another job in the migration batch.
    fn apply_single_job_migration(&mut self, job_id: JobId, new_worker_to_host: &DHashMap<WorkerId, usize>) {
        // Collect all destination hosts from the entire migration plan to avoid
        // incorrectly freeing hosts that another job is migrating TO.
        let all_dst_hosts: DHashSet<usize> = if let Some(plan) = &self.pending_migration {
            plan.jobs.iter()
                .flat_map(|jm| jm.worker_to_host.values().copied())
                .collect()
        } else {
            DHashSet::default()
        };
        
        if DEBUG_MIGRATION {
            let mut dst_list: Vec<_> = all_dst_hosts.iter().copied().collect();
            dst_list.sort();
            println!("{} DEBUG apply_single_job_migration job={} all_dst_hosts={:?}", self.now_us, job_id, dst_list);
        }
        
        if let Some(job) = self.jobs.get_mut(&job_id) {
            if DEBUG_MIGRATION {
                let mut old_hosts: Vec<_> = job.worker_to_host.values().copied().collect();
                old_hosts.sort();
                println!("{} DEBUG   job={} old_hosts={:?}", self.now_us, job_id, old_hosts);
                
                let mut new_hosts: Vec<_> = new_worker_to_host.values().copied().collect();
                new_hosts.sort();
                println!("{} DEBUG   job={} new_hosts_from_migration={:?}", self.now_us, job_id, new_hosts);
            }
            
            // Clear previous inverse mapping entries for this job
            {
                let mut h2w = self.context.host_to_worker.borrow_mut();
                for &old_host in job.worker_to_host.values() {
                    if old_host < h2w.len() {
                        h2w[old_host] = None;
                    }
                }
            }
            
            // Free old hosts ONLY if they're not a destination for any job in this migration batch
            let mut freed_hosts = Vec::new();
            let mut protected_hosts = Vec::new();
            for &old_host in job.worker_to_host.values() {
                if old_host < self.host_busy.len() && !all_dst_hosts.contains(&old_host) {
                    self.host_busy[old_host] = false;
                    self.host_assignment[old_host] = None;
                    freed_hosts.push(old_host);
                } else if all_dst_hosts.contains(&old_host) {
                    protected_hosts.push(old_host);
                }
            }
            if DEBUG_MIGRATION {
                freed_hosts.sort();
                protected_hosts.sort();
                println!("{} DEBUG   job={} freed_hosts={:?} protected_hosts={:?}", self.now_us, job_id, freed_hosts, protected_hosts);
            }
            
            // Merge current placement with new placement
            for (&wid, &new_host) in new_worker_to_host.iter() {
                job.worker_to_host.insert(wid, new_host);
            }
            
            // Assign new hosts
            let mut assigned_hosts = Vec::new();
            for &new_host in job.worker_to_host.values() {
                if new_host < self.host_busy.len() {
                    // DEBUG: Check if we're overwriting another job's host
                    if DEBUG_MIGRATION && self.host_busy[new_host] && self.host_assignment[new_host] != Some(job_id) {
                        println!("{} DEBUG   WARNING: host {} busy, owned by {:?}, overwriting with job={}",
                            self.now_us, new_host, self.host_assignment[new_host], job_id);
                    }
                    self.host_busy[new_host] = true;
                    self.host_assignment[new_host] = Some(job_id);
                    assigned_hosts.push(new_host);
                }
            }
            if DEBUG_MIGRATION {
                assigned_hosts.sort();
                println!("{} DEBUG   job={} final_assigned_hosts={:?}", self.now_us, job_id, assigned_hosts);
            }
            
            // Update workers' host_index
            for (wid, host) in job.worker_to_host.iter() {
                if let Some(w) = job.workers.get_mut(wid) {
                    w.host_index = *host;
                }
            }
            
            // Mirror to context
            self.context.placements.borrow_mut().insert(job_id, job.worker_to_host.clone());
            
            // Update inverse placement for new mapping
            {
                let mut h2w = self.context.host_to_worker.borrow_mut();
                for (&wid, &host) in job.worker_to_host.iter() {
                    if host < h2w.len() {
                        h2w[host] = Some((job_id, wid));
                    }
                }
            }
        }
    }
    
    /// Called when all migration flows for a job have completed.
    /// Either restarts the job immediately or schedules a delayed restart.
    fn complete_job_migration(&mut self, job_id: JobId) {
        println!("{} CompleteJobMigration job={}", self.now_us, job_id);
        
        // Remove from pending migrations
        self.pending_job_migrations.remove(&job_id);
        self.jobs_migrating.remove(&job_id);
        self.migration_flows_by_job.remove(&job_id);
        
        // Notify flow scheduler that this job can resume
        self.flow_scheduler.on_migration_end(self.now_us);
        
        // Reassign ranks to optimize ring ordering (left-to-right in topology)
        // This must happen BEFORE on_migration_end so routes are computed with new ranks
        self.reassign_ranks_after_migration(job_id);
        
        // Notify system module that this job's migration is complete
        // The system module will re-record the job and recompute routes
        {
            let job = self.jobs.get(&job_id).expect("job must exist");
            let topo_ref = self.network_sim.topology();
            self.system_module.on_migration_end(
                self.now_us,
                &self.context,
                job_id,
                job,
                topo_ref,
                &mut self.scheduler,
                &mut self.flow_scheduler,
            );
        }
        // Drain any timer requests from on_migration_end
        self.drain_pending_timers();
        
        // Either restart immediately or schedule a delayed restart
        if self.post_migration_delay_us > 0 {
            // Schedule delayed restart
            let restart_time = self.now_us + self.post_migration_delay_us;
            println!("{} SchedulePostMigrationRestart job={} delay={}us restart_at={}",
                self.now_us, job_id, self.post_migration_delay_us, restart_time);
            self.event_queue.push(MLEvent {
                time_us: restart_time,
                kind: MLEventKind::PostMigrationRestart,
                job_id: Some(job_id),
                worker_id: None,
                event_id: None,
                flow_id: None,
                timer_id: None,
            });
        } else {
            // Restart immediately
            self.restart_job_after_migration(job_id);
        }
        
        // Check if all migrations are complete (but not job restarts - those may be delayed)
        if self.pending_job_migrations.is_empty() {
            println!("{} AllMigrationsComplete", self.now_us);
            self.migrating = false;
            self.pending_migration = None;
            self.migration_dependents.clear();
            
            // Print cluster placement after all migrations complete
            self.print_cluster_placement();
            
            // Try to schedule any queued jobs now that migrations are complete
            self.try_schedule_queued_jobs();
            
            // Trigger reconfiguration if one was deferred during migration
            if self.reconfigure_deferred_for_migration {
                self.reconfigure_deferred_for_migration = false;
                self.request_reconfigure();
            }
        }
    }
    
    /// Actually restarts a job after migration by starting the next iteration for all workers.
    /// Called either immediately after complete_job_migration or after a delay via PostMigrationRestart event.
    fn restart_job_after_migration(&mut self, job_id: JobId) {
        println!("{} RestartJobAfterMigration job={}", self.now_us, job_id);
        
        // Start next iteration for all workers in this job
        let worker_info: Vec<(WorkerId, usize)> = if let Some(job) = self.jobs.get(&job_id) {
            job.workers.iter()
                .filter(|(_, w)| w.current_iteration < w.total_iterations)
                .map(|(&wid, w)| (wid, w.template_events.len()))
                .collect()
        } else {
            Vec::new()
        };
        
        let worker_event_ids: Vec<(WorkerId, Vec<usize>)> = worker_info.into_iter()
            .map(|(wid, template_count)| (wid, self.next_event_ids(template_count)))
            .collect();
        
        if let Some(job) = self.jobs.get_mut(&job_id) {
            for (wid, event_ids) in worker_event_ids {
                if let Some(worker) = job.workers.get_mut(&wid) {
                    worker.start_next_iteration(event_ids);
                }
            }
        }
    }
    
    /// Reassigns worker ranks after migration to optimize ring ordering.
    /// 
    /// Within each segment (pipeline stage), workers are reordered so that
    /// their WorkerIds are assigned left-to-right in the topology. This ensures
    /// ring flows (e.g., AllReduce) travel in a consistent direction through
    /// the topology, minimizing path conflicts.
    /// 
    /// This involves:
    /// 1. Swapping worker objects between WorkerId keys
    /// 2. Rewriting template dst_worker/src_worker references
    /// 3. Rebuilding derived state (flow indices, context caches)
    fn reassign_ranks_after_migration(&mut self, job_id: JobId) {
        let job = match self.jobs.get(&job_id) {
            Some(j) => j,
            None => return,
        };
        
        // Determine segments: for pipeline jobs, each stage is a segment
        // For non-pipeline jobs, the whole job is one segment
        let segments: Vec<Vec<WorkerId>> = if let Some(pipeline_info) = &job.pipeline_stages {
            (0..pipeline_info.num_stages)
                .map(|stage| pipeline_info.stage_workers(stage).collect())
                .collect()
        } else {
            // Single segment with all workers
            vec![job.workers.keys().copied().collect()]
        };
        
        // Collect all old_to_new mappings for all segments first
        let mut all_old_to_new: HashMap<WorkerId, WorkerId> = HashMap::new();
        
        for segment_worker_ids in &segments {
            if segment_worker_ids.is_empty() {
                continue;
            }
            
            // Get current host for each worker in segment
            let mut worker_hosts: Vec<(WorkerId, usize)> = segment_worker_ids.iter()
                .filter_map(|&wid| {
                    job.worker_to_host.get(&wid).map(|&host| (wid, host))
                })
                .collect();
            
            if worker_hosts.is_empty() {
                continue;
            }
            
            // Sort by host index (left-to-right in topology)
            // CONTRACT: Topologies must assign host indices such that sorting by
            // host_index gives left-to-right ordering (e.g., leaf * hosts_per_leaf + offset)
            worker_hosts.sort_by_key(|&(_, host)| host);
            
            // Compute old_id -> new_id mapping
            // Target IDs are the sorted segment worker IDs
            let mut sorted_target_ids: Vec<WorkerId> = segment_worker_ids.clone();
            sorted_target_ids.sort();
            
            for (i, &(old_id, _)) in worker_hosts.iter().enumerate() {
                let new_id = sorted_target_ids[i];
                if old_id != new_id {
                    all_old_to_new.insert(old_id, new_id);
                }
            }
        }
        
        // If no changes needed, return early
        if all_old_to_new.is_empty() {
            return;
        }
        
        println!(
            "{} RankReassignment job={} swaps={}",
            self.now_us, job_id, all_old_to_new.len()
        );
        
        // Now mutate the job
        let job = self.jobs.get_mut(&job_id).unwrap();
        
        // Step 1: Extract and reinsert worker objects with new IDs
        // We need to handle this carefully to avoid overwriting during the swap
        let workers_to_move: Vec<WorkerId> = all_old_to_new.keys().copied().collect();
        let mut extracted: HashMap<WorkerId, crate::simulator::ml_worker::MLWorker> = HashMap::new();
        
        for &old_id in &workers_to_move {
            if let Some(worker) = job.workers.remove(&old_id) {
                extracted.insert(old_id, worker);
            }
        }
        
        for (old_id, mut worker) in extracted {
            let new_id = all_old_to_new[&old_id];
            worker.id = new_id;
            job.workers.insert(new_id, worker);
        }
        
        // Step 2: Update flow destinations based on FlowKind
        // Each flow kind has different semantics for how to remap after rank reassignment.
        use crate::simulator::ml_worker::FlowKind;
        
        let num_workers = job.num_workers;
        let ring_count = job.ring_count;
        
        // For pipeline jobs, get workers_per_stage
        let workers_per_stage = job.pipeline_stages.as_ref()
            .map(|p| p.workers_per_stage)
            .unwrap_or(num_workers);
        let num_stages = job.pipeline_stages.as_ref()
            .map(|p| p.num_stages)
            .unwrap_or(1);
        
        // For ring flows within a pipeline stage, compute the DP stride
        let dp_stride = if let Some(ref pipeline_info) = job.pipeline_stages {
            let rings_per_stage = pipeline_info.rings_per_stage;
            if rings_per_stage > 0 { workers_per_stage / rings_per_stage } else { 1 }
        } else {
            ring_count  // For non-pipeline jobs, stride = ring_count
        };
        
        for (&worker_id, worker) in job.workers.iter_mut() {
            // For pipeline jobs, determine stage and local position
            let stage = worker_id / workers_per_stage;
            let stage_start = stage * workers_per_stage;
            let local_id = worker_id - stage_start;
            
            for event in &mut worker.template_events {
                // Handle sends
                if let Some(send) = &mut event.flow_send {
                    match send.flow_kind {
                        FlowKind::Ring => {
                            // Ring flow: dst = (new_worker_id + stride) % segment_size
                            if job.pipeline_stages.is_some() {
                                // Within-stage DP ring
                                let next_local = (local_id + dp_stride) % workers_per_stage;
                                send.dst_worker = stage_start + next_local;
                            } else {
                                // Non-pipeline strided ring
                                send.dst_worker = (worker_id + dp_stride) % num_workers;
                            }
                        }
                        FlowKind::Pipeline => {
                            // Pipeline flow: dst = new_worker_id + workers_per_stage (forward)
                            // or dst = new_worker_id - workers_per_stage (backward)
                            // Determine direction: if old dst was in a later stage, it's forward
                            let old_dst = send.dst_worker;
                            let old_dst_stage = old_dst / workers_per_stage;
                            if old_dst_stage > stage && stage < num_stages - 1 {
                                // Forward: send to same local position in next stage
                                send.dst_worker = worker_id + workers_per_stage;
                            } else if old_dst_stage < stage && stage > 0 {
                                // Backward: send to same local position in previous stage
                                send.dst_worker = worker_id - workers_per_stage;
                            }
                            // else: keep as-is (shouldn't happen for valid pipeline)
                        }
                        FlowKind::AllToAll | FlowKind::Other => {
                            // Remap using old_to_new mapping
                            if let Some(&new_dst) = all_old_to_new.get(&send.dst_worker) {
                                send.dst_worker = new_dst;
                            }
                        }
                    }
                }
                
                // Handle receives
                if let Some(recv) = &mut event.flow_receive {
                    match recv.flow_kind {
                        FlowKind::Ring => {
                            // Ring flow: src = (new_worker_id - stride + segment_size) % segment_size
                            if job.pipeline_stages.is_some() {
                                // Within-stage DP ring
                                let prev_local = (local_id + workers_per_stage - dp_stride) % workers_per_stage;
                                recv.src_worker = stage_start + prev_local;
                            } else {
                                // Non-pipeline strided ring
                                recv.src_worker = (worker_id + num_workers - dp_stride) % num_workers;
                            }
                        }
                        FlowKind::Pipeline => {
                            // Pipeline flow: src = new_worker_id - workers_per_stage (forward recv)
                            // or src = new_worker_id + workers_per_stage (backward recv)
                            let old_src = recv.src_worker;
                            let old_src_stage = old_src / workers_per_stage;
                            if old_src_stage < stage && stage > 0 {
                                // Forward recv: receive from same local position in previous stage
                                recv.src_worker = worker_id - workers_per_stage;
                            } else if old_src_stage > stage && stage < num_stages - 1 {
                                // Backward recv: receive from same local position in next stage
                                recv.src_worker = worker_id + workers_per_stage;
                            }
                            // else: keep as-is
                        }
                        FlowKind::AllToAll | FlowKind::Other => {
                            // Remap using old_to_new mapping
                            if let Some(&new_src) = all_old_to_new.get(&recv.src_worker) {
                                recv.src_worker = new_src;
                            }
                        }
                    }
                }
            }
        }
        
        // Step 3: Rebuild job-level derived state
        job.rebuild_worker_to_host();
        job.rebuild_flow_indices();
        
        // Step 4: Update context placements
        self.context.placements.borrow_mut().insert(job_id, job.worker_to_host.clone());
        
        // Step 5: Rebuild host_to_worker inverse mapping for this job
        self.rebuild_host_to_worker_for_job(job_id);
        
        // Step 6: Rebuild neighbor caches for this job
        self.rebuild_neighbor_caches_for_job(job_id);
    }
    
    /// Rebuilds the host_to_worker inverse mapping for a specific job.
    fn rebuild_host_to_worker_for_job(&mut self, job_id: JobId) {
        let job = match self.jobs.get(&job_id) {
            Some(j) => j,
            None => return,
        };
        
        let mut h2w = self.context.host_to_worker.borrow_mut();
        
        // Clear old entries for this job
        for entry in h2w.iter_mut() {
            if let Some((jid, _)) = entry {
                if *jid == job_id {
                    *entry = None;
                }
            }
        }
        
        // Set new entries
        for (&wid, &host) in job.worker_to_host.iter() {
            if host < h2w.len() {
                h2w[host] = Some((job_id, wid));
            }
        }
    }
    
    /// Rebuilds the worker neighbor caches for a specific job.
    fn rebuild_neighbor_caches_for_job(&mut self, job_id: JobId) {
        let job = match self.jobs.get(&job_id) {
            Some(j) => j,
            None => return,
        };
        
        // Clear old entries for this job
        {
            let mut send_neighbors = self.context.worker_send_neighbors.borrow_mut();
            let mut recv_neighbors = self.context.worker_recv_neighbors.borrow_mut();
            
            let send_keys: Vec<(JobId, WorkerId)> = send_neighbors.keys()
                .copied()
                .filter(|(jid, _)| *jid == job_id)
                .collect();
            for k in send_keys {
                send_neighbors.remove(&k);
            }
            
            let recv_keys: Vec<(JobId, WorkerId)> = recv_neighbors.keys()
                .copied()
                .filter(|(jid, _)| *jid == job_id)
                .collect();
            for k in recv_keys {
                recv_neighbors.remove(&k);
            }
        }
        
        // Rebuild entries
        {
            let mut send_neighbors = self.context.worker_send_neighbors.borrow_mut();
            let mut recv_neighbors = self.context.worker_recv_neighbors.borrow_mut();
            
            for (&wid, worker) in job.workers.iter() {
                send_neighbors.insert((job_id, wid), worker.get_send_neighbors());
                recv_neighbors.insert((job_id, wid), worker.get_receive_neighbors());
            }
        }
    }

    fn handle_migration_begin(&mut self) {
        if self.migrating { return; }
        let plan = match self.pending_migration.take() { Some(p) => p, None => return };
        self.migrating = true;
        self.migration_complete_enqueued = false;
        self.active_migration_flows.clear();
        self.jobs_migrating.clear();
        self.migration_flows_by_job.clear();
        
        // Collect migration info for each worker that's actually moving
        let migrations = self.compute_worker_migrations(&plan);
        
        // Mark all affected jobs as migrating (they will be un-paused per-job as their flows complete)
        for job_migration in &plan.jobs {
            self.jobs_migrating.insert(job_migration.job_id);
        }
        
        // Notify flow scheduler about affected jobs; hosts will be remapped in-place for queued flows
        let affected: Vec<JobId> = plan.jobs.iter().map(|j| j.job_id).collect();
        self.flow_scheduler.on_migration_begin(self.now_us, &self.context, &affected);
        
        // Apply migration atomically (update placements)
        self.apply_migration_plan(&plan);
        
        // Create migration flows for each worker that moved
        for info in migrations {
            if info.src_host == info.dst_host {
                // Worker didn't actually move, no migration flow needed
                continue;
            }
            if info.model_size_bytes == 0 {
                // No data to transfer
                continue;
            }
            
            println!(
                "{} MigrationFlow job={} worker={} src_host={} -> dst_host={} size={}",
                self.now_us, info.job_id, info.worker_id, info.src_host, info.dst_host, info.model_size_bytes
            );
            
            // Create a network flow from src_host to dst_host
            let flow_id = self.network_sim.add_flow_arrival(
                self.now_us,
                info.src_host,
                info.dst_host,
                info.model_size_bytes,
                0, // migration flows don't have a job_flow_idx
            );
            
            // Register the flow in waiting_flows so routers can look it up
            // Migration flows use a special job_flow_idx based on worker_id to avoid conflicts
            let mig_flow_idx = migration_flow_idx(info.worker_id);
            self.waiting_flows.borrow_mut().insert(
                flow_id,
                (info.job_id, mig_flow_idx, 0, info.worker_id, info.worker_id, usize::MAX, usize::MAX)
            );
            self.flow_to_job.insert(flow_id, info.job_id);
            
            // Track this flow for per-job completion
            self.migration_flows_by_job
                .entry(info.job_id)
                .or_default()
                .insert(flow_id);
            self.active_migration_flows.insert(flow_id, info);
        }
        
        // Check for jobs that had no actual migrations (no workers moved or zero-size)
        // These jobs can be un-paused immediately
        let jobs_without_flows: Vec<JobId> = self.jobs_migrating.iter()
            .copied()
            .filter(|jid| !self.migration_flows_by_job.contains_key(jid))
            .collect();
        for job_id in jobs_without_flows {
            println!("{} JobMigrationComplete job={} (no flows needed)", self.now_us, job_id);
            self.jobs_migrating.remove(&job_id);
            
            // Reassign ranks even for jobs without flows (for left-to-right ordering)
            self.reassign_ranks_after_migration(job_id);
            
            // Notify system module that this job's migration is complete
            let job = self.jobs.get(&job_id).expect("job must exist");
            let topo_ref = self.network_sim.topology();
            self.system_module.on_migration_end(
                self.now_us,
                &self.context,
                job_id,
                job,
                topo_ref,
                &mut self.scheduler,
                &mut self.flow_scheduler,
            );
            // Drain any timer requests from on_migration_end
            self.drain_pending_timers();
        }

        // If all jobs are done (no flows for any), complete the migration phase
        if self.jobs_migrating.is_empty() {
            println!("{} MigrationComplete (no flows needed)", self.now_us);
            self.event_queue.push(MLEvent { 
                time_us: self.now_us, 
                kind: MLEventKind::MigrationEnd, 
                job_id: None, 
                worker_id: None, 
                event_id: None, 
                flow_id: None,
                timer_id: None,
            });
        }
    }
    
    /// Compute migration information for each worker that will move to a new host.
    /// Returns a list of WorkerMigrationInfo for workers whose host is changing.
    fn compute_worker_migrations(&self, plan: &MigrationPlan) -> Vec<WorkerMigrationInfo> {
        let mut migrations = Vec::new();
        let placements = self.context.placements.borrow();
        
        for job_migration in &plan.jobs {
            let job_id = job_migration.job_id;
            let current_placement = placements.get(&job_id);
            
            if let Some(job) = self.jobs.get(&job_id) {
                for (&worker_id, &new_host) in &job_migration.worker_to_host {
                    // Get current host for this worker
                    let current_host = current_placement
                        .and_then(|p| p.get(&worker_id))
                        .copied()
                        .unwrap_or_else(|| {
                            // Fall back to job's worker host_index if not in placements
                            job.workers.get(&worker_id)
                                .map(|w| w.host_index)
                                .unwrap_or(0)
                        });
                    
                    if current_host != new_host {
                        // Worker is moving - compute model size
                        let model_size = self.compute_worker_model_size(job_id, worker_id);
                        migrations.push(WorkerMigrationInfo {
                            job_id,
                            worker_id,
                            src_host: current_host,
                            dst_host: new_host,
                            model_size_bytes: model_size,
                        });
                    }
                }
            }
        }
        
        migrations
    }
    
    /// Compute the model size for a worker by summing all FlowSend sizes in its template.
    /// This is a placeholder implementation - the actual model size calculation may change.
    fn compute_worker_model_size(&self, job_id: JobId, worker_id: WorkerId) -> u64 {
        if let Some(job) = self.jobs.get(&job_id) {
            if let Some(worker) = job.workers.get(&worker_id) {
                return worker.template_events.iter()
                    .filter_map(|ev| {
                        if ev.kind == WorkerEventKind::FlowSend {
                            ev.flow_send.as_ref().map(|f| f.size_bytes)
                        } else {
                            None
                        }
                    })
                    .sum();
            }
        }
        0
    }

    fn handle_migration_end(&mut self) {
        if !self.migrating { return; }
        self.migrating = false;
        self.migration_complete_enqueued = false;
        // Print cluster placement after migration completes
        self.print_cluster_placement();
        // Schedule a poll in case there are queued flows
        if let Some(t) = self.flow_scheduler.on_migration_end(self.now_us) {
            self.event_queue.push(MLEvent { time_us: t.max(self.now_us), kind: MLEventKind::FlowSchedulerPoll, job_id: None, worker_id: None, event_id: None, flow_id: None, timer_id: None });
        }
        
        // Try to schedule any queued jobs now that migrations are complete
        self.try_schedule_queued_jobs();
    }

    /// Completes a flow send event
    fn complete_flow_send(&mut self, job_id: JobId, worker_id: WorkerId, event_id: usize) {
        //println!("{} FlowSendComplete {} {}", self.now_us, job_id, worker_id);
        let send_size_bytes = {
            let job = self.jobs.get(&job_id)
                .unwrap_or_else(|| panic!("Job {} not found during flow send completion", job_id));
            let worker = &job.workers[&worker_id];
            worker
                .get_running_event(event_id)
                .and_then(|ev| ev.flow_send.as_ref().map(|f| f.size_bytes))
                .unwrap_or(0)
        };
        
        let job = self.jobs.get_mut(&job_id)
            .unwrap_or_else(|| panic!("Job {} not found during flow send completion", job_id));
        let worker = job.get_worker_mut(worker_id)
            .unwrap_or_else(|| panic!("Worker {} not found in job {} during flow send completion", worker_id, job_id));
        // Update per-worker sent bytes progress for the current iteration
        {
            let mut map = self.context.worker_send_progress.borrow_mut();
            if let Some((sent_in_iter, per_iter_total)) = map.get_mut(&(job_id, worker_id)) {
                let new_sent = sent_in_iter.saturating_add(send_size_bytes);
                // Cap within a single iteration
                *sent_in_iter = new_sent.min(*per_iter_total);
            }
        }
        
        match worker.notify_event_completed_ex(event_id) {
            WorkerNotifyResult::EventDone => {}
            WorkerNotifyResult::IterationCompleted { iteration_idx } => {
                self.mark_worker_iteration_complete(job_id, worker_id, iteration_idx);
                // Reset per-worker progress for the next iteration
                let mut map = self.context.worker_send_progress.borrow_mut();
                if let Some((sent_in_iter, _per_iter_total)) = map.get_mut(&(job_id, worker_id)) {
                    *sent_in_iter = 0;
                }
            }
            WorkerNotifyResult::JobCompleted => {
                // Treat as completion of the worker's final iteration
                let iter_idx = worker.current_iteration.saturating_sub(1);
                self.mark_worker_iteration_complete(job_id, worker_id, iter_idx);
                // Reset per-worker progress (will be cleaned up at job completion anyway)
                let mut map = self.context.worker_send_progress.borrow_mut();
                if let Some((sent_in_iter, _per_iter_total)) = map.get_mut(&(job_id, worker_id)) {
                    *sent_in_iter = 0;
                }
            }
        }
    }

    /// Completes a flow receive event
    fn complete_flow_receive(&mut self, job_id: JobId, worker_id: WorkerId, event_id: usize) {
        //println!("{} FlowReceiveComplete {} {}", self.now_us, job_id, worker_id);
        
        let job = self.jobs.get_mut(&job_id)
            .unwrap_or_else(|| panic!("Job {} not found during flow receive completion", job_id));
        let worker = job.get_worker_mut(worker_id)
            .unwrap_or_else(|| panic!("Worker {} not found in job {} during flow receive completion", worker_id, job_id));
        
        match worker.notify_event_completed_ex(event_id) {
            WorkerNotifyResult::EventDone => {}
            WorkerNotifyResult::IterationCompleted { iteration_idx } => {
                self.mark_worker_iteration_complete(job_id, worker_id, iteration_idx);
            }
            WorkerNotifyResult::JobCompleted => {
                let iter_idx = worker.current_iteration.saturating_sub(1);
                self.mark_worker_iteration_complete(job_id, worker_id, iter_idx);
            }
        }
    }
    
    /// Handles when a job completes
    fn handle_job_completion(&mut self, job_id: JobId) {
        let completion_time = self.now_us;

        // Log job timing metrics
        let job = self.jobs.get(&job_id)
            .unwrap_or_else(|| panic!("Job {} not found during completion", job_id));
        let submit_time = job.submit_time_us;
        let schedule_time = job.schedule_time_us.unwrap_or(submit_time);
        let real_runtime = completion_time.saturating_sub(schedule_time);
        let wait_time = schedule_time.saturating_sub(submit_time);
        let total_runtime = completion_time.saturating_sub(submit_time);
        
        println!("{} JobComplete {} submit={} schedule={} complete={} wait={}us real_runtime={}us total_runtime={}us",
            self.now_us, job_id, submit_time, schedule_time, completion_time,
            wait_time, real_runtime, total_runtime);
        
        // Remove from active jobs
        self.active_jobs.remove(&job_id);
        // Mirror removal into shared context
        self.context.active_jobs.borrow_mut().remove(&job_id);
        
        // Print final placement (entire cluster) and capture assigned hosts before freeing them
        let assigned_hosts = {
            let job_ref = self.jobs.get(&job_id)
                .unwrap_or_else(|| panic!("Job {} not found during completion", job_id));
            self.print_cluster_placement();
            job_ref.get_assigned_hosts()
        };

        if DEBUG_MIGRATION {
            let mut hosts_to_free: Vec<_> = assigned_hosts.iter().copied().collect();
            hosts_to_free.sort();
            println!("{} DEBUG handle_job_completion job={} assigned_hosts={:?}", self.now_us, job_id, hosts_to_free);
        }
        
        // Free up the hosts used by this job
        // CRITICAL FIX: Only free hosts that are still owned by this job!
        // During migrations, another job may have already claimed these hosts via apply_single_job_migration.
        let mut actually_freed = Vec::new();
        let mut skipped = Vec::new();
        for &host_index in assigned_hosts.iter() {
            if self.host_assignment[host_index] == Some(job_id) {
                self.host_busy[host_index] = false;
                self.host_assignment[host_index] = None;
                actually_freed.push(host_index);
            } else {
                // Host is owned by another job (probably took it during a migration swap)
                skipped.push((host_index, self.host_assignment[host_index]));
            }
        }
        actually_freed.sort();
        if DEBUG_MIGRATION && !skipped.is_empty() {
            println!("{} DEBUG   actually_freed={:?} skipped={:?}", self.now_us, actually_freed, skipped);
        }
        // Clear inverse placement entries for hosts we actually freed
        // (not for hosts that were taken over by another job during migration)
        {
            let mut h2w = self.context.host_to_worker.borrow_mut();
            for &host_index in actually_freed.iter() {
                debug_assert!(host_index < h2w.len(), "Freed host index {} out of range {}", host_index, h2w.len());
                h2w[host_index] = None;
            }
        }
        // Remove placement from context
        self.context.placements.borrow_mut().remove(&job_id);
        // Remove neighbor maps for this job
        {
            let mut send_neighbors = self.context.worker_send_neighbors.borrow_mut();
            let mut recv_neighbors = self.context.worker_recv_neighbors.borrow_mut();
            let send_keys: Vec<(JobId, WorkerId)> = send_neighbors.keys().copied().filter(|(jid, _)| *jid == job_id).collect();
            for k in send_keys { send_neighbors.remove(&k); }
            let recv_keys: Vec<(JobId, WorkerId)> = recv_neighbors.keys().copied().filter(|(jid, _)| *jid == job_id).collect();
            for k in recv_keys { recv_neighbors.remove(&k); }
        }
        
        // Notify the scheduler that this job completed
        self.scheduler.notify_job_completed(job_id, completion_time);

        // Handle job completing while a migration is pending.
        // 
        // Background: During migration, jobs vacate their old hosts but "protect" (don't free)
        // hosts that are destinations for other jobs in the same migration batch. This allows
        // jobs to swap hosts without race conditions. However, if a destination job completes
        // before it can migrate, those protected hosts become orphaned - still marked busy but
        // never claimed by anyone.
        //
        // This cleanup handles that case by:
        // 1. Freeing any hosts that were protected for this job (since it will never claim them)
        // 2. Removing this job from the migration plan
        // 3. Unblocking jobs that were waiting for this job to reach its migration barrier
        if self.pending_job_migrations.remove(&job_id).is_some() {
            if DEBUG_MIGRATION {
                println!("{} JobCompletedDuringMigration job={} (removing from migration plan)", self.now_us, job_id);
            }
            
            // Free orphaned hosts: hosts that were protected for this job but will never be claimed.
            // A host is orphaned if:
            //   - It was a destination for this job
            //   - It's still assigned to another job that already migrated away
            //   - That job no longer tracks this host in its worker_to_host
            if let Some(ref plan) = self.pending_migration {
                if let Some(job_mig) = plan.jobs.iter().find(|j| j.job_id == job_id) {
                    let dst_hosts: Vec<usize> = job_mig.worker_to_host.values().copied().collect();
                    
                    for &host in &dst_hosts {
                        if host < self.host_busy.len() {
                            if let Some(owner_job_id) = self.host_assignment[host] {
                                if owner_job_id != job_id {
                                    let owner_still_uses_host = self.jobs.get(&owner_job_id)
                                        .map(|j| j.worker_to_host.values().any(|&h| h == host))
                                        .unwrap_or(false);
                                    
                                    if !owner_still_uses_host {
                                        // Host is orphaned - free it
                                        if DEBUG_MIGRATION {
                                            println!("{} DEBUG   freeing orphaned host {} (protected for job {}, owned by {})",
                                                self.now_us, host, job_id, owner_job_id);
                                        }
                                        self.host_busy[host] = false;
                                        self.host_assignment[host] = None;
                                        let mut h2w = self.context.host_to_worker.borrow_mut();
                                        if host < h2w.len() {
                                            h2w[host] = None;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Remove from the migration plan so future migrations don't protect hosts for this job
            if let Some(ref mut plan) = self.pending_migration {
                plan.jobs.retain(|j| j.job_id != job_id);
            }
            
            // Unblock dependent jobs that were waiting for this job
            if let Some(dependents) = self.migration_dependents.remove(&job_id) {
                for dep_job_id in dependents {
                    if let Some(state) = self.pending_job_migrations.get_mut(&dep_job_id) {
                        state.waiting_for.remove(&job_id);
                    }
                    self.try_start_job_migration(dep_job_id);
                }
            }
        }

        // Notify system module that a job has completed (no active job borrow)
        {
            let topo_ref = self.network_sim.topology();
            let job_ref = &self.jobs[&job_id];
            self.system_module.on_job_completed(
                self.now_us,
                &self.context,
                job_id,
                job_ref,
                topo_ref,
                &mut self.scheduler,
                &mut self.flow_scheduler,
            );
        }
        // Drain any timer requests from on_job_completed
        self.drain_pending_timers();
        // Request reconfiguration to update routing/scheduling after job completion
        self.request_reconfigure();

        // Try to schedule any queued jobs now that resources are available
        self.try_schedule_queued_jobs();
        // Cleanup per-worker progress entries for this job
        {
            let mut map = self.context.worker_send_progress.borrow_mut();
            let keys: Vec<(JobId, WorkerId)> = map.keys().copied().filter(|(jid, _)| *jid == job_id).collect();
            for k in keys { map.remove(&k); }
        }
        // Reset per-job flow completion counters due to active-set change
        {
            let mut m = self.context.flow_completions_per_job.borrow_mut();
            m.clear();
            for jid in self.active_jobs.iter() { m.insert(*jid, 0); }
        }
        
        // Clean up completed job to free memory
        self.context.job_iterations.borrow_mut().remove(&job_id);
        self.worker_iter_barrier.remove(&job_id);
        self.jobs.remove(&job_id);
    }
    
    /// Attempts to schedule any queued jobs
    fn try_schedule_queued_jobs(&mut self) {
        // Don't schedule new jobs while migrations are in progress or pending.
        // self.migrating is set when a migration plan is created (before flows start).
        // jobs_migrating tracks jobs with active migration flows.
        // pending_job_migrations tracks jobs waiting to start migration flows.
        if self.migrating || !self.jobs_migrating.is_empty() || !self.pending_job_migrations.is_empty() {
            return;
        }
        
        while let Some(next_job_id) = self.scheduler.get_next_job_to_schedule() {
            // Check if the job still exists and is queued
            if let Some(job) = self.jobs.get_mut(&next_job_id) {
                if job.state == crate::simulator::ml_job::JobState::Queued {
                    if self.scheduler.try_schedule_job(job, self.network_sim.topology(), &self.host_busy) {
                        println!("{} JobScheduled {}", self.now_us, next_job_id);

                        // Job was scheduled successfully
                        job.mark_scheduled(self.now_us);
                        
                        if DEBUG_MIGRATION {
                            let mut hosts_to_claim: Vec<_> = job.get_assigned_hosts().iter().copied().collect();
                            hosts_to_claim.sort();
                            println!("{} DEBUG try_schedule_queued_jobs job={} claiming_hosts={:?}", self.now_us, next_job_id, hosts_to_claim);
                        }
                        
                        // Mark assigned hosts as busy
                        for &host_index in job.get_assigned_hosts().iter() {
                            // DEBUG: Check if host is already busy (collision!)
                            if DEBUG_MIGRATION && self.host_busy[host_index] {
                                println!("{} DEBUG   COLLISION: host {} already busy, owned by {:?}, but job {} claiming it!",
                                    self.now_us, host_index, self.host_assignment[host_index], next_job_id);
                            }
                            self.host_busy[host_index] = true;
                            self.host_assignment[host_index] = Some(next_job_id);
                        }
                        
                        // Start the job and initialize all workers
                        job.mark_running(self.now_us);
                        self.active_jobs.insert(next_job_id);
                        // Mirror into shared context as a read-only snapshot
                        let snapshot = Self::make_active_job_snapshot(&*job);
                        self.context.active_jobs.borrow_mut().insert(next_job_id, snapshot);

                        // Initialize iteration tracking in context
                        self.context.job_iterations.borrow_mut().insert(next_job_id, (job.total_iterations, 0));
                        // Mirror placement into context
                        self.context.placements.borrow_mut().insert(next_job_id, job.worker_to_host.clone());
                        // Populate inverse placement map for this job's workers
                        {
                            let mut h2w = self.context.host_to_worker.borrow_mut();
                            for (&wid, &host) in job.worker_to_host.iter() {
                                debug_assert!(host < h2w.len(), "Assigned host index {} out of range {}", host, h2w.len());
                                h2w[host] = Some((next_job_id, wid));
                            }
                        }
                        // Populate per-worker neighbor maps for this job
                        {
                            let mut send_neighbors = self.context.worker_send_neighbors.borrow_mut();
                            let mut recv_neighbors = self.context.worker_recv_neighbors.borrow_mut();
                            for (&wid, worker) in job.workers.iter() {
                                send_neighbors.insert((next_job_id, wid), worker.get_send_neighbors());
                                recv_neighbors.insert((next_job_id, wid), worker.get_receive_neighbors());
                            }
                        }

                        // Print placement of the entire cluster now that scheduling occurred
                        self.print_cluster_placement();

                    } else {
                        // Job couldn't be scheduled, stop trying for now
                        break;
                    }
                } else {
                    // Job is no longer queued, remove it from scheduler and continue
                    self.scheduler.dequeue_job();
                    continue;
                }

                // At this point, any borrow of `job` has ended (scope exited). Proceed with notifications.

                // Linearize ranks left-to-right in topology to minimize ring cross-ToR flows
                self.reassign_ranks_after_migration(next_job_id);
                // Rebuild worker_send_progress with potentially reassigned worker IDs
                {
                    let job = &self.jobs[&next_job_id];
                    let mut progress = self.context.worker_send_progress.borrow_mut();
                    progress.retain(|(jid, _), _| *jid != next_job_id);
                    for (&wid, worker) in job.workers.iter() {
                        let per_iter_total: u64 = worker
                            .template_events
                            .iter()
                            .filter(|ev| ev.kind == crate::simulator::ml_worker::WorkerEventKind::FlowSend)
                            .map(|ev| ev.flow_send.as_ref().map(|f| f.size_bytes).unwrap_or(0))
                            .sum();
                        progress.insert((next_job_id, wid), (0u64, per_iter_total));
                    }
                }

                // Notify system module that a job has been scheduled
                let topo_ref = self.network_sim.topology();
                let job_ref = &self.jobs[&next_job_id];
                self.system_module.on_job_scheduled(
                    self.now_us,
                    &self.context,
                    next_job_id,
                    job_ref,
                    topo_ref,
                    &mut self.scheduler,
                    &mut self.flow_scheduler,
                );
                // Drain any timer requests from on_job_scheduled
                self.drain_pending_timers();
                // Request reconfiguration for the newly scheduled job
                self.request_reconfigure();
                        
                        // Remove this job from the scheduler's queue
                        self.scheduler.dequeue_job();
                        
                        // Start all workers with first iteration (similar to job arrival)
                        let worker_ids: Vec<crate::simulator::ml_worker::WorkerId> = {
                            let job = &self.jobs[&next_job_id];
                            job.workers.keys().copied().collect()
                        };
                        let template_counts: std::collections::HashMap<crate::simulator::ml_worker::WorkerId, usize> = {
                            let job = &self.jobs[&next_job_id];
                            worker_ids
                                .iter()
                                .map(|&worker_id| (worker_id, job.workers[&worker_id].template_events.len()))
                                .collect()
                        };
                        
                        // Pre-allocate event IDs for all workers
                        let mut event_id_batches = std::collections::HashMap::new();
                        for &worker_id in &worker_ids {
                            let template_count = template_counts[&worker_id];
                            let event_ids = self.next_event_ids(template_count);
                            event_id_batches.insert(worker_id, event_ids);
                        }
                        
                        for worker_id in worker_ids {
                            let job = self.jobs.get_mut(&next_job_id).unwrap();
                            let worker = job.workers.get_mut(&worker_id).unwrap();
                            // Pass shared context into the worker
                            worker.set_context(self.context.clone());
                            let event_ids = event_id_batches.remove(&worker_id).unwrap();
                            worker.start(event_ids);
                        }
            } else {
                // Job doesn't exist anymore, remove it from scheduler
                self.scheduler.dequeue_job();
            }
        }
    }

    /// Schedules ready work for all running jobs
    fn schedule_ready_work(&mut self) {
        let current_time = self.now_us;
        // ensure context is up-to-date (defensive)
        self.context.time_us.set(current_time);
        
        // Reuse scratch buffer for job IDs to avoid allocation
        self.scratch_job_ids.clear();
        self.scratch_job_ids.extend(
            self.jobs.iter()
                .filter(|(_, job)| job.state == JobState::Running)
                .map(|(job_id, _)| *job_id)
        );
        
        for i in 0..self.scratch_job_ids.len() {
            let job_id = self.scratch_job_ids[i];
            // Skip jobs that are currently migrating - they are paused until migration completes
            if self.jobs_migrating.contains(&job_id) {
                continue;
            }
            // Reuse scratch buffer for worker IDs
            self.scratch_worker_ids.clear();
            self.scratch_worker_ids.extend(self.jobs[&job_id].workers.keys().copied());
            
            for j in 0..self.scratch_worker_ids.len() {
                let worker_id = self.scratch_worker_ids[j];
                // Get all ready events from the worker (concurrent execution!)
                let ready_events = {
                    let job = self.jobs.get_mut(&job_id).unwrap();
                    let worker = job.workers.get_mut(&worker_id).unwrap();
                    worker.get_all_ready_events()
                };
                
                // Schedule all ready events
                for ready_event in ready_events {
                    match ready_event.kind {
                        WorkerEventKind::Compute => {
                            let completion_time = current_time + ready_event.compute_duration_us;
                            self.running_compute.insert((job_id, worker_id), (ready_event.id, completion_time));
                            
                            self.event_queue.push(MLEvent {
                                time_us: completion_time,
                                kind: MLEventKind::ComputeCompletion,
                                job_id: Some(job_id),
                                worker_id: Some(worker_id),
                                event_id: Some(ready_event.id),
                                flow_id: None,
                                timer_id: None,
                            });
                        }
                        WorkerEventKind::FlowSend => {
                            self.event_queue.push(MLEvent {
                                time_us: current_time,
                                kind: MLEventKind::FlowSendReady,
                                job_id: Some(job_id),
                                worker_id: Some(worker_id),
                                event_id: Some(ready_event.id),
                                flow_id: None,
                                timer_id: None,
                            });
                        }
                        WorkerEventKind::FlowReceive => {
                            // Flow receives start immediately but don't complete until the corresponding flow finishes
                            self.event_queue.push(MLEvent {
                                time_us: current_time,
                                kind: MLEventKind::FlowReceiveReady,
                                job_id: Some(job_id),
                                worker_id: Some(worker_id),
                                event_id: Some(ready_event.id),
                                flow_id: None,
                                timer_id: None,
                            });
                        }
                    }
                }
            }
            
            // Check if the job is completed
            let job_completed = {
                let job = &self.jobs[&job_id];
                job.is_completed()
            };
            
            if job_completed {
                // Emit a JobComplete event instead of handling completion inline
                {
                    let job = self.jobs.get_mut(&job_id).unwrap();
                    job.mark_completed(current_time);
                };
                //job.mark_completed(current_time);
                self.event_queue.push(MLEvent {
                    time_us: current_time,
                    kind: MLEventKind::JobComplete,
                    job_id: Some(job_id),
                    worker_id: None,
                    event_id: None,
                    flow_id: None,
                    timer_id: None,
                });
            }
        }
    }

    pub fn dump_bandwidth(&self) {
        println!("{} DumpBandwidthStart", self.now_us);
        let flow_rates = self.network_sim.get_rates();
        let mut job_rates = DHashMap::default();
        for (flow_id, rate) in flow_rates {
            let job_id = self.flow_to_job.get(&flow_id).unwrap();
            let prev_rate = job_rates.get(job_id).copied().unwrap_or(f64::MAX);
            if rate < prev_rate {
                job_rates.insert(job_id, rate);
            }
        }
        for job_id in self.jobs.keys() {
            if !job_rates.contains_key(job_id) {
                job_rates.insert(job_id, 0.0);
            }
        }
        for (job_id, rate) in job_rates {
            println!("{} {} {}", self.now_us, job_id, rate);
        }
        println!("{} DumpBandwidthEnd", self.now_us);
    }
} 
