use std::collections::VecDeque;
use crate::utils::{DHashMap, DHashSet};
use crate::network::flow::FlowId;
use crate::simulator::ml_job::JobId;
use crate::simulator::ml_simulator::MLContext;
pub use WorkerNotifyResult::*;

/// Unique identifier for ML workers
pub type WorkerId = usize;

/// Lightweight info about a ready event - avoids cloning full WorkerEvent.
/// Contains only the fields needed by the simulator to schedule the event.
#[derive(Debug, Clone, Copy)]
pub struct ReadyEventInfo {
    pub id: usize,
    pub kind: WorkerEventKind,
    /// Duration in microseconds (only valid for Compute events)
    pub compute_duration_us: u64,
}

/// Richer notification result for event completion
pub enum WorkerNotifyResult {
    EventDone,
    IterationCompleted { iteration_idx: usize },
    JobCompleted,
}

/// Types of events that an ML worker can execute
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkerEventKind {
    /// A computation event that takes a fixed amount of time
    Compute,
    /// A flow sending event that initiates network communication
    FlowSend,
    /// A flow receiving event that waits for network communication
    FlowReceive,
}

/// Classification of flow types for rank reassignment and routing.
/// This determines how destinations are recomputed after migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlowKind {
    /// Ring-based collective (AllReduce, StridedRing).
    /// After reassignment: dst = (new_worker_id + stride) % segment_size
    Ring,
    /// Pipeline parallel inter-stage transfer.
    /// After reassignment: dst = new_worker_id +/- workers_per_stage
    Pipeline,
    /// All-to-all collective communication.
    /// After reassignment: use old_to_new mapping on dst_worker
    AllToAll,
    /// Other point-to-point or unclassified flows.
    /// After reassignment: use old_to_new mapping on dst_worker
    #[default]
    Other,
}

/// A computation event that simulates compute work
#[derive(Debug, Clone)]
pub struct ComputeEvent {
    /// Duration of the computation in milliseconds
    pub duration_us: u64,
    /// Optional name/description for the computation
    pub name: Option<String>,
}

/// A flow sending event that initiates network communication
#[derive(Debug, Clone)]
pub struct FlowSendEvent {
    /// Destination worker ID to send the flow to
    pub dst_worker: WorkerId,
    /// Size of the data to send in bytes
    pub size_bytes: u64,
    /// Optional name/description for the flow
    pub name: Option<String>,
    /// Classification of this flow for routing and rank reassignment.
    pub flow_kind: FlowKind,
}

impl FlowSendEvent {
    /// Returns true if this is a ring flow (for priority routing via edge coloring).
    #[inline]
    pub fn is_ring_flow(&self) -> bool {
        self.flow_kind == FlowKind::Ring
    }
}

/// A flow receiving event that waits for network communication
#[derive(Debug, Clone)]
pub struct FlowReceiveEvent {
    /// Source worker ID to receive the flow from
    pub src_worker: WorkerId,
    /// Expected size of the data to receive in bytes
    pub size_bytes: u64,
    /// Optional name/description for the flow
    pub name: Option<String>,
    /// Classification of this flow for rank reassignment.
    pub flow_kind: FlowKind,
}

/// A generic worker event that can be one of the three types
#[derive(Debug, Clone)]
pub struct WorkerEvent {
    /// Unique ID for this event
    pub id: usize,
    /// Stable template ID for this event (does not change across iterations)
    pub template_id: usize,
    /// The type and data of the event
    pub kind: WorkerEventKind,
    /// Event-specific data
    pub compute: Option<ComputeEvent>,
    pub flow_send: Option<FlowSendEvent>,
    pub flow_receive: Option<FlowReceiveEvent>,
    /// Dependencies: list of event IDs that must complete before this event
    pub dependencies: Vec<usize>,
    /// Current state of the event
    pub state: EventState,
}

/// State of a worker event during execution
#[derive(Debug, Clone, PartialEq)]
pub enum EventState {
    /// Event is waiting for dependencies to complete
    Waiting,
    /// Event is ready to execute (dependencies satisfied)
    Ready,
    /// Event is currently executing
    Running,
    /// Event has completed successfully
    Completed,
}

impl WorkerEvent {
    /// Creates a new computation event
    pub fn new_compute(id: usize, duration_us: u64, dependencies: Vec<usize>) -> Self {
        Self {
            id,
            template_id: id,
            kind: WorkerEventKind::Compute,
            compute: Some(ComputeEvent { duration_us, name: None }),
            flow_send: None,
            flow_receive: None,
            dependencies,
            state: EventState::Waiting,
        }
    }
    
    /// Creates a new flow send event (defaults to Other flow kind)
    pub fn new_flow_send(id: usize, dst_worker: WorkerId, size_bytes: u64, dependencies: Vec<usize>) -> Self {
        Self::new_flow_send_with_kind(id, dst_worker, size_bytes, dependencies, FlowKind::Other)
    }
    
    /// Creates a new flow send event with ring flow flag (legacy compatibility)
    pub fn new_flow_send_ex(id: usize, dst_worker: WorkerId, size_bytes: u64, dependencies: Vec<usize>, is_ring_flow: bool) -> Self {
        let flow_kind = if is_ring_flow { FlowKind::Ring } else { FlowKind::Other };
        Self::new_flow_send_with_kind(id, dst_worker, size_bytes, dependencies, flow_kind)
    }
    
    /// Creates a new flow send event with explicit flow kind
    pub fn new_flow_send_with_kind(id: usize, dst_worker: WorkerId, size_bytes: u64, dependencies: Vec<usize>, flow_kind: FlowKind) -> Self {
        Self {
            id,
            template_id: id,
            kind: WorkerEventKind::FlowSend,
            compute: None,
            flow_send: Some(FlowSendEvent { dst_worker, size_bytes, name: None, flow_kind }),
            flow_receive: None,
            dependencies,
            state: EventState::Waiting,
        }
    }
    
    /// Creates a new flow receive event (defaults to Other flow kind)
    pub fn new_flow_receive(id: usize, src_worker: WorkerId, size_bytes: u64, dependencies: Vec<usize>) -> Self {
        Self::new_flow_receive_with_kind(id, src_worker, size_bytes, dependencies, FlowKind::Other)
    }
    
    /// Creates a new flow receive event with explicit flow kind
    pub fn new_flow_receive_with_kind(id: usize, src_worker: WorkerId, size_bytes: u64, dependencies: Vec<usize>, flow_kind: FlowKind) -> Self {
        Self {
            id,
            template_id: id,
            kind: WorkerEventKind::FlowReceive,
            compute: None,
            flow_send: None,
            flow_receive: Some(FlowReceiveEvent { src_worker, size_bytes, name: None, flow_kind }),
            dependencies,
            state: EventState::Waiting,
        }
    }
}

/// An ML worker that executes a DAG of events across multiple iterations
#[derive(Debug)]
pub struct MLWorker {
    // Unique identifier for the job
    pub job_id: JobId,
    /// Unique identifier for this worker
    pub id: WorkerId,
    /// Host index in the network topology where this worker is located
    pub host_index: usize,
    /// Template DAG of events to execute each iteration
    pub template_events: Vec<WorkerEvent>,
    /// Events waiting for dependencies (event_id -> event)
    waiting_events: DHashMap<usize, WorkerEvent>,
    /// Ready queue: events whose dependencies are all satisfied
    ready_queue: VecDeque<WorkerEvent>,
    /// Currently executing events (event_id -> event)
    pub running_events: DHashMap<usize, WorkerEvent>,
    /// Completed events for current iteration
    pub completed_events: Vec<WorkerEvent>,
    /// Number of unsatisfied dependencies per event (event_id -> count)
    pending_dep_count: DHashMap<usize, usize>,
    /// Reverse dependency map: event_id -> list of events that depend on it
    dependents: DHashMap<usize, Vec<usize>>,
    /// Current iteration number (0-indexed)
    pub current_iteration: usize,
    /// Total number of iterations to complete
    pub total_iterations: usize,
    /// Active network flows initiated by this worker
    pub active_flows: Vec<FlowId>,
    /// Shared simulation context (injected by the simulator)
    pub context: Option<MLContext>,
    /// Start time of the current/last iteration in ms, used to compute duration
    pub last_iteration_start_time_us: Option<u64>,
}

impl MLWorker {
    /// Creates a new ML worker
    pub fn new(job_id: JobId, id: WorkerId, host_index: usize, total_iterations: usize) -> Self {
        Self {
            job_id,
            id,
            host_index,
            template_events: Vec::new(),
            waiting_events: DHashMap::default(),
            ready_queue: VecDeque::new(),
            running_events: DHashMap::default(),
            completed_events: Vec::new(),
            pending_dep_count: DHashMap::default(),
            dependents: DHashMap::default(),
            current_iteration: 0,
            total_iterations,
            active_flows: Vec::new(),
            context: None,
            last_iteration_start_time_us: None,
        }
    }

    /// Injects the shared simulation context into the worker. This is called
    /// by the simulator once the worker has been scheduled.
    pub fn set_context(&mut self, context: MLContext) {
        self.context = Some(context);
    }
    
    /// Adds an event template that will be executed each iteration
    pub fn add_event_template(&mut self, event: WorkerEvent) {
        self.template_events.push(event);
    }
    
    /// Starts the worker by initializing the first iteration
    pub fn start(&mut self, event_ids: Vec<usize>) {
        if self.total_iterations > 0 {
            self.reset_iteration(event_ids);
        }
    }
    
    /// Returns distinct destination worker IDs this worker sends to (from templates)
    pub fn get_send_neighbors(&self) -> Vec<WorkerId> {
        let mut seen = DHashSet::default();
        let mut neighbors = Vec::new();
        for ev in &self.template_events {
            if let Some(fs) = &ev.flow_send {
                if seen.insert(fs.dst_worker) {
                    neighbors.push(fs.dst_worker);
                }
            }
        }
        neighbors
    }
    
    /// Returns distinct source worker IDs this worker receives from (from templates)
    pub fn get_receive_neighbors(&self) -> Vec<WorkerId> {
        let mut seen = DHashSet::default();
        let mut neighbors = Vec::new();
        for ev in &self.template_events {
            if let Some(fr) = &ev.flow_receive {
                if seen.insert(fr.src_worker) {
                    neighbors.push(fr.src_worker);
                }
            }
        }
        neighbors
    }
    
    /// Resets the DAG for the current iteration with new globally unique event IDs
    fn reset_iteration(&mut self, event_ids: Vec<usize>) {
        let cur_time = if let Some(ctx) = &self.context { ctx.time_us.get() } else { 0 };
        //let elapsed_us = if let Some(start_us) = self.last_iteration_start_time_us { cur_time - start_us } else { 0 };
        //println!("{} Iteration {} {} {} {}", cur_time, self.job_id, self.id, self.current_iteration, elapsed_us);
        // mark the start of the new iteration
        self.last_iteration_start_time_us = Some(cur_time);
        
        // Clear all iteration state
        self.waiting_events.clear();
        self.ready_queue.clear();
        self.completed_events.clear();
        self.running_events.clear();
        self.pending_dep_count.clear();
        self.dependents.clear();
        
        if event_ids.len() != self.template_events.len() {
            panic!("Event ID count mismatch: expected {}, got {}", self.template_events.len(), event_ids.len());
        }
        
        // Create a mapping from template IDs to new global IDs
        let mut id_mapping = std::collections::HashMap::new();
        for (i, template_event) in self.template_events.iter().enumerate() {
            id_mapping.insert(template_event.id, event_ids[i]);
        }
        
        // First pass: create events and build dependency structures
        let mut all_events = Vec::new();
        for template_event in &self.template_events {
            let new_id = id_mapping[&template_event.id];
            let new_dependencies: Vec<usize> = template_event.dependencies.iter()
                .map(|dep_id| id_mapping[dep_id])
                .collect();
                
            let mut new_event = template_event.clone();
            new_event.template_id = template_event.id;
            new_event.id = new_id;
            new_event.dependencies = new_dependencies.clone();
            new_event.state = EventState::Waiting;
            
            // Track pending dependency count
            self.pending_dep_count.insert(new_id, new_dependencies.len());
            
            // Build reverse dependency map
            for dep_id in &new_dependencies {
                self.dependents.entry(*dep_id).or_default().push(new_id);
            }
            
            all_events.push(new_event);
        }
        
        // Second pass: add events to ready queue or waiting map
        for event in all_events {
            if event.dependencies.is_empty() {
                // No dependencies - immediately ready
                self.ready_queue.push_back(event);
            } else {
                // Has dependencies - wait
                self.waiting_events.insert(event.id, event);
            }
        }
    }
    
    /// Gets all ready events that can execute concurrently.
    /// Returns lightweight ReadyEventInfo to avoid cloning full WorkerEvent structs.
    pub fn get_all_ready_events(&mut self) -> Vec<ReadyEventInfo> {
        let mut ready_events = Vec::new();
        
        // Drain the ready queue - move events into running_events (no clone!)
        while let Some(mut event) = self.ready_queue.pop_front() {
            // Extract info before moving
            let info = ReadyEventInfo {
                id: event.id,
                kind: event.kind,
                compute_duration_us: event.compute.as_ref().map_or(0, |c| c.duration_us),
            };
            
            event.state = EventState::Running;
            self.running_events.insert(event.id, event); // Move, not clone!
            ready_events.push(info);
        }
        
        ready_events
    }
    
    // Moved WorkerNotifyResult to module scope

    /// Notifies the worker that an event has completed.
    /// 
    /// When an iteration completes, this returns `IterationCompleted` but does NOT
    /// automatically start the next iteration. The simulator must call 
    /// `start_next_iteration()` to begin the next iteration (after all workers
    /// in the job have completed their current iteration - barrier synchronization).
    pub fn notify_event_completed_ex(&mut self, event_id: usize) -> WorkerNotifyResult {
        // Verify this event is running
        if !self.running_events.contains_key(&event_id) {
            panic!("Event {} is not currently running for worker {}", event_id, self.id);
        }
        
        // Mark the event as completed
        let mut completed_event = self.running_events.remove(&event_id).unwrap();
        completed_event.state = EventState::Completed;
        self.completed_events.push(completed_event);
        
        // Update dependency counts for all events that depend on this one
        if let Some(dependent_ids) = self.dependents.remove(&event_id) {
            for dep_id in dependent_ids {
                if let Some(count) = self.pending_dep_count.get_mut(&dep_id) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        // All dependencies satisfied - move to ready queue
                        if let Some(event) = self.waiting_events.remove(&dep_id) {
                            self.ready_queue.push_back(event);
                        }
                    }
                }
            }
        }
        
        // Check if current iteration is complete
        if self.is_current_iteration_complete() {
            let finished_iter = self.current_iteration;
            self.current_iteration += 1;
            
            // Check if worker is completely done (no more iterations)
            if self.current_iteration >= self.total_iterations {
                return WorkerNotifyResult::JobCompleted;
            }
            
            // Iteration complete but more iterations remain.
            // Do NOT start next iteration here - wait for barrier synchronization.
            // The simulator will call start_next_iteration() when all workers are ready.
            return WorkerNotifyResult::IterationCompleted { iteration_idx: finished_iter };
        }
        
        WorkerNotifyResult::EventDone // Current iteration still has work
    }
    
    /// Starts the next iteration with the given event IDs.
    /// Called by the simulator when all workers in the job have completed their
    /// current iteration (barrier synchronization).
    pub fn start_next_iteration(&mut self, event_ids: Vec<usize>) {
        self.reset_iteration(event_ids);
    }
    
    /// Returns true if the worker is waiting at an iteration barrier
    /// (completed current iteration but hasn't started the next one yet).
    pub fn is_waiting_for_barrier(&self) -> bool {
        self.is_current_iteration_complete() 
            && self.current_iteration < self.total_iterations
    }

    /// Backward-compatible wrapper returning true if job completed
    pub fn notify_event_completed(&mut self, event_id: usize) -> bool {
        matches!(
            self.notify_event_completed_ex(event_id),
            WorkerNotifyResult::JobCompleted
        )
    }
    
    /// Checks if the current iteration is complete (all events done)
    fn is_current_iteration_complete(&self) -> bool {
        self.waiting_events.is_empty() && self.ready_queue.is_empty() && self.running_events.is_empty()
    }
    
    /// Checks if the worker has completely finished all iterations
    pub fn is_completely_finished(&self) -> bool {
        self.current_iteration >= self.total_iterations
    }
    
    /// Checks if the worker has any more events to execute in the current iteration
    pub fn has_pending_events(&self) -> bool {
        !self.waiting_events.is_empty() || !self.ready_queue.is_empty() || !self.running_events.is_empty()
    }
    
    /// Gets a running event by ID (for flow operations)
    pub fn get_running_event(&self, event_id: usize) -> Option<&WorkerEvent> {
        self.running_events.get(&event_id)
    }
    
    /// Gets all currently running events
    pub fn get_running_events(&self) -> &DHashMap<usize, WorkerEvent> {
        &self.running_events
    }
}