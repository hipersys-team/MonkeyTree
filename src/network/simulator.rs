use std::cmp::Ordering;
use std::collections::BinaryHeap;
use indexmap::IndexMap;

use crate::network::flow::{FlowDesc, FlowId, FlowState};
use crate::network::topology::Topology;
use crate::network::alloc::BandwidthAllocator;

/// Types of events that can occur in the network simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind { 
    /// A new flow arrives and starts transmitting
    Arrival, 
    /// An existing flow completes transmission
    Completion 
}

/// Internal event structure for the discrete event simulation.
/// 
/// Events are ordered by time, with arrival events processed before completion
/// events when they occur at the same time.
#[derive(Debug, Clone)]
struct Event { 
    /// Time when this event should be processed (in microseconds)
    time_us: u64, 
    /// Type of event (arrival or completion)
    kind: EventKind, 
    /// ID of the flow associated with this event
    flow_id: FlowId, 
    /// Flow description (only present for arrival events)
    desc: Option<FlowDesc> 
}

impl Event { 
    /// Returns the ordering key for this event.
    /// 
    /// Events are ordered first by time, then by type (arrivals before completions).
    /// This ensures deterministic processing when events occur at the same time.
    fn key(&self) -> (u64, u8) { (self.time_us, if matches!(self.kind, EventKind::Arrival) {0} else {1}) } 
}

// Event ordering implementations for the priority queue (BinaryHeap)
// Note: BinaryHeap is a max-heap, so we reverse the comparison to get min-heap behavior
impl Ord for Event { fn cmp(&self, o: &Self) -> Ordering { o.key().cmp(&self.key()) } }
impl PartialOrd for Event { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl PartialEq for Event { fn eq(&self, o: &Self) -> bool { self.key() == o.key() } }
impl Eq for Event {}

/// Discrete event network simulator with max-min fair bandwidth allocation.
/// 
/// The simulator is now generic over topology types, allowing it to work with
/// any topology that implements the `NetworkTopology` trait. The topology
/// includes its own routing strategy, ensuring type safety and tight integration
/// between topology and routing algorithms.
pub struct Simulator<T: Topology, A: BandwidthAllocator> {
    /// Current simulation time in microseconds
    pub now_us: u64,
    /// Time when rates were last calculated (for tracking progress)
    last_recalc_us: u64,
    /// Network topology with embedded routing strategy
    topo: T,
    /// Counter for generating unique flow IDs
    next_id: FlowId,
    /// Static descriptions of currently active flows
    active_desc: IndexMap<FlowId, FlowDesc>,
    /// Dynamic state of currently active flows
    active_state: IndexMap<FlowId, FlowState>,
    /// Priority queue of future events, ordered by time
    queue: BinaryHeap<Event>,
    /// Bandwidth allocator
    allocator: A,
}

impl<T: Topology, A: BandwidthAllocator> Simulator<T, A> {
    /// Creates a new network simulator with the given topology.
    /// 
    /// The topology must include its own routing strategy, ensuring that
    /// routing and topology are properly integrated.
    /// 
    /// # Arguments
    /// * `topo` - The network topology with embedded routing strategy
    /// 
    /// # Returns
    /// A new simulator instance with simulation time set to 0 and no active flows.
    pub fn new(topo: T, allocator: A) -> Self {
        Self { 
            now_us: 0, 
            last_recalc_us: 0, 
            topo, 
            next_id: 0, 
            active_desc: IndexMap::new(), 
            active_state: IndexMap::new(), 
            queue: BinaryHeap::new(),
            allocator,
        }
    }

    /// Schedules a new flow to arrive at a specified time.
    /// 
    /// # Arguments
    /// * `start_us` - Time when the flow should start (in microseconds)
    /// * `src` - Source host index
    /// * `dst` - Destination host index  
    /// * `size_bytes` - Total size of the flow in bytes
    /// 
    /// # Returns
    /// The unique FlowId assigned to this flow, which can be used to track
    /// the flow's progress in subsequent simulation output.
    pub fn add_flow_arrival(&mut self, start_us: u64, src: usize, dst: usize, size_bytes: u64, job_flow_idx: usize) -> FlowId {
        let fid = self.next_id; self.next_id += 1;
        let desc = FlowDesc { id: fid, job_flow_idx, src, dst, size_bytes };
        self.queue.push(Event { time_us: start_us, kind: EventKind::Arrival, flow_id: fid, desc: Some(desc) });
        fid
    }

    /// Advances the simulation by processing the next event in chronological order.
    /// 
    /// This method processes one event from the event queue, updates the simulation
    /// time, and handles the event (either flow arrival or completion). After
    /// processing, it recalculates bandwidth allocations for all active flows.
    /// 
    /// # Returns
    /// * `Some(EventKind)` - The type of event that was processed
    /// * `None` - If there are no more events to process (simulation complete)
    pub fn advance_next_step(&mut self) -> Option<(EventKind, FlowId)> {
        let ev = self.queue.pop()?;
        self.now_us = ev.time_us;
        let kind = ev.kind;
        let flow_id = ev.flow_id;
        match kind {
            EventKind::Arrival => self.handle_arrival(ev),
            EventKind::Completion => self.handle_completion(flow_id),
        }
        Some((kind, flow_id))
    }

    pub fn peek_time(&self) -> u64 {
        if let Some(ev) = self.queue.peek() {
            ev.time_us
        } else {
            u64::MAX
        }
    }

    /// Returns the current transmission rates of all active flows.
    /// 
    /// # Returns
    /// A vector of tuples containing (FlowId, rate_bps) for each active flow.
    /// The rates are computed using max-min fair allocation and represent the
    /// current bits per second each flow is transmitting.
    pub fn get_rates(&self) -> Vec<(FlowId, f64)> { self.active_state.iter().map(|(id, st)| (*id, st.rate_bps)).collect() }

    /// Returns a reference to the underlying topology.
    /// 
    /// # Returns
    /// A reference to the topology used by this simulator.
    pub fn topology(&self) -> &T { &self.topo }

    // ---------- internals ----------
    
    /// Handles the arrival of a new flow.
    /// 
    /// When a flow arrives, this method:
    /// 1. Computes the routing path from source to destination using the topology's router
    /// 2. Adds the flow to the active flows collections
    /// 3. Triggers bandwidth reallocation for all flows
    fn handle_arrival(&mut self, ev: Event) {
        let desc = ev.desc.unwrap();
        let src_node = self.topo.get_host_by_index(desc.src).unwrap();
        let dst_node = self.topo.get_host_by_index(desc.dst).unwrap();
        //println!("src {:?} dst {:?} src_node: {:?}, dst_node: {:?} active_flows: {}", desc.src, desc.dst, src_node, dst_node, self.active_desc.len());
        let path_cell = self.topo.route(src_node, dst_node, desc.id);
        self.active_desc.insert(desc.id, desc.clone());
        self.active_state.insert(desc.id, FlowState::new(desc.size_bytes, path_cell));
        self.recalc();
    }

    /// Handles the completion of an existing flow.
    /// 
    /// When a flow completes transmission, this method:
    /// 1. Removes the flow from active flows collections
    /// 2. Triggers bandwidth reallocation for remaining flows
    fn handle_completion(&mut self, fid: FlowId) {
        self.topo.complete_flow(fid);
        self.active_desc.swap_remove(&fid);
        self.active_state.swap_remove(&fid);
        self.recalc();
    }

    /// Recalculates bandwidth allocation and reschedules completion events.
    /// 
    /// This method performs the core simulation logic:
    /// 1. Updates remaining bytes for all active flows based on progress since last calculation
    /// 2. Computes max-min fair bandwidth allocation for all active flows
    /// 3. Updates the transmission rates in flow states
    /// 4. Clears old completion events from the queue
    /// 5. Schedules new completion events based on updated rates and remaining data
    /// 
    /// Flows with rates below 1 bps are considered stalled and scheduled to complete
    /// at the maximum possible time (effectively never).
    fn recalc(&mut self) {
        // Update remaining bytes based on progress since last recalculation
        let elapsed_us = self.now_us - self.last_recalc_us;
        if elapsed_us > 0 {
            for (_fid, state) in &mut self.active_state {
                if state.rate_bps > 0.0 {
                    // bits = rate_bps * elapsed_us / 1_000_000 (convert us to seconds)
                    let bits_transmitted = state.rate_bps * (elapsed_us as f64) / 1_000_000.0;
                    let bytes_transmitted = (bits_transmitted / 8.0) as u64;
                    state.remaining_bytes = state.remaining_bytes.saturating_sub(bytes_transmitted);
                }
            }
        }
        self.last_recalc_us = self.now_us;
        
        // Allocate returns rates in the same order as active_desc/active_state iteration
        let rates = self.allocator.allocate(&self.topo, &self.active_desc, &self.active_state);
        
        // Update rates using index-based access (avoids hash lookups)
        for (i, (_fid, state)) in self.active_state.iter_mut().enumerate() {
            state.rate_bps = rates[i];
        }
        
        // clear completion events & reschedule
        self.queue.retain(|e| e.kind == EventKind::Arrival);
        
        // Iterate desc and state together using zip (both have same keys in same order)
        for ((fid, _desc), (_fid2, st)) in self.active_desc.iter().zip(self.active_state.iter()) {
            let bits = st.remaining_bytes as f64 * 8.0;
            // dur_us = bits / rate_bps * 1_000_000 (convert seconds to us)
            let dur_us = if st.rate_bps < 1.0 { u64::MAX } else { (bits / st.rate_bps * 1_000_000.0) as u64 };
            if dur_us != u64::MAX {
                self.queue.push(Event { time_us: self.now_us + dur_us, kind: EventKind::Completion, flow_id: *fid, desc: None });
            }
        }
    }
}

impl<T: Topology, A: BandwidthAllocator> Simulator<T, A> {
    /// Returns the number of currently active flows.
    pub fn active_flow_count(&self) -> usize { self.active_desc.len() }
}
