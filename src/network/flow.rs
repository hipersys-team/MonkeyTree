use crate::network::routing::PathCell;

/// Unique identifier for a network flow.
pub type FlowId = usize;

/// Static description of a network flow containing its basic properties.
/// 
/// This struct holds the immutable characteristics of a flow that are
/// determined when the flow is created and don't change during simulation.
#[derive(Debug, Clone)]
pub struct FlowDesc {
    /// Unique identifier for this flow
    pub id: FlowId,
    /// Job-local stable identifier for this flow (same across iterations)
    pub job_flow_idx: usize,
    /// Source node index (host where the flow originates)
    pub src: usize,
    /// Destination node index (host where the flow terminates)
    pub dst: usize,
    /// Total size of the flow in bytes
    pub size_bytes: u64,
}

/// Dynamic state of a network flow that changes during simulation.
/// 
/// This struct tracks the current status of a flow as it progresses
/// through the network, including how much data remains and the current
/// transmission rate.
#[derive(Debug, Clone)]
pub struct FlowState {
    /// Number of bytes still to be transmitted
    pub remaining_bytes: u64,
    /// Current transmission rate in bits per second
    pub rate_bps: f64,
    /// The routing path (sequence of link IDs) for this flow
    pub path_cell: PathCell,
}

impl FlowState {
    /// Creates a new flow state for a flow of the given size and path.
    /// 
    /// # Arguments
    /// * `size_bytes` - The total size of the flow in bytes
    /// * `path` - The routing path (sequence of link IDs) for this flow
    /// 
    /// # Returns
    /// A new `FlowState` with the remaining bytes set to the total size,
    /// the initial rate set to 0.0 (will be updated by the scheduler),
    /// and the provided routing path.
    pub fn new(size_bytes: u64, path_cell: PathCell) -> Self { 
        Self { 
            remaining_bytes: size_bytes, 
            rate_bps: 0.0, 
            path_cell
        } 
    }
}
