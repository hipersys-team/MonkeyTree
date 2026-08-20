pub mod topology;
pub mod routing;
pub mod alloc;
pub mod flow;
pub mod mltcp;
pub mod mltcp_topo;
pub mod mltcp_topo_bytes;
pub mod mltcp_topo_approx;
mod simulator;

// Re-export *just* what outside code should touch.
pub use simulator::{Simulator, EventKind};
pub use topology::{LinkId, SingleLinkTopology, Topology, FatTreeTopology, FatTree, SingleLink};
pub use routing::{SingleLinkRoute, DebugRouter, FatTreeRouter};