use std::rc::Rc;
use std::cell::RefCell;
use petgraph::graph::NodeIndex;
use crate::network::topology::{LinkId, SingleLinkTopology, FatTreeTopology};
use crate::network::flow::FlowId;
use crate::simulator::ml_simulator::MLContext;

/// A path is an ordered list of link‑ids.
pub type Path = Vec<LinkId>;

#[derive(Clone, Debug)]
pub struct PathCell {
    pub path: Rc<RefCell<Path>>,
}

pub trait SingleLinkRouter {
    fn route(&mut self, topo: &impl SingleLinkTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell;
    fn set_context(&mut self, context: &MLContext);
    fn complete_flow(&mut self, flow_id: FlowId);
}

/// Routing trait specifically for FatTreeTopology that provides access to fat tree structure.
/// 
/// This trait allows routers to access fat tree specific methods like pod/ToR/aggregation
/// switch information for implementing advanced routing algorithms like ECMP.
pub trait FatTreeRouter {
    /// Computes a path from source to destination in a fat tree topology.
    /// 
    /// # Arguments
    /// * `fat_tree` - The fat tree topology with access to pod/ToR/core structure
    /// * `src` - The source node index
    /// * `dst` - The destination node index
    /// * `flow_id` - The unique identifier of the flow requesting the route
    /// 
    /// # Returns
    /// A `Path` (vector of link IDs) representing the route from source to destination.
    /// 
    /// # Panics
    /// May panic if no path exists between the source and destination nodes.
    fn route(&mut self, topo: &impl FatTreeTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell;
    fn set_context(&mut self, context: &MLContext);
    fn complete_flow(&mut self, flow_id: FlowId);
}

pub struct SingleLinkRoute {
    context: Option<MLContext>,
    path_cell: PathCell,
}

impl SingleLinkRoute {
    pub fn new() -> Self {
        let path: Path = vec![0];
        let path_cell = PathCell {
            path: Rc::new(RefCell::new(path))
        };
        Self {
            context: None,
            path_cell: path_cell,
        }
    }
}

impl SingleLinkRouter for SingleLinkRoute {
    fn route(&mut self, _topo: &impl SingleLinkTopology, _src: NodeIndex, _dst: NodeIndex, _flow_id: FlowId) -> PathCell {
        self.path_cell.clone()
    }

    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn complete_flow(&mut self, _flow_id: FlowId) {
        // Do nothing
    }
}

/// A debug router that allows manual specification of routing paths for testing.
/// 
/// This router is useful for testing specific network scenarios where you want
/// to control exactly which path flows take through the network, rather than
/// relying on algorithmic routing decisions.
/// 
/// The router maintains a lookup table of (source, destination) -> path mappings
/// and falls back to shortest path routing for unmapped source-destination pairs.
/// This router can work with any topology type.
#[derive(Debug, Clone)]
pub struct DebugRouter {
    context: Option<MLContext>,
    /// Manually specified routing rules: (src, dst) -> path
    routing_table: std::collections::HashMap<FlowId, PathCell>,
}

impl DebugRouter {
    pub fn new() -> Self {
        Self {
            context: None,
            routing_table: std::collections::HashMap::new(),
        }
    }

    pub fn add_route(&mut self, _src: NodeIndex, _dst: NodeIndex, flow_id: FlowId, path: Path) {
        let path_cell = PathCell {
            path: Rc::new(RefCell::new(path))
        };
        self.routing_table.insert(flow_id, path_cell);
    }

    pub fn remove_route(&mut self, _src: NodeIndex, _dst: NodeIndex, flow_id: FlowId) {
        self.routing_table.remove(&flow_id);
    }

    pub fn clear_routes(&mut self) {
        self.routing_table.clear();
    }

    pub fn num_routes(&self) -> usize {
        self.routing_table.len()
    }
}

impl FatTreeRouter for DebugRouter {
    fn route(&mut self, _topo: &impl FatTreeTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        if let Some(path_cell) = self.routing_table.get(&flow_id) {
            path_cell.clone()
        } else {
            panic!("No route found for flow {} from {} to {}", flow_id, src.index(), dst.index());
        }
    }

    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn complete_flow(&mut self, flow_id: FlowId) {
        self.routing_table.remove(&flow_id);
    }
}