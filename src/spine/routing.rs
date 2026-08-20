use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use petgraph::graph::NodeIndex;
use twox_hash::XxHash64;

use crate::network::flow::FlowId;
use crate::network::routing::{PathCell, Path};
use crate::utils::data::DHashMap;
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::ml_job::JobId;

use super::topology::{SpineTreeTopology, SpineTreeRouter};

// -----------------------------------------------------------------------------
// SpineRouteOracle: stateless route prediction interface
// -----------------------------------------------------------------------------
/// A stateless prediction interface that lets external systems (e.g., Cassini)
/// query what path a router would choose for a given (src, dst) and a
/// deterministic flow key, without mutating the router's internal state or
/// requiring simulator context.
pub trait SpineRouteOracle {
    /// Predict the link-id path that would be chosen between `src` and `dst` for a given job.
    /// Implementations must be pure w.r.t. router state (no caching/mutation).
    fn predict_path(&self, topo: &impl SpineTreeTopology, src: NodeIndex, dst: NodeIndex, job_id: JobId) -> Path;
}

// -----------------------------------------------------------------------------
// SpineSystemRouter: system-directed router that only serves cached paths
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Default)]
pub struct SpineSystemRouter {
    context: Option<MLContext>,
    /// (job_id, job_flow_idx, iter_idx) → full link path
    template_cache: DHashMap<(JobId, usize, usize), Path>,
}

// -----------------------------------------------------------------------------
// MonkeyRouter: system-directed router keyed only by (src, dst)
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Default)]
pub struct MonkeyRouter {
    context: Option<MLContext>,
    /// (src_node, dst_node) → full link path
    template_cache: DHashMap<(NodeIndex, NodeIndex), Path>,
}

impl MonkeyRouter {
    pub fn new() -> Self {
        Self { context: None, template_cache: DHashMap::default() }
    }

    /// Clear all pre-injected templates.
    pub fn clear_templates(&mut self) { self.template_cache.clear(); }

    /// Inject a full link path template keyed by (src_node, dst_node).
    pub fn inject_template(&mut self, src: NodeIndex, dst: NodeIndex, path: Path) {
        println!("MonkeyRouterInject route (src={:?}, dst={:?}) path={:?}", src, dst, path);
        self.template_cache.insert((src, dst), path);
    }
}

impl SpineTreeRouter for MonkeyRouter {
    fn set_context(&mut self, context: &MLContext) { self.context = Some(context.clone()); }

    fn route(&mut self, _topo: &impl SpineTreeTopology, src: NodeIndex, dst: NodeIndex, _flow_id: FlowId) -> PathCell {
        let path = self.template_cache.get(&(src, dst))
            .unwrap_or_else(|| panic!("MonkeyRouter: template not injected for (src={:?}, dst={:?})", src, dst))
            .clone();
        PathCell { path: Rc::new(RefCell::new(path)) }
    }

    fn complete_flow(&mut self, _flow_id: FlowId) { }
}

impl SpineSystemRouter {
    pub fn new() -> Self {
        Self { context: None, template_cache: DHashMap::default() }
    }

    /// Clear all pre-injected templates.
    pub fn clear_templates(&mut self) { self.template_cache.clear(); }

    /// Inject a full link path template keyed by (job_id, job_flow_idx, iter_idx).
    pub fn inject_template(&mut self, job_id: JobId, job_flow_idx: usize, iter_idx: usize, path: Path) {
        println!(
            "RealSpineInjectRoute job_id={} flow_idx={} iter={} path={:?}",
            job_id, job_flow_idx, iter_idx, path
        );
        self.template_cache.insert((job_id, job_flow_idx, iter_idx), path);
    }
}

impl SpineTreeRouter for SpineSystemRouter {
    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn route(&mut self, _topo: &impl SpineTreeTopology, _src: NodeIndex, _dst: NodeIndex, flow_id: FlowId) -> PathCell {
        // Resolve job/flow/iter from context
        let ctx = self.context.as_ref().expect("SpineSystemRouter context not set");
        let map_ref = ctx.waiting_flows.borrow();
        let (job_id, job_flow_idx, iter_idx, _src_w, _dst_w, _send_eid, _recv_eid) = map_ref.get(&flow_id)
            .copied()
            .unwrap_or_else(|| panic!("SpineSystemRouter: missing mapping for flow_id {}", flow_id));
        drop(map_ref);

        // Lookup template
        let key = (job_id, job_flow_idx, iter_idx);
        let path = self.template_cache.get(&key)
            .unwrap_or_else(|| panic!("SpineSystemRouter: template not injected for (job_id={}, flow_idx={}, iter={})", job_id, job_flow_idx, iter_idx))
            .clone();
        PathCell { path: Rc::new(RefCell::new(path)) }
    }

    fn complete_flow(&mut self, _flow_id: FlowId) { }
}

/// Equal-Cost Multi-Path (ECMP) router for any two-layer spine-leaf fabric.
///
/// The algorithm is identical to classic fat-tree ECMP except there is only one
/// choice point: which spine switch to traverse.  Hash-based selection
/// guarantees flow-to-spine consistency while keeping different flows roughly
/// balanced across the spines.
#[derive(Debug, Clone)]
pub struct SpineEcmpRouter {
    context: Option<MLContext>,
    seed: u64,
    /// flow-id → shared, mutable path so that the simulator can update link
    /// utilizations in place if needed.
    path_cache: HashMap<FlowId, PathCell>,
}

impl SpineEcmpRouter {
    /// Create a new ECMP router with the given hash seed.
    pub fn new(seed: u64) -> Self {
        Self { context: None, seed, path_cache: HashMap::new() }
    }

    fn hash_flow_flow(&self, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> usize {
        let ctx = self.context.as_ref().expect("SpineEcmpRouter context not set");
        let map_ref = ctx.waiting_flows.borrow();
        let (job_id, _job_flow_idx, _iter_idx, _src_w, _dst_w, _send_eid, _recv_eid) = map_ref.get(&flow_id)
            .copied()
            .unwrap_or_else(|| panic!("SpineEcmpRouter: missing mapping for flow_id {}", flow_id));
        drop(map_ref);
        self.hash_flow(src, dst, job_id)
    }

    fn hash_flow(&self, src: NodeIndex, dst: NodeIndex, job_id: JobId) -> usize {
        let mut hasher = XxHash64::with_seed(self.seed);
        src.hash(&mut hasher);
        dst.hash(&mut hasher);
        job_id.hash(&mut hasher);
        hasher.finish() as usize
    }

    fn convert_path(&self, topo: &impl SpineTreeTopology, node_path: Vec<NodeIndex>) -> Path {
        let mut link_path = Vec::with_capacity(node_path.len() - 1);
        let graph = topo.topology();
        for window in node_path.windows(2) {
            let from = window[0];
            let to   = window[1];
            let edge_idx = graph
                .find_edge(from, to)
                .expect("ECMP constructed a path over nonexistent edge");
            let link = graph.edge_weight(edge_idx).unwrap();
            link_path.push(link.id);
        }
        link_path
    }
}

impl SpineTreeRouter for SpineEcmpRouter {
    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn route(&mut self, topo: &impl SpineTreeTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        // fast path – already cached
        if let Some(cell) = self.path_cache.get(&flow_id) {
            return cell.clone();
        }

        let mut nodes: Vec<NodeIndex> = Vec::with_capacity(5);

        let src_leaf = topo.get_host_leaf(src);
        let dst_leaf = topo.get_host_leaf(dst);

        nodes.push(src);
        nodes.push(src_leaf);

        if src_leaf != dst_leaf {
            // select spine deterministically but pseudo-randomly
            let spine_idx = self.hash_flow_flow(src, dst, flow_id) % topo.num_spines();
            let spine = topo.get_spine(spine_idx);
            nodes.push(spine);
            nodes.push(dst_leaf);
        }

        nodes.push(dst);

        let link_path = self.convert_path(topo, nodes);
        let cell = PathCell { path: Rc::new(RefCell::new(link_path)) };
        self.path_cache.insert(flow_id, cell.clone());
        cell
    }

    fn complete_flow(&mut self, flow_id: FlowId) {
        self.path_cache.remove(&flow_id);
    }
}

// Provide a stateless prediction for ECMP using the same hashing and seed.
impl SpineRouteOracle for SpineEcmpRouter {
    fn predict_path(&self, topo: &impl SpineTreeTopology, src: NodeIndex, dst: NodeIndex, job_id: JobId) -> Path {
        let mut nodes: Vec<NodeIndex> = Vec::with_capacity(5);

        let src_leaf = topo.get_host_leaf(src);
        let dst_leaf = topo.get_host_leaf(dst);

        nodes.push(src);
        nodes.push(src_leaf);

        if src_leaf != dst_leaf {
            // select spine deterministically but pseudo-randomly
            let spine_idx = self.hash_flow(src, dst, job_id) % topo.num_spines();
            let spine = topo.get_spine(spine_idx);
            nodes.push(spine);
            nodes.push(dst_leaf);// hash on (src, dst, job_id) with the configured seed 
        }

        nodes.push(dst);

        self.convert_path(topo, nodes)
    }
}
