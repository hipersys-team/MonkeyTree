use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use petgraph::graph::NodeIndex;
use twox_hash::XxHash64;

use crate::network::flow::FlowId;
use crate::network::routing::{Path, PathCell};
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::ml_job::JobId;

use super::topology::{RailTopology, RailTreeRouter};

/// ECMP router for rail-optimized topologies.
///
/// Spine selection for cross-pod flows uses hash-based ECMP.
/// Intra-pod flows are routed without touching the spine layer.
#[derive(Debug, Clone)]
pub struct RailEcmpRouter {
    context: Option<MLContext>,
    seed: u64,
    path_cache: HashMap<FlowId, PathCell>,
}

impl RailEcmpRouter {
    pub fn new(seed: u64) -> Self {
        Self { context: None, seed, path_cache: HashMap::new() }
    }

    fn hash_flow(&self, src: NodeIndex, dst: NodeIndex, job_id: JobId) -> usize {
        let mut hasher = XxHash64::with_seed(self.seed);
        src.hash(&mut hasher);
        dst.hash(&mut hasher);
        job_id.hash(&mut hasher);
        hasher.finish() as usize
    }

    fn hash_flow_from_context(&self, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> usize {
        let ctx = self.context.as_ref().expect("RailEcmpRouter context not set");
        let map_ref = ctx.waiting_flows.borrow();
        let (job_id, _job_flow_idx, _iter_idx, _src_w, _dst_w, _send_eid, _recv_eid) = map_ref
            .get(&flow_id)
            .copied()
            .unwrap_or_else(|| panic!("RailEcmpRouter: missing mapping for flow_id {}", flow_id));
        drop(map_ref);
        self.hash_flow(src, dst, job_id)
    }

    fn convert_path(&self, topo: &impl RailTopology, node_path: &[NodeIndex]) -> Path {
        let mut link_path = Vec::with_capacity(node_path.len() - 1);
        let graph = topo.topology();
        for window in node_path.windows(2) {
            let from = window[0];
            let to = window[1];
            let edge_idx = graph
                .find_edge(from, to)
                .unwrap_or_else(|| panic!("ECMP constructed a path over nonexistent edge {:?} -> {:?}", from, to));
            let link = graph.edge_weight(edge_idx).unwrap();
            link_path.push(link.id);
        }
        link_path
    }
}

/// Build the node path for a rail topology given routing decisions.
pub fn build_rail_node_path(
    topo: &impl RailTopology,
    src: NodeIndex,
    dst: NodeIndex,
    spine: Option<NodeIndex>,
) -> Vec<NodeIndex> {
    let src_pod = topo.host_pod(src);
    let dst_pod = topo.host_pod(dst);
    let src_gpu = topo.host_gpu_offset(src);
    let dst_gpu = topo.host_gpu_offset(dst);
    let src_block = topo.host_block_global(src);
    let dst_block = topo.host_block_global(dst);

    let mut nodes: Vec<NodeIndex> = Vec::with_capacity(7);

    if src_block == dst_block {
        // Same host: direct intra-host link
        nodes.push(src);
        nodes.push(dst);
    } else if src_pod == dst_pod && src_gpu == dst_gpu {
        // Same pod, same rail
        let rail = topo.get_rail(src_pod, src_gpu).unwrap();
        nodes.push(src);
        nodes.push(rail);
        nodes.push(dst);
    } else if src_pod == dst_pod {
        // Same pod, different rail: bridge via intra-host to dst's rail
        let bridge = topo.get_host(src_pod, topo.host_block_in_pod(src), dst_gpu).unwrap();
        let rail = topo.get_rail(src_pod, dst_gpu).unwrap();
        nodes.push(src);
        nodes.push(bridge);
        nodes.push(rail);
        nodes.push(dst);
    } else {
        // Cross-pod: src -> src_rail -> spine -> dst_rail -> dst
        let src_rail = topo.get_rail(src_pod, src_gpu).unwrap();
        let dst_rail = topo.get_rail(dst_pod, dst_gpu).unwrap();
        let spine_node = spine.expect("cross-pod route requires a spine");
        nodes.push(src);
        nodes.push(src_rail);
        nodes.push(spine_node);
        nodes.push(dst_rail);
        nodes.push(dst);
    }

    nodes
}

impl RailTreeRouter for RailEcmpRouter {
    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn route(&mut self, topo: &impl RailTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        if let Some(cell) = self.path_cache.get(&flow_id) {
            return cell.clone();
        }

        let src_pod = topo.host_pod(src);
        let dst_pod = topo.host_pod(dst);

        let spine = if src_pod != dst_pod {
            let spine_idx = self.hash_flow_from_context(src, dst, flow_id) % topo.num_spines();
            Some(topo.get_spine(spine_idx))
        } else {
            None
        };

        let nodes = build_rail_node_path(topo, src, dst, spine);
        let link_path = self.convert_path(topo, &nodes);
        let cell = PathCell { path: Rc::new(RefCell::new(link_path)) };
        self.path_cache.insert(flow_id, cell.clone());
        cell
    }

    fn complete_flow(&mut self, flow_id: FlowId) {
        self.path_cache.remove(&flow_id);
    }
}
