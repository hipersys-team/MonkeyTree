use petgraph::graph::{NodeIndex};
use crate::network::topology::{FatTreeTopology, LinkId};
use crate::network::flow::FlowId;
use crate::network::routing::{FatTreeRouter, PathCell};
use std::hash::{Hash, Hasher};
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use twox_hash::XxHash64;
use crate::simulator::ml_simulator::MLContext;

pub struct EcmpRouter {
    context: Option<MLContext>,
    seed: u64,
    path_cache: HashMap<FlowId, PathCell>,
}

impl EcmpRouter {
    pub fn new(seed: u64) -> Self {
        Self { context: None, seed, path_cache: HashMap::new() }
    }

    fn hash_flow(&self, src: NodeIndex, dst: NodeIndex, flow_id: FlowId, level: usize) -> usize {
        let mut hasher = XxHash64::with_seed(self.seed);

        src.hash(&mut hasher);
        dst.hash(&mut hasher);
        flow_id.hash(&mut hasher);
        level.hash(&mut hasher);

        hasher.finish() as usize
    }

    fn convert_path(&self, topo: &impl FatTreeTopology, path: Vec<NodeIndex>) -> Vec<LinkId> {
        let mut link_path = Vec::new();
        let graph = topo.topology();

        for window in path.windows(2) {
            if let Some(edge) = graph.find_edge(window[0], window[1]) {
                let link = graph.edge_weight(edge).unwrap();
                link_path.push(link.id);
            } else {
                panic!("No edge found between nodes {:?} and {:?}", window[0], window[1]);
            }
        }

        link_path
    }
}

impl FatTreeRouter for EcmpRouter {
    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn route(&mut self, topo: &impl FatTreeTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        if let Some(path_cell) = self.path_cache.get(&flow_id) {
            return path_cell.clone();
        }

        let mut path = Vec::new();
        let src_tor = topo.get_host_tor(src);
        let dst_tor = topo.get_host_tor(dst);
        let src_pod = topo.get_host_pod(src);
        let dst_pod = topo.get_host_pod(dst);

        path.push(src);
        path.push(src_tor);

        if src_tor == dst_tor {
            path.push(dst);
            let link_path = self.convert_path(topo, path);
            let path_cell = PathCell {
                path: Rc::new(RefCell::new(link_path))
            };
            self.path_cache.insert(flow_id, path_cell.clone());
            return path_cell;
        }

        let agg_hash = self.hash_flow(src, dst, flow_id, 0) % topo.degree_agg();
        let agg = topo.get_agg(src_pod, agg_hash);
        path.push(agg);

        if src_pod == dst_pod {
            path.push(dst_tor);
            path.push(dst);
            let link_path = self.convert_path(topo, path);
            let path_cell = PathCell {
                path: Rc::new(RefCell::new(link_path))
            };
            self.path_cache.insert(flow_id, path_cell.clone());
            return path_cell;
        }

        let total_cores = topo.total_core_switches();
        let core_hash = self.hash_flow(src, dst, flow_id, 1) % total_cores;
        let core = topo.get_core(core_hash);
        path.push(core);

        let down_agg_hash = self.hash_flow(src, dst, flow_id, 2) % topo.degree_agg();
        let down_agg = topo.get_agg(dst_pod, down_agg_hash);
        path.push(down_agg);

        let dst_tor = topo.get_host_tor(dst);
        path.push(dst_tor);
        path.push(dst);

        let link_path = self.convert_path(topo, path);

        let path_cell = PathCell {
            path: Rc::new(RefCell::new(link_path))
        };
        self.path_cache.insert(flow_id, path_cell.clone());
        path_cell
    }

    fn complete_flow(&mut self, flow_id: FlowId) {
        self.path_cache.remove(&flow_id);
    }
}