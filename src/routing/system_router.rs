use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

use petgraph::graph::NodeIndex;

use crate::network::flow::FlowId;
use crate::network::routing::{FatTreeRouter, PathCell, Path};
use crate::network::topology::FatTreeTopology;
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::ml_job::JobId;

/// A router that delegates all path decisions to an external system module.
///
/// This router itself performs no computation. It maintains an internal cache
/// of paths indexed by `FlowId`. When the simulator asks for a route, this
/// router looks up the flow's path in the cache and returns it. If the path is
/// missing, it panics with an error. The system module is expected to populate
/// or update paths using `inject_or_update_route`.
#[derive(Debug, Clone, Default)]
pub struct SystemRouter {
    context: Option<MLContext>,
    /// flow-id → shared, mutable path
    path_cache: HashMap<FlowId, PathCell>,
    /// (job_id, job_flow_idx, iter_idx) → path template
    template_cache: HashMap<(JobId, usize, usize), Path>,
}

impl SystemRouter {
    pub fn new() -> Self {
        Self { context: None, path_cache: HashMap::new(), template_cache: HashMap::new() }
    }

    /// Inserts or updates a path template keyed by (job_id, job_flow_idx, iter_idx).
    ///
    /// The system module can populate this ahead of time, before flows exist.
    pub fn inject_route(&mut self, job_id: JobId, job_flow_idx: usize, iter_idx: usize, path: Path) {
        self.template_cache.insert((job_id, job_flow_idx, iter_idx), path);
    }

    /// Computes or fetches a path for a flow based on ML context mapping
    /// from flow_id -> (job_id, flow_idx, iter).
    fn ensure_path_for(&mut self, _topo: &impl FatTreeTopology, flow_id: FlowId, src: NodeIndex, dst: NodeIndex) -> PathCell {
        if let Some(cell) = self.path_cache.get(&flow_id) { return cell.clone(); }
        let ctx = self.context.as_ref().expect("SystemRouter context not set");
        let map_ref = ctx.waiting_flows.borrow();
        let (job_id, job_flow_idx, iter_idx, _src_w, _dst_w, _send_eid, _recv_eid) = map_ref.get(&flow_id)
            .copied()
            .unwrap_or_else(|| panic!("SystemRouter: missing mapping for flow_id {}", flow_id));
        drop(map_ref);
        let key = (job_id, job_flow_idx, iter_idx);
        if let Some(path) = self.template_cache.get(&key) {
            let cell = PathCell { path: Rc::new(RefCell::new(path.clone())) };
            self.path_cache.insert(flow_id, cell.clone());
            return cell;
        }
        panic!(
            "SystemRouter: no injected template for (job_id={}, flow_idx={}, iter={}); from {} to {} flow_id={}",
            job_id, job_flow_idx, iter_idx, src.index(), dst.index(), flow_id
        );
    }
}

impl FatTreeRouter for SystemRouter {
    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn complete_flow(&mut self, flow_id: FlowId) {
        self.path_cache.remove(&flow_id);
    }

    fn route(&mut self, topo: &impl FatTreeTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        self.ensure_path_for(topo, flow_id, src, dst)
    }
}

