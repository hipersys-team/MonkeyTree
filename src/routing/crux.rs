use std::collections::HashMap;
use std::cmp::Ordering;
use std::rc::Rc;
use std::cell::RefCell;
use petgraph::graph::{NodeIndex};
use crate::network::topology::{FatTreeTopology, LinkId};
use crate::network::flow::FlowId;
use crate::network::routing::{FatTreeRouter, PathCell};
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::ml_job::JobId;

pub struct CruxRouter {
    context: Option<MLContext>,
    /// Cached paths: flow_id -> PathCell
    path_cache: HashMap<FlowId, PathCell>,
    /// Metadata: flow_id -> (src, dst)
    flow_endpoints: HashMap<FlowId, (NodeIndex, NodeIndex)>,
    /// Current load (weighted) on each link id
    link_load: HashMap<LinkId, usize>,
}

impl CruxRouter {
    pub fn new() -> Self {
        Self {
            context: None,
            path_cache: HashMap::new(),
            flow_endpoints: HashMap::new(),
            link_load: HashMap::new(),
        }
    }

    /// Converts a list of node indices into the corresponding list of link ids
    /// by walking consecutive pairs in the topology graph.
    fn convert_path(&self, topo: &impl FatTreeTopology, node_path: Vec<NodeIndex>) -> Vec<LinkId> {
        let mut link_path = Vec::<LinkId>::new();
        let graph = topo.topology();

        for window in node_path.windows(2) {
            let from = window[0];
            let to = window[1];

            if let Some(edge_idx) = graph.find_edge(from, to) {
                let link = graph.edge_weight(edge_idx).expect("Edge weight missing");
                link_path.push(link.id);
            } else {
                panic!(
                    "CruxRouter: no edge found between nodes {:?} -> {:?}",
                    from, to
                );
            }
        }

        link_path
    }

    /// Calculates a congestion score for a candidate link path. Currently we use
    /// the maximum load on any link along the path.
    fn path_score(&self, link_path: &[LinkId]) -> usize {
        link_path
            .iter()
            .map(|lid| *self.link_load.get(lid).unwrap_or(&0))
            .max()
            .expect("No links in path")
    }

    /// Resolve the job id for a given flow id from the shared context.
    fn flow_job_id(&self, flow_id: FlowId) -> Option<JobId> {
        let ctx = self.context.as_ref()?;
        let map_ref = ctx.waiting_flows.borrow();
        // (JobId, usize, usize, WorkerId, WorkerId, usize, usize)
        map_ref.get(&flow_id).map(|t| t.0)
    }

    /// Compute GPU intensity for a given job based on worker descriptions in the context.
    /// Intensity(us) = sum_worker_compute_us + (sum_flow_bytes * 8_000_000 / link_bps)
    fn job_intensity_us(&self, link_bps: f64, job_id: JobId) -> f64 {
        let ctx = match &self.context { Some(c) => c, None => return 0.0 };
        let jobs = ctx.active_jobs.borrow();
        let info = match jobs.get(&job_id) { Some(i) => i, None => return 0.0 };

        let mut total_compute_us: u128 = 0;
        let mut total_send_bytes: u128 = 0;
        for desc in &info.worker_descriptions {
            for step in &desc.steps {
                total_compute_us = total_compute_us.saturating_add(step.compute_us as u128);
                if let Some(flow) = &step.flow {
                    total_send_bytes = total_send_bytes.saturating_add(flow.size_bytes as u128);
                }
            }
        }

        let compute_us = total_compute_us as f64;
        let data_us = if link_bps > 0.0 {
            // bytes → bits (x8) → milliseconds (*1000 / bps)
            (total_send_bytes as f64) * 8_000_000.0 / link_bps
        } else { 0.0 };
        compute_us + data_us
    }

    /// Recompute paths for ALL active flows from scratch.
    fn recompute_routes(&mut self, topo: &impl FatTreeTopology) {
        // Clear link load and rebuild path cache.
        self.link_load.clear();
        // We'll reuse existing PathCell objects when possible in new map.
        let mut new_cache: HashMap<FlowId, PathCell> = HashMap::new();

        // Sort flows by descending GPU intensity of their job; tie-break by FlowId for determinism.
        let mut flow_ids: Vec<FlowId> = self.flow_endpoints.keys().copied().collect();
        let link_bps = topo.link_bandwidth_bps();
        // Cache per-job intensity to avoid recomputation
        let mut job_intensity: HashMap<JobId, f64> = HashMap::new();
        flow_ids.sort_by(|a, b| {
            let ja = self.flow_job_id(*a);
            let jb = self.flow_job_id(*b);
            let ia = ja.map(|jid| *job_intensity.entry(jid).or_insert_with(|| self.job_intensity_us(link_bps, jid))).unwrap_or(0.0);
            let ib = jb.map(|jid| *job_intensity.entry(jid).or_insert_with(|| self.job_intensity_us(link_bps, jid))).unwrap_or(0.0);
            ib.partial_cmp(&ia).unwrap_or(Ordering::Equal).then_with(|| a.cmp(b))
        });


        for fid in flow_ids {
            let (src, dst) = self.flow_endpoints[&fid];

            // Compute best path given current link load.
            let link_path = self.compute_best_path(topo, src, dst);

            // Update or create PathCell.
            let path_cell = if let Some(existing) = self.path_cache.get(&fid) {
                // Update in-place.
                *existing.path.borrow_mut() = link_path.clone();
                existing.clone()
            } else {
                let cell = PathCell {
                    path: Rc::new(RefCell::new(link_path.clone())),
                };
                cell
            };

            new_cache.insert(fid, path_cell);

            // Increment load on each link.
            for lid in &link_path {
                *self.link_load.entry(*lid).or_insert(0) += 1;
            }
        }

        self.path_cache = new_cache;
    }

    /// Compute the minimum-score link path between src and dst under current link_load.
    fn compute_best_path(&self, topo: &impl FatTreeTopology, src: NodeIndex, dst: NodeIndex) -> Vec<LinkId> {
        let src_tor = topo.get_host_tor(src);
        let dst_tor = topo.get_host_tor(dst);
        let src_pod = topo.get_host_pod(src);
        let dst_pod = topo.get_host_pod(dst);

        let mut best_link_path: Option<Vec<LinkId>> = None;

        if src_tor == dst_tor {
            let node_path = vec![src, src_tor, dst];
            let link_path = self.convert_path(topo, node_path);
            best_link_path = Some(link_path);
        } else if src_pod == dst_pod {
            let mut best_score: Option<usize> = None;
            for agg_idx in 0..topo.degree_agg() {
                let agg = topo.get_agg(src_pod, agg_idx);
                let node_path = vec![src, src_tor, agg, dst_tor, dst];
                let link_path = self.convert_path(topo, node_path);
                let score = self.path_score(&link_path);
                if best_score.map_or(true, |s| score < s) {
                    best_score = Some(score);
                    best_link_path = Some(link_path);
                }
            }
        } else {
            let mut best_score: Option<usize> = None;
            let deg_agg = topo.degree_agg();
            let deg_core = topo.degree_core();

            for agg_idx in 0..deg_agg {
                let agg_src = topo.get_agg(src_pod, agg_idx);
                for core_offset in 0..deg_core {
                    let core_num = agg_idx * deg_core + core_offset;
                    let core = topo.get_core(core_num);

                    let down_agg_idx = core_num / deg_agg;
                    let agg_dst = topo.get_agg(dst_pod, down_agg_idx);

                    let node_path = vec![
                        src,
                        src_tor,
                        agg_src,
                        core,
                        agg_dst,
                        dst_tor,
                        dst,
                    ];
                    let link_path = self.convert_path(topo, node_path);
                    let score = self.path_score(&link_path);
                    if best_score.map_or(true, |s| score < s) {
                        best_score = Some(score);
                        best_link_path = Some(link_path);
                    }
                }
            }
        }

        best_link_path.expect("No path found in CruxRouter::compute_best_path")
    }
 }

impl FatTreeRouter for CruxRouter {
    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn complete_flow(&mut self, flow_id: FlowId) {
        self.path_cache.remove(&flow_id);
        self.flow_endpoints.remove(&flow_id);
        // Note: We don't recompute routes here because we don't have a reference
        // to the topology. The next call to `route` will trigger a full
        // recomputation that takes the removed flow into account.
    }

    fn route(&mut self, topo: &impl FatTreeTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        // Update flow metadata (new or existing)
        self.flow_endpoints.insert(flow_id, (src, dst));

        // Recompute all routes (global re-routing)
        self.recompute_routes(topo);

        // Return the PathCell for this flow (guaranteed to exist after recompute)
        self.path_cache
            .get(&flow_id)
            .expect("PathCell missing after recompute")
            .clone()
    }
}
