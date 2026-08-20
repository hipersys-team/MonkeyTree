use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use petgraph::graph::NodeIndex;

use crate::network::flow::FlowId;
use crate::network::routing::{Path, PathCell};
use crate::network::topology::{LinkId, Topology};
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{JobId, MLJob};
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::system::SystemModule;
use crate::simulator::{FlowScheduler};

use super::topology::{SpineTree, SpineTreeRouter, SpineTreeTopology};

// -----------------------------------------------------------------------------
// Internal state: derive flows from active jobs (like MonkeyTree)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FlowTemplateSpec {
    job_id: JobId,
    job_flow_idx: usize,
    src_host_index: usize,
    dst_host_index: usize,
}

#[derive(Debug, Default, Clone)]
struct JobInfo {
    flows: Vec<FlowTemplateSpec>,
}

#[derive(Debug, Default)]
struct CruxCore {
    jobs: HashMap<JobId, JobInfo>,
}

impl CruxCore {
    fn record_job(&mut self, job: &MLJob) {
        // Build per-job flow specs from worker templates and placement info in the job
        let mut info = JobInfo { flows: Vec::new() };

        for (&src_worker_id, worker) in job.workers.iter() {
            for ev in &worker.template_events {
                if ev.kind == crate::simulator::ml_worker::WorkerEventKind::FlowSend {
                    let send = ev.flow_send.as_ref().expect("FlowSend missing payload");
                    let src_host_index = worker.host_index;
                    let dst_host_index = job
                        .get_worker_host(send.dst_worker)
                        .expect("dst worker host missing");
                    let flow_idx = *job
                        .send_template_to_flow_idx
                        .get(&(src_worker_id, ev.template_id))
                        .expect("missing send_template_to_flow_idx mapping");
                    info.flows.push(FlowTemplateSpec {
                        job_id: job.id,
                        job_flow_idx: flow_idx,
                        src_host_index,
                        dst_host_index,
                    });
                }
            }
        }

        // Deterministic order
        info.flows
            .sort_by_key(|f| (f.job_id, f.job_flow_idx));

        self.jobs.insert(job.id, info);
    }

    fn remove_job(&mut self, job_id: JobId) { self.jobs.remove(&job_id); }

    fn convert_node_path_to_links<T: SpineTreeTopology>(&self, topo: &T, node_path: Vec<NodeIndex>) -> Path {
        let mut link_path = Vec::with_capacity(node_path.len().saturating_sub(1));
        let graph = topo.topology();
        for window in node_path.windows(2) {
            let edge_idx = graph
                .find_edge(window[0], window[1])
                .expect("Path constructed over nonexistent edge");
            let link = graph.edge_weight(edge_idx).unwrap();
            link_path.push(link.id);
        }
        link_path
    }
}

// -----------------------------------------------------------------------------
// SpineCruxRouter: template-based, system-directed router
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct SpineCruxRouter {
    context: Option<MLContext>,
    /// (job_id, job_flow_idx) → full link path
    template_cache: HashMap<(JobId, usize), Path>,
}

impl SpineCruxRouter {
    pub fn new() -> Self { Self { context: None, template_cache: HashMap::new() } }

    pub fn clear_templates(&mut self) { self.template_cache.clear(); }

    pub fn inject_template(&mut self, job_id: JobId, job_flow_idx: usize, path: Path) {
        self.template_cache.insert((job_id, job_flow_idx), path);
    }
}

impl SpineTreeRouter for SpineCruxRouter {
    fn set_context(&mut self, context: &MLContext) { self.context = Some(context.clone()); }

    fn route(&mut self, topo: &impl SpineTreeTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        // Resolve (job_id, flow_idx) from context and serve cached template
        let ctx = self.context.as_ref().expect("SpineCruxRouter context not set");
        let map_ref = ctx.waiting_flows.borrow();
        let (job_id, job_flow_idx, _iter_idx, _src_w, _dst_w, _send_eid, _recv_eid) = map_ref
            .get(&flow_id)
            .copied()
            .unwrap_or_else(|| panic!("SpineCruxRouter: missing mapping for flow_id {}", flow_id));
        drop(map_ref);
        let key = (job_id, job_flow_idx);
        
        let path = if let Some(cached_path) = self.template_cache.get(&key) {
            cached_path.clone()
        } else {
            // Fallback: compute path on-the-fly using hash-based spine selection
            // This handles migration flows or other flows that weren't pre-computed
            let src_leaf = topo.get_host_leaf(src);
            let dst_leaf = topo.get_host_leaf(dst);
            
            let mut nodes: Vec<NodeIndex> = Vec::with_capacity(5);
            nodes.push(src);
            nodes.push(src_leaf);
            
            if src_leaf != dst_leaf {
                // Need to go through a spine - use hash-based selection
                let hash = (job_id as usize).wrapping_mul(31).wrapping_add(job_flow_idx);
                let spine_idx = hash % topo.num_spines();
                let spine = topo.get_spine(spine_idx);
                nodes.push(spine);
                nodes.push(dst_leaf);
            }
            
            nodes.push(dst);
            
            // Convert node path to link path
            let graph = topo.topology();
            let mut link_path = Vec::with_capacity(nodes.len().saturating_sub(1));
            for window in nodes.windows(2) {
                let edge_idx = graph
                    .find_edge(window[0], window[1])
                    .expect("Path constructed over nonexistent edge");
                let link = graph.edge_weight(edge_idx).unwrap();
                link_path.push(link.id);
            }
            link_path
        };
        
        PathCell { path: Rc::new(RefCell::new(path)) }
    }

    fn complete_flow(&mut self, _flow_id: FlowId) { }
}

// -----------------------------------------------------------------------------
// CruxSystemModule: recomputes routing on job events using job intensity
// -----------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct CruxSystemModule {
    core: CruxCore,
}

impl CruxSystemModule {
    fn job_intensity_us(&self, ctx: &MLContext, link_bps: f64, job_id: JobId) -> f64 {
        let jobs = ctx.active_jobs.borrow();
        let info = match jobs.get(&job_id) { Some(i) => i, None => return 0.0 };
        let mut total_compute_us: u128 = 0;
        let mut total_send_bytes: u128 = 0;
        for desc in &info.worker_descriptions {
            for step in &desc.steps {
                total_compute_us = total_compute_us.saturating_add(step.compute_us as u128);
                if let Some(flow) = &step.flow { total_send_bytes = total_send_bytes.saturating_add(flow.size_bytes as u128); }
            }
        }
        let compute_us = total_compute_us as f64;
        let data_us = if link_bps > 0.0 { (total_send_bytes as f64) * 8_000_000.0 / link_bps } else { 0.0 };
        compute_us + data_us
    }

    fn path_score(link_load: &HashMap<LinkId, usize>, link_path: &[LinkId]) -> usize {
        link_path
            .iter()
            .map(|lid| *link_load.get(lid).unwrap_or(&0))
            .max()
            .unwrap_or(0)
    }

    fn compute_best_path<T: SpineTreeTopology>(topo: &T, core: &CruxCore, src: NodeIndex, dst: NodeIndex, link_load: &HashMap<LinkId, usize>) -> Path {
        let src_leaf = topo.get_host_leaf(src);
        let dst_leaf = topo.get_host_leaf(dst);
        if src_leaf == dst_leaf {
            let nodes = vec![src, src_leaf, dst];
            return core.convert_node_path_to_links(topo, nodes);
        }
        let mut best: Option<(usize, Path)> = None;
        for spine_idx in 0..topo.num_spines() {
            let spine = topo.get_spine(spine_idx);
            let nodes = vec![src, src_leaf, spine, dst_leaf, dst];
            let path = core.convert_node_path_to_links(topo, nodes);
            let score = Self::path_score(link_load, &path);
            if best.as_ref().map_or(true, |(s, _)| score < *s) { best = Some((score, path)); }
        }
        best.map(|(_, p)| p).expect("CruxSystemModule: no path found")
    }

    fn recompute_all<S: JobScheduler, FS: FlowScheduler>(&mut self, ctx: &MLContext, topo: &SpineTree<SpineCruxRouter>) {
        // Compute per-flow templates in descending job-intensity order
        let mut link_load: HashMap<LinkId, usize> = HashMap::new();
        let link_bps = topo.link_bandwidth_bps();

        let mut flows: Vec<(JobId, usize, usize, usize)> = Vec::new(); // (job_id, job_flow_idx, src_host_index, dst_host_index)
        for (&jid, info) in self.core.jobs.iter() {
            for f in &info.flows { flows.push((jid, f.job_flow_idx, f.src_host_index, f.dst_host_index)); }
        }

        // Pre-compute job intensities
        let mut job_intensity: HashMap<JobId, f64> = HashMap::new();
        flows.sort_by(|a, b| {
            let ia = *job_intensity.entry(a.0).or_insert_with(|| self.job_intensity_us(ctx, link_bps, a.0));
            let ib = *job_intensity.entry(b.0).or_insert_with(|| self.job_intensity_us(ctx, link_bps, b.0));
            ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal).then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
        });

        // Install templates on the router
        // Group flows by (src_host, dst_host) so all flows between the same hosts use the same path
        {
            let mut router = topo.router.borrow_mut();
            router.clear_templates();

            // Cache of already-computed paths by (src_host, dst_host)
            let mut path_cache: HashMap<(usize, usize), Path> = HashMap::new();

            for (job_id, job_flow_idx, src_h, dst_h) in flows.into_iter() {
                let path = if let Some(cached) = path_cache.get(&(src_h, dst_h)) {
                    cached.clone()
                } else {
                    let src = topo.get_host_by_index(src_h).expect("invalid src host");
                    let dst = topo.get_host_by_index(dst_h).expect("invalid dst host");
                    let new_path = Self::compute_best_path(topo, &self.core, src, dst, &link_load);
                    for lid in new_path.iter() { *link_load.entry(*lid).or_insert(0) += 1; }
                    path_cache.insert((src_h, dst_h), new_path.clone());
                    new_path
                };
                router.inject_template(job_id, job_flow_idx, path);
            }
        }
    }
}

impl<S, FS> SystemModule<SpineTree<SpineCruxRouter>, S, FS> for CruxSystemModule
where
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_init(&mut self, _ctx: &MLContext, _topo: &SpineTree<SpineCruxRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) {}

    fn on_job_scheduled(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        job: &MLJob,
        _topo: &SpineTree<SpineCruxRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        self.core.record_job(job);
    }

    fn on_job_completed(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        job_id: JobId,
        _job: &MLJob,
        _topo: &SpineTree<SpineCruxRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        self.core.remove_job(job_id);
    }

    fn on_reconfigure(
        &mut self,
        _now_us: u64,
        ctx: &MLContext,
        topo: &SpineTree<SpineCruxRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) -> Option<crate::simulator::system::MigrationPlan> {
        self.recompute_all::<S, FS>(ctx, topo);
        None
    }
}


