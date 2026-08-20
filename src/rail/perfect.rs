use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use petgraph::graph::NodeIndex;

use crate::network::flow::FlowId;
use crate::network::routing::{Path, PathCell};
use crate::network::topology::Topology;
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{JobId, MLJob};
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::ml_worker::WorkerId;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::system::SystemModule;

use crate::monkeytree::perfect_matching::{compute_edge_coloring, collapse_colors};

use super::topology::{RailTree, RailTopology, RailTreeRouter};
use super::routing::build_rail_node_path;

// ---------------------------------------------------------------------------
// RailPerfectRouter
// ---------------------------------------------------------------------------

/// Router that uses bipartite edge coloring to assign cross-pod flows to spines.
///
/// The bipartite graph has vertices = global rail indices (pod * block_size + rail_offset).
/// Edges = cross-pod flows between (src_rail, dst_rail).
#[derive(Debug, Clone, Default)]
pub struct RailPerfectRouter {
    context: Option<MLContext>,
    /// (job_id, job_flow_idx) -> spine index
    flow_to_spine: HashMap<(JobId, usize), usize>,
    path_cache: HashMap<FlowId, PathCell>,
}

impl RailPerfectRouter {
    pub fn new() -> Self {
        Self { context: None, flow_to_spine: HashMap::new(), path_cache: HashMap::new() }
    }

    pub fn clear(&mut self) {
        self.flow_to_spine.clear();
        self.path_cache.clear();
    }

    pub fn inject_spine_assignment(&mut self, job_id: JobId, job_flow_idx: usize, spine_idx: usize) {
        self.flow_to_spine.insert((job_id, job_flow_idx), spine_idx);
    }

    fn convert_node_path_to_links(&self, topo: &impl Topology, nodes: &[NodeIndex]) -> Path {
        let mut link_path = Vec::with_capacity(nodes.len().saturating_sub(1));
        let graph = topo.topology();
        for window in nodes.windows(2) {
            let edge_idx = graph
                .find_edge(window[0], window[1])
                .expect("Path constructed over nonexistent edge");
            link_path.push(graph.edge_weight(edge_idx).unwrap().id);
        }
        link_path
    }
}

impl RailTreeRouter for RailPerfectRouter {
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
            let ctx = self.context.as_ref().expect("RailPerfectRouter context not set");
            let map_ref = ctx.waiting_flows.borrow();
            let (job_id, job_flow_idx, _iter_idx, _src_w, _dst_w, _send_eid, _recv_eid) = map_ref
                .get(&flow_id)
                .copied()
                .unwrap_or_else(|| panic!("RailPerfectRouter: missing mapping for flow_id {}", flow_id));
            drop(map_ref);

            let spine_idx = self.flow_to_spine.get(&(job_id, job_flow_idx)).copied().unwrap_or_else(|| {
                let hash = (job_id as usize).wrapping_mul(31).wrapping_add(job_flow_idx);
                hash % topo.num_spines()
            });
            Some(topo.get_spine(spine_idx))
        } else {
            None
        };

        let nodes = build_rail_node_path(topo, src, dst, spine);
        let link_path = self.convert_node_path_to_links(topo, &nodes);
        let cell = PathCell { path: Rc::new(RefCell::new(link_path)) };
        self.path_cache.insert(flow_id, cell.clone());
        cell
    }

    fn complete_flow(&mut self, flow_id: FlowId) {
        self.path_cache.remove(&flow_id);
    }
}

// ---------------------------------------------------------------------------
// Flow tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FlowTemplateSpec {
    job_id: JobId,
    job_flow_idx: usize,
    src_worker_id: WorkerId,
    dst_worker_id: WorkerId,
    is_ring_flow: bool,
}

#[derive(Debug, Default, Clone)]
struct JobInfo {
    flows: Vec<FlowTemplateSpec>,
}

#[derive(Debug, Default)]
struct PerfectCore {
    jobs: HashMap<JobId, JobInfo>,
}

impl PerfectCore {
    fn record_job(&mut self, job: &MLJob) {
        let mut info = JobInfo { flows: Vec::new() };
        for (&src_worker_id, worker) in job.workers.iter() {
            for ev in &worker.template_events {
                if ev.kind == crate::simulator::ml_worker::WorkerEventKind::FlowSend {
                    let send = ev.flow_send.as_ref().expect("FlowSend missing payload");
                    let flow_idx = *job
                        .send_template_to_flow_idx
                        .get(&(src_worker_id, ev.template_id))
                        .expect("missing send_template_to_flow_idx mapping");
                    info.flows.push(FlowTemplateSpec {
                        job_id: job.id,
                        job_flow_idx: flow_idx,
                        src_worker_id,
                        dst_worker_id: send.dst_worker,
                        is_ring_flow: send.is_ring_flow(),
                    });
                }
            }
        }
        info.flows.sort_by_key(|f| (f.job_id, f.job_flow_idx));
        self.jobs.insert(job.id, info);
    }

    fn remove_job(&mut self, job_id: JobId) { self.jobs.remove(&job_id); }
}

// ---------------------------------------------------------------------------
// RailPerfectRoutingSystem: routing-only system module (no migration)
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct RailPerfectRoutingSystem {
    core: PerfectCore,
}

impl RailPerfectRoutingSystem {
    pub fn new() -> Self { Self { core: PerfectCore::default() } }

    fn recompute_routes(&mut self, ctx: &MLContext, topo: &RailTree<RailPerfectRouter>) {
        let placements = ctx.placements.borrow();
        let block_size = topo.block_size;
        let num_spines = topo.num_spines;
        let total_rails = topo.num_pods * block_size;

        let mut ring_flows: Vec<(usize, usize, (JobId, usize))> = Vec::new();
        let mut other_flows: Vec<(usize, usize, (JobId, usize))> = Vec::new();

        for (&jid, info) in self.core.jobs.iter() {
            let job_placements = match placements.get(&jid) { Some(p) => p, None => continue };
            for f in &info.flows {
                let src_host = match job_placements.get(&f.src_worker_id) { Some(&h) => h, None => continue };
                let dst_host = match job_placements.get(&f.dst_worker_id) { Some(&h) => h, None => continue };
                let src_pod = src_host / (topo.blocks_per_pod * block_size);
                let dst_pod = dst_host / (topo.blocks_per_pod * block_size);

                if src_pod != dst_pod {
                    let src_gpu = src_host % block_size;
                    let dst_gpu = dst_host % block_size;
                    let src_rail_global = src_pod * block_size + src_gpu;
                    let dst_rail_global = dst_pod * block_size + dst_gpu;

                    if f.is_ring_flow {
                        ring_flows.push((src_rail_global, dst_rail_global, (jid, f.job_flow_idx)));
                    } else {
                        other_flows.push((src_host, dst_host, (jid, f.job_flow_idx)));
                    }
                }
            }
        }

        let mut router = topo.router.borrow_mut();
        router.clear();

        if ring_flows.is_empty() && other_flows.is_empty() { return; }

        if !ring_flows.is_empty() {
            let edges: Vec<(usize, usize, usize)> = ring_flows.iter()
                .enumerate()
                .map(|(idx, &(src, dst, _))| (src, dst, idx))
                .collect();

            let coloring = compute_edge_coloring(total_rails, edges);

            println!("[RailPerfectRouting] Ring flows: {} flows, {} colors needed, {} spines available",
                ring_flows.len(), coloring.num_colors, num_spines);

            let final_coloring = if coloring.num_colors > num_spines {
                collapse_colors(&coloring, num_spines)
            } else {
                coloring.edge_to_color.clone()
            };

            for (edge_idx, &(_, _, (job_id, job_flow_idx))) in ring_flows.iter().enumerate() {
                if let Some(&spine_idx) = final_coloring.get(&edge_idx) {
                    router.inject_spine_assignment(job_id, job_flow_idx, spine_idx);
                }
            }
        }

        if !other_flows.is_empty() {
            let mut pair_to_spine: HashMap<(usize, usize), usize> = HashMap::new();
            let mut next_spine = 0;
            for &(src_host, dst_host, _) in other_flows.iter() {
                pair_to_spine.entry((src_host, dst_host)).or_insert_with(|| {
                    let s = next_spine % num_spines;
                    next_spine += 1;
                    s
                });
            }
            for &(src_host, dst_host, (job_id, job_flow_idx)) in other_flows.iter() {
                let spine_idx = pair_to_spine[&(src_host, dst_host)];
                router.inject_spine_assignment(job_id, job_flow_idx, spine_idx);
            }
        }
    }
}

impl<S, FS> SystemModule<RailTree<RailPerfectRouter>, S, FS> for RailPerfectRoutingSystem
where S: JobScheduler, FS: FlowScheduler,
{
    fn on_init(&mut self, _ctx: &MLContext, _topo: &RailTree<RailPerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) {}

    fn on_job_scheduled(&mut self, _now_us: u64, ctx: &MLContext, _job_id: JobId, job: &MLJob, topo: &RailTree<RailPerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) {
        self.core.record_job(job);
        self.recompute_routes(ctx, topo);
    }

    fn on_job_completed(&mut self, _now_us: u64, ctx: &MLContext, job_id: JobId, _job: &MLJob, topo: &RailTree<RailPerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) {
        self.core.remove_job(job_id);
        self.recompute_routes(ctx, topo);
    }

    fn on_reconfigure(&mut self, _now_us: u64, _ctx: &MLContext, _topo: &RailTree<RailPerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) -> Option<crate::simulator::system::MigrationPlan> {
        None
    }
}
