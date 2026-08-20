//! Fat-Tree Perfect Routing via Two-Phase Bipartite Edge Coloring
//!
//! Phase 1 (ToR ↔ Agg): Per-pod edge coloring on (src_tor_local, dst_tor_local)
//!   assigns which aggregation switch each cross-ToR flow uses.
//!
//! Phase 2 (Agg ↔ Core): Edge coloring on virtual switches for cross-pod flows.
//!   All agg switches in a pod are collapsed into one virtual switch.
//!   Virtual link rank r = all physical links from any agg to core r.
//!   The coloring assigns a core rank to each cross-pod flow.
//!
//! Translation to physical paths:
//!   - agg_src = Phase 1 assignment (source pod coloring)
//!   - core = core[rank from Phase 2]
//!   - agg_dst = round-robin among agg switches in the destination pod

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use petgraph::graph::NodeIndex;

use crate::network::flow::FlowId;
use crate::network::routing::{Path, PathCell, FatTreeRouter};
use crate::network::topology::{FatTreeTopology, Topology};

use super::convert_node_path_to_links;
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::ml_worker::WorkerId;
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::system::SystemModule;
use crate::monkeytree::perfect_matching::{compute_edge_coloring, collapse_colors};

// ---------------------------------------------------------------------------
// FatTreePerfectRouter: hash-based fallback for flows without system-injected paths
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct FatTreePerfectRouter {
    context: Option<MLContext>,
    /// (job_id, job_flow_idx) → pre-computed link path
    injected: HashMap<(JobId, usize), Path>,
    path_cache: HashMap<FlowId, PathCell>,
}

impl FatTreePerfectRouter {
    pub fn new() -> Self { Self::default() }

    pub fn clear(&mut self) {
        self.injected.clear();
        self.path_cache.clear();
    }

    pub fn inject_path(&mut self, job_id: JobId, job_flow_idx: usize, path: Path) {
        self.injected.insert((job_id, job_flow_idx), path);
    }
}

impl FatTreeRouter for FatTreePerfectRouter {
    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn route(&mut self, topo: &impl FatTreeTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        if let Some(cell) = self.path_cache.get(&flow_id) {
            return cell.clone();
        }

        let ctx = self.context.as_ref().expect("FatTreePerfectRouter: context not set");
        let map_ref = ctx.waiting_flows.borrow();
        let (job_id, job_flow_idx, _iter, _sw, _dw, _se, _re) = map_ref
            .get(&flow_id)
            .copied()
            .unwrap_or_else(|| panic!("FatTreePerfectRouter: no mapping for flow_id {}", flow_id));
        drop(map_ref);

        let path = if let Some(p) = self.injected.get(&(job_id, job_flow_idx)) {
            p.clone()
        } else {
            compute_hash_path(topo, src, dst, flow_id)
        };

        let cell = PathCell { path: Rc::new(RefCell::new(path)) };
        self.path_cache.insert(flow_id, cell.clone());
        cell
    }

    fn complete_flow(&mut self, flow_id: FlowId) {
        self.path_cache.remove(&flow_id);
    }
}

fn compute_hash_path(topo: &impl FatTreeTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> Path {
    let src_tor = topo.get_host_tor(src);
    let dst_tor = topo.get_host_tor(dst);

    if src_tor == dst_tor {
        return convert_node_path_to_links(topo, &[src, dst_tor, dst]);
    }

    let src_pod = topo.get_host_pod(src);
    let dst_pod = topo.get_host_pod(dst);
    let hash = flow_id.wrapping_mul(2654435761);

    if src_pod == dst_pod {
        let agg_idx = hash % topo.degree_agg();
        let agg = topo.get_agg(src_pod, agg_idx);
        convert_node_path_to_links(topo, &[src, src_tor, agg, dst_tor, dst])
    } else {
        let agg_src_idx = hash % topo.degree_agg();
        let total_cores = topo.total_core_switches();
        let core_idx = (hash / topo.degree_agg()) % total_cores.max(1);
        let agg_dst_idx = (hash / (topo.degree_agg() * total_cores.max(1))) % topo.degree_agg();
        let agg_src = topo.get_agg(src_pod, agg_src_idx);
        let core = topo.get_core(core_idx);
        let agg_dst = topo.get_agg(dst_pod, agg_dst_idx);
        convert_node_path_to_links(topo, &[src, src_tor, agg_src, core, agg_dst, dst_tor, dst])
    }
}

// ---------------------------------------------------------------------------
// Flow classification helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FlowTemplateSpec {
    job_id: JobId,
    job_flow_idx: usize,
    src_worker_id: WorkerId,
    dst_worker_id: WorkerId,
    is_ring_flow: bool,
}

#[derive(Debug, Default)]
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
        for (&src_wid, worker) in job.workers.iter() {
            for ev in &worker.template_events {
                if ev.kind == crate::simulator::ml_worker::WorkerEventKind::FlowSend {
                    let send = ev.flow_send.as_ref().expect("FlowSend missing payload");
                    let flow_idx = *job
                        .send_template_to_flow_idx
                        .get(&(src_wid, ev.template_id))
                        .expect("missing send_template_to_flow_idx mapping");
                    info.flows.push(FlowTemplateSpec {
                        job_id: job.id,
                        job_flow_idx: flow_idx,
                        src_worker_id: src_wid,
                        dst_worker_id: send.dst_worker,
                        is_ring_flow: send.is_ring_flow(),
                    });
                }
            }
        }
        info.flows.sort_by_key(|f| (f.job_id, f.job_flow_idx));
        self.jobs.insert(job.id, info);
    }

    fn remove_job(&mut self, job_id: JobId) {
        self.jobs.remove(&job_id);
    }
}

// ---------------------------------------------------------------------------
// Classified flow types used during route computation
// ---------------------------------------------------------------------------

struct IntraPodFlow {
    src_tor_local: usize,
    dst_tor_local: usize,
    pod: usize,
    src_host: usize,
    dst_host: usize,
    job_id: JobId,
    job_flow_idx: usize,
}

struct CrossPodFlow {
    src_tor_local: usize,
    dst_tor_local: usize,
    src_pod: usize,
    dst_pod: usize,
    src_host: usize,
    dst_host: usize,
    job_id: JobId,
    job_flow_idx: usize,
}

// ---------------------------------------------------------------------------
// FatTreePerfectSystem: system module implementing two-phase edge coloring
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct FatTreePerfectSystem {
    core: PerfectCore,
}

impl FatTreePerfectSystem {
    pub fn new() -> Self { Self::default() }

    pub fn record_job(&mut self, job: &MLJob) { self.core.record_job(job); }

    pub fn remove_job(&mut self, job_id: JobId) { self.core.remove_job(job_id); }

    pub fn recompute_routes_pub(&mut self, ctx: &MLContext, topo: &crate::network::topology::FatTree<FatTreePerfectRouter>) {
        self.recompute_routes(ctx, topo);
    }

    fn recompute_routes(&mut self, ctx: &MLContext, topo: &crate::network::topology::FatTree<FatTreePerfectRouter>) {
        let placements = ctx.placements.borrow();
        let hosts_per_tor = topo.hosts_per_tor;
        let tors_per_pod = topo.degree_tor;
        let num_pods = topo.num_pods;
        let num_agg = topo.degree_agg;
        let total_cores = topo.degree_core;

        // Classify all cross-ToR flows
        let mut intra_pod_ring: Vec<IntraPodFlow> = Vec::new();
        let mut intra_pod_other: Vec<IntraPodFlow> = Vec::new();
        let mut cross_pod_ring: Vec<CrossPodFlow> = Vec::new();
        let mut cross_pod_other: Vec<CrossPodFlow> = Vec::new();

        for (&jid, info) in self.core.jobs.iter() {
            let job_pl = match placements.get(&jid) {
                Some(p) => p,
                None => continue,
            };
            for f in &info.flows {
                let src_host = match job_pl.get(&f.src_worker_id) {
                    Some(&h) => h,
                    None => continue,
                };
                let dst_host = match job_pl.get(&f.dst_worker_id) {
                    Some(&h) => h,
                    None => continue,
                };
                let src_pod = src_host / (tors_per_pod * hosts_per_tor);
                let dst_pod = dst_host / (tors_per_pod * hosts_per_tor);
                let src_tor_global = src_host / hosts_per_tor;
                let dst_tor_global = dst_host / hosts_per_tor;
                if src_tor_global == dst_tor_global {
                    continue; // intra-ToR, handled by router fallback
                }
                let src_tor_local = src_tor_global % tors_per_pod;
                let dst_tor_local = dst_tor_global % tors_per_pod;

                if src_pod == dst_pod {
                    let flow = IntraPodFlow {
                        src_tor_local, dst_tor_local, pod: src_pod,
                        src_host, dst_host, job_id: jid, job_flow_idx: f.job_flow_idx,
                    };
                    if f.is_ring_flow { intra_pod_ring.push(flow); }
                    else { intra_pod_other.push(flow); }
                } else {
                    let flow = CrossPodFlow {
                        src_tor_local, dst_tor_local, src_pod, dst_pod,
                        src_host, dst_host, job_id: jid, job_flow_idx: f.job_flow_idx,
                    };
                    if f.is_ring_flow { cross_pod_ring.push(flow); }
                    else { cross_pod_other.push(flow); }
                }
            }
        }

        let mut router = topo.router.borrow_mut();
        router.clear();

        // =====================================================================
        // Phase 1: ToR ↔ Agg edge coloring (per pod) with virtual ToRs
        // =====================================================================
        // The bipartite graph has tors_per_pod + 2 nodes per side:
        //   0..tors_per_pod-1  = real ToRs
        //   tors_per_pod       = "origin" virtual ToR (outbound cross-pod dst)
        //   tors_per_pod + 1   = "target" virtual ToR (inbound cross-pod src)
        //
        // Intra-pod ring:  (src_tor_local, dst_tor_local)
        // Outbound cross-pod ring (src in this pod): (src_tor_local, origin)
        // Inbound cross-pod ring (dst in this pod):  (target, dst_tor_local)

        let origin_idx = tors_per_pod;
        let target_idx = tors_per_pod + 1;
        let coloring_n = tors_per_pod + 2;

        let mut intra_ring_agg: HashMap<usize, usize> = HashMap::new();
        let mut cross_ring_agg_src: HashMap<usize, usize> = HashMap::new();
        let mut cross_ring_agg_dst: HashMap<usize, usize> = HashMap::new();

        for pod in 0..num_pods {
            let intra_indices: Vec<usize> = intra_pod_ring.iter().enumerate()
                .filter(|(_, f)| f.pod == pod)
                .map(|(i, _)| i)
                .collect();
            let outbound_indices: Vec<usize> = cross_pod_ring.iter().enumerate()
                .filter(|(_, f)| f.src_pod == pod)
                .map(|(i, _)| i)
                .collect();
            let inbound_indices: Vec<usize> = cross_pod_ring.iter().enumerate()
                .filter(|(_, f)| f.dst_pod == pod)
                .map(|(i, _)| i)
                .collect();

            if intra_indices.is_empty() && outbound_indices.is_empty() && inbound_indices.is_empty() {
                continue;
            }

            let mut edges: Vec<(usize, usize, usize)> = Vec::new();
            // (edge_type, global_index): 0=intra, 1=outbound cross, 2=inbound cross
            let mut edge_map: Vec<(u8, usize)> = Vec::new();

            for &gi in &intra_indices {
                let f = &intra_pod_ring[gi];
                let local_id = edges.len();
                edges.push((f.src_tor_local, f.dst_tor_local, local_id));
                edge_map.push((0, gi));
            }
            for &gi in &outbound_indices {
                let f = &cross_pod_ring[gi];
                let local_id = edges.len();
                edges.push((f.src_tor_local, origin_idx, local_id));
                edge_map.push((1, gi));
            }
            for &gi in &inbound_indices {
                let f = &cross_pod_ring[gi];
                let local_id = edges.len();
                edges.push((target_idx, f.dst_tor_local, local_id));
                edge_map.push((2, gi));
            }

            let coloring = compute_edge_coloring(coloring_n, edges);
            let final_coloring = if coloring.num_colors > num_agg {
                collapse_colors(&coloring, num_agg)
            } else {
                coloring.edge_to_color.clone()
            };

            for (local_id, &(etype, gi)) in edge_map.iter().enumerate() {
                if let Some(&color) = final_coloring.get(&local_id) {
                    match etype {
                        0 => { intra_ring_agg.insert(gi, color); }
                        1 => { cross_ring_agg_src.insert(gi, color); }
                        2 => { cross_ring_agg_dst.insert(gi, color); }
                        _ => {}
                    }
                }
            }
        }

        // =====================================================================
        // Phase 2: Agg ↔ Core edge coloring (cross-pod ring flows only)
        //
        // Virtual switch model: all aggs in a pod = one virtual switch.
        // Virtual link rank r = all physical links from any agg to core r.
        // Edge coloring on (src_tor_local, dst_tor_local) → core rank.
        // =====================================================================

        let mut cross_ring_core_rank: HashMap<usize, usize> = HashMap::new();

        if !cross_pod_ring.is_empty() {
            let edges: Vec<(usize, usize, usize)> = cross_pod_ring.iter()
                .enumerate()
                .map(|(i, f)| (f.src_tor_local, f.dst_tor_local, i))
                .collect();

            let coloring = compute_edge_coloring(tors_per_pod, edges);
            let final_coloring = if coloring.num_colors > total_cores {
                collapse_colors(&coloring, total_cores)
            } else {
                coloring.edge_to_color.clone()
            };

            for (i, _) in cross_pod_ring.iter().enumerate() {
                if let Some(&rank) = final_coloring.get(&i) {
                    cross_ring_core_rank.insert(i, rank);
                }
            }
        }

        // =====================================================================
        // Inject paths for intra-pod ring flows
        // =====================================================================

        for (i, f) in intra_pod_ring.iter().enumerate() {
            let agg_idx = intra_ring_agg.get(&i).copied().unwrap_or(0);
            let src_node = topo.get_host_by_index(f.src_host).unwrap();
            let dst_node = topo.get_host_by_index(f.dst_host).unwrap();
            let src_tor = topo.get_host_tor(src_node);
            let dst_tor = topo.get_host_tor(dst_node);
            let agg = topo.get_agg(f.pod, agg_idx);
            let path = convert_node_path_to_links(topo, &[src_node, src_tor, agg, dst_tor, dst_node]);
            router.inject_path(f.job_id, f.job_flow_idx, path);
        }

        // =====================================================================
        // Inject paths for cross-pod ring flows
        //   agg_src = Phase 1 outbound  |  core = Phase 2 rank
        //   agg_dst = Phase 1 inbound (from dst pod's coloring)
        // =====================================================================

        for (i, f) in cross_pod_ring.iter().enumerate() {
            let agg_src_idx = cross_ring_agg_src.get(&i).copied().unwrap_or(0);
            let core_idx = cross_ring_core_rank.get(&i).copied().unwrap_or(0);
            let agg_dst_idx = cross_ring_agg_dst.get(&i).copied().unwrap_or(0);

            let src_node = topo.get_host_by_index(f.src_host).unwrap();
            let dst_node = topo.get_host_by_index(f.dst_host).unwrap();
            let src_tor = topo.get_host_tor(src_node);
            let dst_tor = topo.get_host_tor(dst_node);
            let agg_src = topo.get_agg(f.src_pod, agg_src_idx);
            let core = topo.get_core(core_idx);
            let agg_dst = topo.get_agg(f.dst_pod, agg_dst_idx);

            let path = convert_node_path_to_links(
                topo, &[src_node, src_tor, agg_src, core, agg_dst, dst_tor, dst_node]);
            router.inject_path(f.job_id, f.job_flow_idx, path);
        }

        // =====================================================================
        // Inject paths for intra-pod non-ring flows (round-robin agg)
        // =====================================================================

        {
            let mut pair_to_path: HashMap<(usize, usize), Path> = HashMap::new();
            let mut next_agg = 0usize;

            for f in &intra_pod_other {
                let path = if let Some(p) = pair_to_path.get(&(f.src_host, f.dst_host)) {
                    p.clone()
                } else {
                    let src_node = topo.get_host_by_index(f.src_host).unwrap();
                    let dst_node = topo.get_host_by_index(f.dst_host).unwrap();
                    let src_tor = topo.get_host_tor(src_node);
                    let dst_tor = topo.get_host_tor(dst_node);
                    let agg_idx = next_agg % num_agg;
                    next_agg += 1;
                    let agg = topo.get_agg(f.pod, agg_idx);
                    let p = convert_node_path_to_links(topo, &[src_node, src_tor, agg, dst_tor, dst_node]);
                    pair_to_path.insert((f.src_host, f.dst_host), p.clone());
                    p
                };
                router.inject_path(f.job_id, f.job_flow_idx, path);
            }
        }

        // =====================================================================
        // Inject paths for cross-pod non-ring flows (round-robin everything)
        // =====================================================================

        {
            let mut pair_to_path: HashMap<(usize, usize), Path> = HashMap::new();
            let mut next_agg_src = 0usize;
            let mut next_core = 0usize;
            let mut next_agg_dst = 0usize;

            for f in &cross_pod_other {
                let path = if let Some(p) = pair_to_path.get(&(f.src_host, f.dst_host)) {
                    p.clone()
                } else {
                    let src_node = topo.get_host_by_index(f.src_host).unwrap();
                    let dst_node = topo.get_host_by_index(f.dst_host).unwrap();
                    let src_tor = topo.get_host_tor(src_node);
                    let dst_tor = topo.get_host_tor(dst_node);
                    let agg_src_idx = next_agg_src % num_agg;
                    next_agg_src += 1;
                    let core_idx = next_core % total_cores.max(1);
                    next_core += 1;
                    let agg_dst_idx = next_agg_dst % num_agg;
                    next_agg_dst += 1;
                    let agg_src = topo.get_agg(f.src_pod, agg_src_idx);
                    let core = topo.get_core(core_idx);
                    let agg_dst = topo.get_agg(f.dst_pod, agg_dst_idx);
                    let p = convert_node_path_to_links(
                        topo, &[src_node, src_tor, agg_src, core, agg_dst, dst_tor, dst_node]);
                    pair_to_path.insert((f.src_host, f.dst_host), p.clone());
                    p
                };
                router.inject_path(f.job_id, f.job_flow_idx, path);
            }
        }

        let n_total = intra_pod_ring.len() + intra_pod_other.len()
            + cross_pod_ring.len() + cross_pod_other.len();
        if n_total > 0 {
            println!(
                "[FatTreePerfect] Routed {} flows: {} intra-ring, {} intra-other, {} cross-ring, {} cross-other",
                n_total, intra_pod_ring.len(), intra_pod_other.len(),
                cross_pod_ring.len(), cross_pod_other.len()
            );
        }
    }
}

impl<S, FS> SystemModule<crate::network::topology::FatTree<FatTreePerfectRouter>, S, FS>
    for FatTreePerfectSystem
where
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_job_scheduled(
        &mut self, _now_us: u64, ctx: &MLContext, _job_id: JobId, job: &MLJob,
        topo: &crate::network::topology::FatTree<FatTreePerfectRouter>,
        _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {
        self.core.record_job(job);
        self.recompute_routes(ctx, topo);
    }

    fn on_job_completed(
        &mut self, _now_us: u64, ctx: &MLContext, job_id: JobId, _job: &MLJob,
        topo: &crate::network::topology::FatTree<FatTreePerfectRouter>,
        _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {
        self.core.remove_job(job_id);
        self.recompute_routes(ctx, topo);
    }

    fn on_reconfigure(
        &mut self, _now_us: u64, ctx: &MLContext,
        topo: &crate::network::topology::FatTree<FatTreePerfectRouter>,
        _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) -> Option<crate::simulator::system::MigrationPlan> {
        self.recompute_routes(ctx, topo);
        None
    }

    fn on_migration_end(
        &mut self, _now_us: u64, ctx: &MLContext, job_id: JobId, job: &MLJob,
        topo: &crate::network::topology::FatTree<FatTreePerfectRouter>,
        _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {
        self.core.remove_job(job_id);
        self.core.record_job(job);
        self.recompute_routes(ctx, topo);
    }
}
