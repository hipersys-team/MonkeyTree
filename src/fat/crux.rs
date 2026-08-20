//! Fat-Tree Crux: system-directed routing via greedy path selection
//!
//! Mirrors the spine Crux design: flows are sorted by descending job intensity,
//! then each unique (src_host, dst_host) pair is assigned the path with the
//! lowest bottleneck link load among all candidate paths.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use petgraph::graph::NodeIndex;

use crate::network::flow::FlowId;
use crate::network::routing::{Path, PathCell, FatTreeRouter};
use crate::network::topology::{FatTree, FatTreeTopology, LinkId, Topology};

use super::convert_node_path_to_links;
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{JobId, MLJob};
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::system::SystemModule;

// ---------------------------------------------------------------------------
// Flow tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FlowTemplateSpec {
    job_id: JobId,
    job_flow_idx: usize,
    src_host_index: usize,
    dst_host_index: usize,
}

#[derive(Debug, Default)]
struct CruxCore {
    jobs: HashMap<JobId, Vec<FlowTemplateSpec>>,
}

impl CruxCore {
    fn record_job(&mut self, job: &MLJob) {
        let mut flows = Vec::new();
        for (&src_wid, worker) in job.workers.iter() {
            for ev in &worker.template_events {
                if ev.kind == crate::simulator::ml_worker::WorkerEventKind::FlowSend {
                    let send = ev.flow_send.as_ref().expect("FlowSend missing payload");
                    let flow_idx = *job
                        .send_template_to_flow_idx
                        .get(&(src_wid, ev.template_id))
                        .expect("missing send_template_to_flow_idx mapping");
                    flows.push(FlowTemplateSpec {
                        job_id: job.id,
                        job_flow_idx: flow_idx,
                        src_host_index: worker.host_index,
                        dst_host_index: job.get_worker_host(send.dst_worker)
                            .expect("dst worker host missing"),
                    });
                }
            }
        }
        flows.sort_by_key(|f| (f.job_id, f.job_flow_idx));
        self.jobs.insert(job.id, flows);
    }

    fn remove_job(&mut self, job_id: JobId) { self.jobs.remove(&job_id); }
}

// ---------------------------------------------------------------------------
// FatTreeCruxRouter: template-based router for fat trees
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct FatTreeCruxRouter {
    context: Option<MLContext>,
    template_cache: HashMap<(JobId, usize), Path>,
}

impl FatTreeCruxRouter {
    pub fn new() -> Self { Self::default() }

    pub fn clear_templates(&mut self) { self.template_cache.clear(); }

    pub fn inject_template(&mut self, job_id: JobId, job_flow_idx: usize, path: Path) {
        self.template_cache.insert((job_id, job_flow_idx), path);
    }
}

impl FatTreeRouter for FatTreeCruxRouter {
    fn set_context(&mut self, context: &MLContext) { self.context = Some(context.clone()); }

    fn route(&mut self, topo: &impl FatTreeTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        let ctx = self.context.as_ref().expect("FatTreeCruxRouter: context not set");
        let map_ref = ctx.waiting_flows.borrow();
        let (job_id, job_flow_idx, _iter, _sw, _dw, _se, _re) = map_ref
            .get(&flow_id)
            .copied()
            .unwrap_or_else(|| panic!("FatTreeCruxRouter: no mapping for flow_id {}", flow_id));
        drop(map_ref);

        let path = if let Some(cached) = self.template_cache.get(&(job_id, job_flow_idx)) {
            cached.clone()
        } else {
            compute_fallback_path(topo, src, dst, job_id, job_flow_idx)
        };

        PathCell { path: Rc::new(RefCell::new(path)) }
    }

    fn complete_flow(&mut self, _flow_id: FlowId) {}
}

fn compute_fallback_path(
    topo: &impl FatTreeTopology, src: NodeIndex, dst: NodeIndex,
    job_id: JobId, job_flow_idx: usize,
) -> Path {
    let src_tor = topo.get_host_tor(src);
    let dst_tor = topo.get_host_tor(dst);
    if src_tor == dst_tor {
        return convert_node_path_to_links(topo, &[src, dst_tor, dst]);
    }
    let src_pod = topo.get_host_pod(src);
    let dst_pod = topo.get_host_pod(dst);
    let hash = (job_id as usize).wrapping_mul(31).wrapping_add(job_flow_idx);

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
// FatTreeCruxSystem: greedy best-path system module
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct FatTreeCruxSystem {
    core: CruxCore,
}

impl FatTreeCruxSystem {
    pub fn new() -> Self { Self::default() }

    fn job_intensity_us(&self, ctx: &MLContext, link_bps: f64, job_id: JobId) -> f64 {
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
            (total_send_bytes as f64) * 8_000_000.0 / link_bps
        } else {
            0.0
        };
        compute_us + data_us
    }

    fn path_score(link_load: &HashMap<LinkId, usize>, path: &[LinkId]) -> usize {
        path.iter().map(|lid| *link_load.get(lid).unwrap_or(&0)).max().unwrap_or(0)
    }

    fn compute_best_path(
        topo: &FatTree<FatTreeCruxRouter>,
        src: NodeIndex, dst: NodeIndex,
        link_load: &HashMap<LinkId, usize>,
    ) -> Path {
        let src_tor = topo.get_host_tor(src);
        let dst_tor = topo.get_host_tor(dst);

        if src_tor == dst_tor {
            return convert_node_path_to_links(topo, &[src, src_tor, dst]);
        }

        let src_pod = topo.get_host_pod(src);
        let dst_pod = topo.get_host_pod(dst);

        if src_pod == dst_pod {
            let mut best: Option<(usize, Path)> = None;
            for agg_idx in 0..topo.degree_agg {
                let agg = topo.get_agg(src_pod, agg_idx);
                let nodes = [src, src_tor, agg, dst_tor, dst];
                let path = convert_node_path_to_links(topo, &nodes);
                let score = Self::path_score(link_load, &path);
                if best.as_ref().map_or(true, |(s, _)| score < *s) {
                    best = Some((score, path));
                }
            }
            return best.map(|(_, p)| p).expect("no intra-pod path found");
        }

        // Cross-pod: enumerate (agg_src, core, agg_dst)
        let mut best: Option<(usize, Path)> = None;
        let deg_agg = topo.degree_agg;
        let total_cores = topo.degree_core;

        for agg_src_idx in 0..deg_agg {
            let agg_src = topo.get_agg(src_pod, agg_src_idx);
            for core_idx in 0..total_cores {
                let core = topo.get_core(core_idx);
                for agg_dst_idx in 0..deg_agg {
                    let agg_dst = topo.get_agg(dst_pod, agg_dst_idx);
                    let nodes = [src, src_tor, agg_src, core, agg_dst, dst_tor, dst];
                    let path = convert_node_path_to_links(topo, &nodes);
                    let score = Self::path_score(link_load, &path);
                    if best.as_ref().map_or(true, |(s, _)| score < *s) {
                        best = Some((score, path));
                    }
                }
            }
        }
        best.map(|(_, p)| p).expect("no cross-pod path found")
    }

    fn recompute_all(&mut self, ctx: &MLContext, topo: &FatTree<FatTreeCruxRouter>) {
        let mut link_load: HashMap<LinkId, usize> = HashMap::new();
        let link_bps = topo.link_bandwidth_bps();

        let mut flows: Vec<(JobId, usize, usize, usize)> = Vec::new();
        for (&jid, specs) in self.core.jobs.iter() {
            for f in specs {
                flows.push((jid, f.job_flow_idx, f.src_host_index, f.dst_host_index));
            }
        }

        let mut job_intensity: HashMap<JobId, f64> = HashMap::new();
        flows.sort_by(|a, b| {
            let ia = *job_intensity
                .entry(a.0)
                .or_insert_with(|| self.job_intensity_us(ctx, link_bps, a.0));
            let ib = *job_intensity
                .entry(b.0)
                .or_insert_with(|| self.job_intensity_us(ctx, link_bps, b.0));
            ib.partial_cmp(&ia)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
        });

        let mut router = topo.router.borrow_mut();
        router.clear_templates();

        let mut path_cache: HashMap<(usize, usize), Path> = HashMap::new();

        for (job_id, job_flow_idx, src_h, dst_h) in flows {
            let path = if let Some(cached) = path_cache.get(&(src_h, dst_h)) {
                cached.clone()
            } else {
                let src = topo.get_host_by_index(src_h).expect("invalid src host");
                let dst = topo.get_host_by_index(dst_h).expect("invalid dst host");
                let new_path = Self::compute_best_path(topo, src, dst, &link_load);
                for lid in &new_path { *link_load.entry(*lid).or_insert(0) += 1; }
                path_cache.insert((src_h, dst_h), new_path.clone());
                new_path
            };
            router.inject_template(job_id, job_flow_idx, path);
        }
    }
}

impl<S, FS> SystemModule<FatTree<FatTreeCruxRouter>, S, FS> for FatTreeCruxSystem
where
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_job_scheduled(
        &mut self, _now_us: u64, _ctx: &MLContext, _job_id: JobId, job: &MLJob,
        _topo: &FatTree<FatTreeCruxRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {
        self.core.record_job(job);
    }

    fn on_job_completed(
        &mut self, _now_us: u64, _ctx: &MLContext, job_id: JobId, _job: &MLJob,
        _topo: &FatTree<FatTreeCruxRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {
        self.core.remove_job(job_id);
    }

    fn on_reconfigure(
        &mut self, _now_us: u64, ctx: &MLContext,
        topo: &FatTree<FatTreeCruxRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) -> Option<crate::simulator::system::MigrationPlan> {
        self.recompute_all(ctx, topo);
        None
    }
}
