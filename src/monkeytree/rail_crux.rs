//! MonkeyTree + Crux combined system module for rail-optimized topologies.

use std::collections::HashMap;
use std::time::Instant;

use petgraph::graph::NodeIndex;

use crate::network::routing::Path;
use crate::network::topology::{LinkId, Topology};
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::ml_worker::WorkerId;
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::system::{SystemModule, MigrationPlan, MIGRATION_FLOW_IDX_BASE};
use crate::rail::{RailTree, RailTopology, RailCruxRouter};
use crate::rail::routing::build_rail_node_path;

use super::rail_fragmentation::compute_pod_fragmentation;
use super::fragmentation::{SegmentId, JobSegment, print_segment_fragmentation_summary};
use super::ilp::{SegmentILPInput, solve_segment_migration_ilp, compute_segment_migrations, SolveStatus};
use super::system::MonkeyTreeConfig;

// ---------------------------------------------------------------------------
// Flow tracking (same pattern as spine crux)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FlowTemplateSpec {
    job_id: JobId,
    job_flow_idx: usize,
    src_worker_id: WorkerId,
    dst_worker_id: WorkerId,
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
                    });
                }
            }
        }
        info.flows.sort_by_key(|f| (f.job_id, f.job_flow_idx));
        self.jobs.insert(job.id, info);
    }

    fn remove_job(&mut self, job_id: JobId) { self.jobs.remove(&job_id); }
}

fn convert_node_path_to_links(topo: &impl Topology, nodes: &[NodeIndex]) -> Path {
    let mut link_path = Vec::with_capacity(nodes.len().saturating_sub(1));
    let graph = topo.topology();
    for window in nodes.windows(2) {
        let edge_idx = graph.find_edge(window[0], window[1]).expect("Path over nonexistent edge");
        link_path.push(graph.edge_weight(edge_idx).unwrap().id);
    }
    link_path
}

// ---------------------------------------------------------------------------
// RailMonkeyTreeCrux
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct RailMonkeyTreeCrux {
    config: MonkeyTreeConfig,
    crux_core: CruxCore,
}

impl RailMonkeyTreeCrux {
    pub fn new(config: MonkeyTreeConfig) -> Self {
        Self { config, crux_core: CruxCore::default() }
    }

    pub fn with_threshold(threshold: usize) -> Self {
        Self::new(MonkeyTreeConfig { fragmentation_threshold: threshold, block_size: 1 })
    }

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
        (total_compute_us as f64) + if link_bps > 0.0 { (total_send_bytes as f64) * 8_000_000.0 / link_bps } else { 0.0 }
    }

    fn path_score(link_load: &HashMap<LinkId, usize>, path: &[LinkId]) -> usize {
        path.iter().map(|lid| *link_load.get(lid).unwrap_or(&0)).max().unwrap_or(0)
    }

    fn compute_best_path(
        topo: &RailTree<RailCruxRouter>,
        src: NodeIndex,
        dst: NodeIndex,
        link_load: &HashMap<LinkId, usize>,
    ) -> Path {
        let src_pod = topo.host_pod(src);
        let dst_pod = topo.host_pod(dst);

        if src_pod == dst_pod {
            let nodes = build_rail_node_path(topo, src, dst, None);
            return convert_node_path_to_links(topo, &nodes);
        }

        let mut best: Option<(usize, Path)> = None;
        for spine_idx in 0..topo.num_spines() {
            let spine = topo.get_spine(spine_idx);
            let nodes = build_rail_node_path(topo, src, dst, Some(spine));
            let path = convert_node_path_to_links(topo, &nodes);
            let score = Self::path_score(link_load, &path);
            if best.as_ref().map_or(true, |(s, _)| score < *s) {
                best = Some((score, path));
            }
        }
        best.map(|(_, p)| p).expect("RailMonkeyTreeCrux: no path found")
    }

    fn recompute_routes(&mut self, ctx: &MLContext, topo: &RailTree<RailCruxRouter>) {
        let mut link_load: HashMap<LinkId, usize> = HashMap::new();
        let link_bps = topo.link_bandwidth_bps();
        let placements = ctx.placements.borrow();

        let mut flows: Vec<(JobId, usize, usize, usize)> = Vec::new();
        for (&jid, info) in self.crux_core.jobs.iter() {
            let jp = match placements.get(&jid) { Some(p) => p, None => continue };
            for f in &info.flows {
                let src_host = match jp.get(&f.src_worker_id) { Some(&h) => h, None => continue };
                let dst_host = match jp.get(&f.dst_worker_id) { Some(&h) => h, None => continue };
                flows.push((jid, f.job_flow_idx, src_host, dst_host));
            }
        }

        let mut job_intensity: HashMap<JobId, f64> = HashMap::new();
        flows.sort_by(|a, b| {
            let ia = *job_intensity.entry(a.0).or_insert_with(|| self.job_intensity_us(ctx, link_bps, a.0));
            let ib = *job_intensity.entry(b.0).or_insert_with(|| self.job_intensity_us(ctx, link_bps, b.0));
            ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
        });

        {
            let mut router = topo.router.borrow_mut();
            router.clear_templates();

            let mut path_cache: HashMap<(usize, usize), Path> = HashMap::new();
            for (job_id, job_flow_idx, src_h, dst_h) in flows.into_iter() {
                let path = if let Some(cached) = path_cache.get(&(src_h, dst_h)) {
                    cached.clone()
                } else {
                    let src = topo.get_host_by_index(src_h).expect("invalid src host");
                    let dst = topo.get_host_by_index(dst_h).expect("invalid dst host");
                    let new_path = Self::compute_best_path(topo, src, dst, &link_load);
                    for lid in new_path.iter() { *link_load.entry(*lid).or_insert(0) += 1; }
                    path_cache.insert((src_h, dst_h), new_path.clone());
                    new_path
                };
                router.inject_template(job_id, job_flow_idx, path);
            }
        }
    }

    fn check_and_plan_migration(&self, ctx: &MLContext, topo: &RailTree<RailCruxRouter>) -> Option<MigrationPlan> {
        let frag = compute_pod_fragmentation(ctx, topo);

        println!(
            "[RailMonkeyTreeCrux] Fragmentation: max={}, threshold={}, fragmented_segments={}",
            frag.max_fragmentation, self.config.fragmentation_threshold, frag.fragmented_segments.len()
        );

        if frag.max_fragmentation <= self.config.fragmentation_threshold {
            return None;
        }

        print_segment_fragmentation_summary(&frag, ctx.time_us.get());
        println!("[RailMonkeyTreeCrux] Threshold exceeded! Solving ILP...");

        let placements = ctx.placements.borrow();
        let gpus_per_pod = topo.blocks_per_pod * topo.block_size;
        let num_pods = topo.num_pods;

        let fragmented_segments: Vec<SegmentId> = frag.fragmented_segments.iter().copied().collect();
        let segments: HashMap<SegmentId, JobSegment> = frag.segments.iter().map(|s| (s.id, s.clone())).collect();

        let mut initial_allocation: HashMap<(SegmentId, usize), usize> = HashMap::new();
        for segment in &frag.segments {
            if let Some(wh) = placements.get(&segment.id.job_id) {
                for &wid in &segment.worker_ids {
                    if let Some(&host) = wh.get(&wid) {
                        *initial_allocation.entry((segment.id, host / gpus_per_pod)).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut nonfrag_workers_per_pod = vec![0; num_pods];
        for segment in &frag.segments {
            if !frag.fragmented_segments.contains(&segment.id) {
                if let Some(wh) = placements.get(&segment.id.job_id) {
                    for &wid in &segment.worker_ids {
                        if let Some(&host) = wh.get(&wid) {
                            nonfrag_workers_per_pod[host / gpus_per_pod] += 1;
                        }
                    }
                }
            }
        }

        let ilp_input = SegmentILPInput {
            fragmented_segments,
            segments,
            num_tors: num_pods,
            initial_allocation,
            tor_capacity: gpus_per_pod,
            target_lambda: self.config.fragmentation_threshold,
            nonfrag_workers_per_tor: nonfrag_workers_per_pod,
            block_size: self.config.block_size,
            pod_config: None,
        };

        let ilp_start = Instant::now();
        let solution = match solve_segment_migration_ilp(&ilp_input) {
            Ok(sol) => sol,
            Err(e) => { eprintln!("[RailMonkeyTreeCrux] ILP error: {}", e); return None; }
        };
        println!("[RailMonkeyTreeCrux] ILP solve time: {:.3}ms", ilp_start.elapsed().as_secs_f64() * 1000.0);

        if solution.status != SolveStatus::Optimal { return None; }
        if solution.num_moves == 0 { return None; }

        println!("[RailMonkeyTreeCrux] ILP solved: {} moves", solution.num_moves);

        let placements_map: HashMap<JobId, crate::utils::DHashMap<WorkerId, usize>> =
            placements.iter().map(|(k, v)| (*k, v.clone())).collect();
        let migrations = compute_segment_migrations(&ilp_input, &solution, &placements_map, gpus_per_pod);

        if migrations.is_empty() { return None; }

        self.inject_migration_routes(&placements, &migrations, topo);

        Some(MigrationPlan { jobs: migrations })
    }

    fn inject_migration_routes(
        &self,
        current_placements: &crate::utils::DHashMap<JobId, crate::utils::DHashMap<WorkerId, usize>>,
        migrations: &[crate::simulator::system::JobMigration],
        topo: &RailTree<RailCruxRouter>,
    ) {
        let mut router = topo.router.borrow_mut();
        let link_load: HashMap<LinkId, usize> = HashMap::new();

        for job_migration in migrations {
            let job_id = job_migration.job_id;
            let current_hosts = match current_placements.get(&job_id) { Some(h) => h, None => continue };

            for (&worker_id, &new_host) in job_migration.worker_to_host.iter() {
                let old_host = match current_hosts.get(&worker_id) { Some(&h) => h, None => continue };
                if old_host == new_host { continue; }

                let src = topo.get_host_by_index(old_host).expect("invalid src host");
                let dst = topo.get_host_by_index(new_host).expect("invalid dst host");

                let src_pod = topo.host_pod(src);
                let dst_pod = topo.host_pod(dst);
                let path = if src_pod == dst_pod {
                    let nodes = build_rail_node_path(topo, src, dst, None);
                    convert_node_path_to_links(topo, &nodes)
                } else {
                    // Use least-loaded spine (with empty load map for migration routes)
                    let mut best: Option<(usize, Path)> = None;
                    for spine_idx in 0..topo.num_spines() {
                        let spine = topo.get_spine(spine_idx);
                        let nodes = build_rail_node_path(topo, src, dst, Some(spine));
                        let p = convert_node_path_to_links(topo, &nodes);
                        let score = Self::path_score(&link_load, &p);
                        if best.as_ref().map_or(true, |(s, _)| score < *s) { best = Some((score, p)); }
                    }
                    best.map(|(_, p)| p).expect("no path")
                };

                let migration_flow_idx = MIGRATION_FLOW_IDX_BASE + worker_id;
                router.inject_template(job_id, migration_flow_idx, path);
            }
        }
    }
}

impl<S, FS> SystemModule<RailTree<RailCruxRouter>, S, FS> for RailMonkeyTreeCrux
where S: JobScheduler, FS: FlowScheduler,
{
    fn on_init(&mut self, _ctx: &MLContext, _topo: &RailTree<RailCruxRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) {}

    fn on_job_scheduled(&mut self, _now_us: u64, _ctx: &MLContext, _job_id: JobId, job: &MLJob, _topo: &RailTree<RailCruxRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) {
        self.crux_core.record_job(job);
    }

    fn on_job_completed(&mut self, _now_us: u64, _ctx: &MLContext, job_id: JobId, _job: &MLJob, _topo: &RailTree<RailCruxRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) {
        self.crux_core.remove_job(job_id);
    }

    fn on_reconfigure(&mut self, _now_us: u64, ctx: &MLContext, topo: &RailTree<RailCruxRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) -> Option<MigrationPlan> {
        self.recompute_routes(ctx, topo);
        self.check_and_plan_migration(ctx, topo)
    }

    fn on_migration_end(&mut self, _now_us: u64, ctx: &MLContext, job_id: JobId, job: &MLJob, topo: &RailTree<RailCruxRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS) {
        self.crux_core.remove_job(job_id);
        self.crux_core.record_job(job);
        self.recompute_routes(ctx, topo);
    }
}
