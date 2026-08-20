//! Fat-Tree MonkeyTree + Perfect Routing
//!
//! Combines MonkeyTree's fragmentation monitoring and ILP-based migration
//! with the two-phase edge coloring perfect routing for fat tree topologies.
//!
//! Uses the same cross-ToR fragmentation definition as the spine version:
//! a segment is fragmented if its workers span multiple ToRs.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::network::topology::{FatTree, FatTreeTopology, Topology};
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::ml_worker::WorkerId;
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::system::{SystemModule, MigrationPlan, MIGRATION_FLOW_IDX_BASE};

use crate::monkeytree::fragmentation::{
    SegmentId, JobSegment, SegmentFragmentation, ToRSegmentStats,
    build_segments_from_context, print_segment_fragmentation_summary,
};
use crate::monkeytree::ilp::{
    SegmentILPInput, solve_segment_migration_ilp, compute_segment_migrations, SolveStatus,
    PodConstraintConfig,
};
use crate::monkeytree::system::MonkeyTreeConfig;
use crate::monkeytree::perfect_matching::{compute_edge_coloring, collapse_colors};

use super::perfect::{FatTreePerfectRouter, FatTreePerfectSystem};
use super::convert_node_path_to_links;

/// Compute cross-ToR fragmentation for a fat-tree topology.
///
/// Identical in concept to the spine version: a segment is fragmented if
/// its workers span more than one ToR, regardless of pod boundaries.
fn compute_fat_tree_fragmentation(
    ctx: &MLContext,
    hosts_per_tor: usize,
    num_tors: usize,
) -> SegmentFragmentation {
    let placements = ctx.placements.borrow();
    let segments = build_segments_from_context(ctx);

    let mut worker_to_host: HashMap<(JobId, WorkerId), usize> = HashMap::new();
    for (job_id, worker_hosts) in placements.iter() {
        for (wid, &host) in worker_hosts.iter() {
            worker_to_host.insert((*job_id, *wid), host);
        }
    }

    let mut fragmented_segments = HashSet::new();
    for segment in &segments {
        let tors: HashSet<usize> = segment.worker_ids.iter()
            .filter_map(|w| worker_to_host.get(&(segment.id.job_id, *w)))
            .map(|host| host / hosts_per_tor)
            .collect();
        if tors.len() > 1 {
            fragmented_segments.insert(segment.id);
        }
    }

    let segment_by_id: HashMap<SegmentId, &JobSegment> = segments.iter()
        .map(|s| (s.id, s))
        .collect();

    let mut per_tor = Vec::with_capacity(num_tors);
    for tor_idx in 0..num_tors {
        let tor_host_start = tor_idx * hosts_per_tor;
        let tor_host_end = (tor_idx + 1) * hosts_per_tor;

        let mut segments_present = HashSet::new();
        let mut segment_worker_counts = HashMap::new();

        for segment in &segments {
            let workers_on_tor: usize = segment.worker_ids.iter()
                .filter(|w| {
                    worker_to_host.get(&(segment.id.job_id, **w))
                        .map(|&h| h >= tor_host_start && h < tor_host_end)
                        .unwrap_or(false)
                })
                .count();
            if workers_on_tor > 0 {
                segments_present.insert(segment.id);
                segment_worker_counts.insert(segment.id, workers_on_tor);
            }
        }

        let fragmented_count: usize = segments_present.iter()
            .filter(|s| fragmented_segments.contains(s))
            .filter_map(|s| segment_by_id.get(s))
            .map(|s| s.ring_count)
            .sum();

        per_tor.push(ToRSegmentStats {
            tor_index: tor_idx,
            segments_present,
            fragmented_segment_count: fragmented_count,
            segment_worker_counts,
        });
    }

    let max_fragmentation = per_tor.iter()
        .map(|s| s.fragmented_segment_count)
        .max()
        .unwrap_or(0);

    SegmentFragmentation {
        per_tor,
        fragmented_segments,
        segments,
        max_fragmentation,
    }
}

// ---------------------------------------------------------------------------
// FatTreeMonkeyTreePerfect
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct FatTreeMonkeyTreePerfect {
    config: MonkeyTreeConfig,
    perfect: FatTreePerfectSystem,
}

impl FatTreeMonkeyTreePerfect {
    pub fn new(config: MonkeyTreeConfig) -> Self {
        Self {
            config,
            perfect: FatTreePerfectSystem::new(),
        }
    }

    fn check_and_plan_migration(
        &self,
        ctx: &MLContext,
        topo: &FatTree<FatTreePerfectRouter>,
    ) -> Option<MigrationPlan> {
        let hosts_per_tor = topo.hosts_per_tor;
        let num_tors = topo.num_pods * topo.degree_tor;

        let frag = compute_fat_tree_fragmentation(ctx, hosts_per_tor, num_tors);

        println!(
            "[FatTreeMonkeyTreePerfect] Fragmentation: max={}, threshold={}, fragmented_segments={}",
            frag.max_fragmentation,
            self.config.fragmentation_threshold,
            frag.fragmented_segments.len()
        );

        if frag.max_fragmentation <= self.config.fragmentation_threshold {
            return None;
        }

        print_segment_fragmentation_summary(&frag, ctx.time_us.get());
        println!("[FatTreeMonkeyTreePerfect] Threshold exceeded! Solving ILP...");

        solve_and_migrate("[FatTreeMonkeyTreePerfect]", &frag, ctx, topo, &self.config)
    }
}

fn solve_and_migrate(
    label: &str,
    frag: &SegmentFragmentation,
    ctx: &MLContext,
    topo: &FatTree<FatTreePerfectRouter>,
    config: &MonkeyTreeConfig,
) -> Option<MigrationPlan> {
    let hosts_per_tor = topo.hosts_per_tor;
    let num_tors = topo.num_pods * topo.degree_tor;
    let placements = ctx.placements.borrow();

    let fragmented_segments: Vec<SegmentId> = frag.fragmented_segments.iter().copied().collect();
    let segments: HashMap<SegmentId, JobSegment> = frag.segments.iter()
        .map(|s| (s.id, s.clone()))
        .collect();

    let mut initial_allocation: HashMap<(SegmentId, usize), usize> = HashMap::new();
    for segment in &frag.segments {
        if let Some(worker_hosts) = placements.get(&segment.id.job_id) {
            for &wid in &segment.worker_ids {
                if let Some(&host) = worker_hosts.get(&wid) {
                    let tor = host / hosts_per_tor;
                    *initial_allocation.entry((segment.id, tor)).or_insert(0) += 1;
                }
            }
        }
    }

    let mut nonfrag_workers_per_tor = vec![0usize; num_tors];
    for segment in &frag.segments {
        if !frag.fragmented_segments.contains(&segment.id) {
            if let Some(worker_hosts) = placements.get(&segment.id.job_id) {
                for &wid in &segment.worker_ids {
                    if let Some(&host) = worker_hosts.get(&wid) {
                        let tor = host / hosts_per_tor;
                        nonfrag_workers_per_tor[tor] += 1;
                    }
                }
            }
        }
    }

    println!(
        "{} ILP input: {} fragmented segments, {} total segments",
        label, fragmented_segments.len(), segments.len()
    );

    let ilp_input = SegmentILPInput {
        fragmented_segments,
        segments,
        num_tors,
        initial_allocation,
        tor_capacity: hosts_per_tor,
        target_lambda: config.fragmentation_threshold,
        nonfrag_workers_per_tor,
        block_size: config.block_size,
        pod_config: None,
    };

    let ilp_start = Instant::now();
    let solution = match solve_segment_migration_ilp(&ilp_input) {
        Ok(sol) => sol,
        Err(e) => {
            eprintln!("{} ILP error: {}", label, e);
            return None;
        }
    };
    let ilp_duration = ilp_start.elapsed();
    println!("{} ILP solve time: {:.3}ms", label, ilp_duration.as_secs_f64() * 1000.0);

    if solution.status != SolveStatus::Optimal {
        println!("{} ILP status: {:?}", label, solution.status);
        return None;
    }
    if solution.num_moves == 0 {
        println!("{} No moves needed.", label);
        return None;
    }

    println!("{} ILP solved: {} moves", label, solution.num_moves);

    let placements_map: HashMap<JobId, crate::utils::DHashMap<WorkerId, usize>> =
        placements.iter().map(|(k, v)| (*k, v.clone())).collect();

    let migrations = compute_segment_migrations(
        &ilp_input, &solution, &placements_map, hosts_per_tor,
    );

    if migrations.is_empty() {
        return None;
    }

    let total_workers: usize = migrations.iter().map(|m| m.worker_to_host.len()).sum();
    println!("{} Migration plan: {} jobs, {} workers", label, migrations.len(), total_workers);

    inject_migration_routes_for_fat_tree(&placements, &migrations, topo);

    Some(MigrationPlan { jobs: migrations })
}

fn inject_migration_routes_for_fat_tree(
    current_placements: &crate::utils::DHashMap<JobId, crate::utils::DHashMap<WorkerId, usize>>,
    migrations: &[crate::simulator::system::JobMigration],
    topo: &FatTree<FatTreePerfectRouter>,
) {
    let mut router = topo.router.borrow_mut();
    let hosts_per_tor = topo.hosts_per_tor;
    let tors_per_pod = topo.degree_tor;
    let num_agg = topo.degree_agg;
    let total_cores = topo.degree_core;

    let mut migration_flows: Vec<(JobId, usize, usize, usize)> = Vec::new();

    for jm in migrations {
        let cur = match current_placements.get(&jm.job_id) {
            Some(c) => c,
            None => continue,
        };
        for (&wid, &new_host) in jm.worker_to_host.iter() {
            let old_host = match cur.get(&wid) { Some(&h) => h, None => continue };
            if old_host == new_host { continue; }
            let mig_idx = MIGRATION_FLOW_IDX_BASE + wid;
            migration_flows.push((jm.job_id, mig_idx, old_host, new_host));
        }
    }

    if migration_flows.is_empty() { return; }

    let origin_idx = tors_per_pod;
    let target_idx = tors_per_pod + 1;
    let coloring_n = tors_per_pod + 2;
    let num_pods = topo.num_pods;

    struct MigFlowInfo {
        src_h: usize,
        dst_h: usize,
        src_pod: usize,
        dst_pod: usize,
        src_tor_local: usize,
        dst_tor_local: usize,
        job_id: JobId,
        mig_flow_idx: usize,
    }

    let mut cross_tor_flows: Vec<MigFlowInfo> = Vec::new();
    for &(job_id, mig_flow_idx, src_h, dst_h) in migration_flows.iter() {
        let src_tor = src_h / hosts_per_tor;
        let dst_tor = dst_h / hosts_per_tor;
        if src_tor == dst_tor { continue; }
        cross_tor_flows.push(MigFlowInfo {
            src_h, dst_h,
            src_pod: src_h / (tors_per_pod * hosts_per_tor),
            dst_pod: dst_h / (tors_per_pod * hosts_per_tor),
            src_tor_local: src_tor % tors_per_pod,
            dst_tor_local: dst_tor % tors_per_pod,
            job_id, mig_flow_idx,
        });
    }

    let mut agg_src_map: HashMap<usize, usize> = HashMap::new();
    let mut agg_dst_map: HashMap<usize, usize> = HashMap::new();

    for pod in 0..num_pods {
        let mut edges: Vec<(usize, usize, usize)> = Vec::new();
        let mut edge_map: Vec<(u8, usize)> = Vec::new();

        for (fi, f) in cross_tor_flows.iter().enumerate() {
            if f.src_pod == pod && f.dst_pod == pod {
                let local_id = edges.len();
                edges.push((f.src_tor_local, f.dst_tor_local, local_id));
                edge_map.push((0, fi));
            } else if f.src_pod == pod && f.dst_pod != pod {
                let local_id = edges.len();
                edges.push((f.src_tor_local, origin_idx, local_id));
                edge_map.push((1, fi));
            } else if f.dst_pod == pod && f.src_pod != pod {
                let local_id = edges.len();
                edges.push((target_idx, f.dst_tor_local, local_id));
                edge_map.push((2, fi));
            }
        }

        if edges.is_empty() { continue; }

        let coloring = compute_edge_coloring(coloring_n, edges);
        let final_coloring = if coloring.num_colors > num_agg {
            collapse_colors(&coloring, num_agg)
        } else {
            coloring.edge_to_color.clone()
        };

        for (local_id, &(etype, fi)) in edge_map.iter().enumerate() {
            if let Some(&color) = final_coloring.get(&local_id) {
                match etype {
                    0 => {
                        agg_src_map.insert(fi, color);
                        agg_dst_map.insert(fi, color);
                    }
                    1 => { agg_src_map.insert(fi, color); }
                    2 => { agg_dst_map.insert(fi, color); }
                    _ => {}
                }
            }
        }
    }

    let mut next_core = 0usize;
    for (fi, f) in cross_tor_flows.iter().enumerate() {
        let src_node = topo.get_host_by_index(f.src_h).unwrap();
        let dst_node = topo.get_host_by_index(f.dst_h).unwrap();
        let src_tor = topo.get_host_tor(src_node);
        let dst_tor = topo.get_host_tor(dst_node);
        let agg_src_idx = agg_src_map.get(&fi).copied().unwrap_or(0);

        let path = if f.src_pod == f.dst_pod {
            let agg = topo.get_agg(f.src_pod, agg_src_idx);
            convert_node_path_to_links(topo, &[src_node, src_tor, agg, dst_tor, dst_node])
        } else {
            let agg_dst_idx = agg_dst_map.get(&fi).copied().unwrap_or(0);
            let core_idx = next_core % total_cores.max(1);
            next_core += 1;
            let agg_src = topo.get_agg(f.src_pod, agg_src_idx);
            let core = topo.get_core(core_idx);
            let agg_dst = topo.get_agg(f.dst_pod, agg_dst_idx);
            convert_node_path_to_links(topo, &[src_node, src_tor, agg_src, core, agg_dst, dst_tor, dst_node])
        };
        router.inject_path(f.job_id, f.mig_flow_idx, path);
    }
}

impl<S, FS> SystemModule<FatTree<FatTreePerfectRouter>, S, FS> for FatTreeMonkeyTreePerfect
where
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_job_scheduled(
        &mut self, _now_us: u64, _ctx: &MLContext, _job_id: JobId, job: &MLJob,
        _topo: &FatTree<FatTreePerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {
        self.perfect.record_job(job);
    }

    fn on_job_completed(
        &mut self, _now_us: u64, _ctx: &MLContext, job_id: JobId, _job: &MLJob,
        _topo: &FatTree<FatTreePerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {
        self.perfect.remove_job(job_id);
    }

    fn on_reconfigure(
        &mut self, _now_us: u64, ctx: &MLContext,
        topo: &FatTree<FatTreePerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) -> Option<MigrationPlan> {
        self.perfect.recompute_routes_pub(ctx, topo);
        self.check_and_plan_migration(ctx, topo)
    }

    fn on_migration_end(
        &mut self, _now_us: u64, ctx: &MLContext, job_id: JobId, job: &MLJob,
        topo: &FatTree<FatTreePerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {
        self.perfect.remove_job(job_id);
        self.perfect.record_job(job);
        self.perfect.recompute_routes_pub(ctx, topo);
    }
}

// ---------------------------------------------------------------------------
// MonkeyTree3: cross-ToR + cross-pod fragmentation
// ---------------------------------------------------------------------------

/// Compute combined fragmentation for MonkeyTree3.
///
/// Returns (cross_tor_max, cross_pod_max) where:
/// - cross_tor_max: same as compute_fat_tree_fragmentation (max per-ToR weighted frag)
/// - cross_pod_max: for each pod, sum of ring_count for segments spanning multiple pods
fn compute_fat_tree_fragmentation_with_pod(
    ctx: &MLContext,
    hosts_per_tor: usize,
    tors_per_pod: usize,
    num_pods: usize,
) -> (SegmentFragmentation, usize) {
    let num_tors = num_pods * tors_per_pod;
    let hosts_per_pod = hosts_per_tor * tors_per_pod;
    let frag = compute_fat_tree_fragmentation(ctx, hosts_per_tor, num_tors);

    let placements = ctx.placements.borrow();
    let mut worker_to_host: HashMap<(JobId, WorkerId), usize> = HashMap::new();
    for (job_id, worker_hosts) in placements.iter() {
        for (wid, &host) in worker_hosts.iter() {
            worker_to_host.insert((*job_id, *wid), host);
        }
    }

    let segment_by_id: HashMap<SegmentId, &JobSegment> = frag.segments.iter()
        .map(|s| (s.id, s))
        .collect();

    let mut per_pod_frag = vec![0usize; num_pods];
    for segment in &frag.segments {
        let pods: HashSet<usize> = segment.worker_ids.iter()
            .filter_map(|w| worker_to_host.get(&(segment.id.job_id, *w)))
            .map(|&host| host / hosts_per_pod)
            .collect();
        if pods.len() > 1 {
            let rc = segment_by_id.get(&segment.id)
                .map(|s| s.ring_count)
                .unwrap_or(1);
            for &pod in &pods {
                per_pod_frag[pod] += rc;
            }
        }
    }

    let cross_pod_max = per_pod_frag.iter().copied().max().unwrap_or(0);
    (frag, cross_pod_max)
}

#[derive(Debug)]
pub struct FatTreeMonkeyTree3 {
    config: MonkeyTreeConfig,
    perfect: FatTreePerfectSystem,
}

impl FatTreeMonkeyTree3 {
    pub fn new(config: MonkeyTreeConfig) -> Self {
        Self {
            config,
            perfect: FatTreePerfectSystem::new(),
        }
    }

    fn check_and_plan_migration(
        &self,
        ctx: &MLContext,
        topo: &FatTree<FatTreePerfectRouter>,
    ) -> Option<MigrationPlan> {
        let hosts_per_tor = topo.hosts_per_tor;
        let tors_per_pod = topo.degree_tor;
        let num_pods = topo.num_pods;
        let num_tors = num_pods * tors_per_pod;

        let (frag, cross_pod_max) = compute_fat_tree_fragmentation_with_pod(
            ctx, hosts_per_tor, tors_per_pod, num_pods,
        );

        let threshold = self.config.fragmentation_threshold;
        let triggered_by_tor = frag.max_fragmentation > threshold;
        let triggered_by_pod = cross_pod_max > threshold;

        println!(
            "[MonkeyTree3] Frag: tor_max={}, pod_max={}, threshold={}, fragmented_segments={}",
            frag.max_fragmentation, cross_pod_max, threshold, frag.fragmented_segments.len()
        );

        if !triggered_by_tor && !triggered_by_pod {
            return None;
        }

        print_segment_fragmentation_summary(&frag, ctx.time_us.get());
        println!(
            "[MonkeyTree3] Threshold exceeded (tor={}, pod={})! Solving ILP...",
            triggered_by_tor, triggered_by_pod
        );

        let placements = ctx.placements.borrow();

        let fragmented_segments: Vec<SegmentId> = frag.fragmented_segments.iter().copied().collect();
        let segments: HashMap<SegmentId, JobSegment> = frag.segments.iter()
            .map(|s| (s.id, s.clone()))
            .collect();

        let mut initial_allocation: HashMap<(SegmentId, usize), usize> = HashMap::new();
        for segment in &frag.segments {
            if let Some(worker_hosts) = placements.get(&segment.id.job_id) {
                for &wid in &segment.worker_ids {
                    if let Some(&host) = worker_hosts.get(&wid) {
                        let tor = host / hosts_per_tor;
                        *initial_allocation.entry((segment.id, tor)).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut nonfrag_workers_per_tor = vec![0usize; num_tors];
        for segment in &frag.segments {
            if !frag.fragmented_segments.contains(&segment.id) {
                if let Some(worker_hosts) = placements.get(&segment.id.job_id) {
                    for &wid in &segment.worker_ids {
                        if let Some(&host) = worker_hosts.get(&wid) {
                            let tor = host / hosts_per_tor;
                            nonfrag_workers_per_tor[tor] += 1;
                        }
                    }
                }
            }
        }

        // MonkeyTree3 adds a real cross-pod fragmentation constraint to the ILP,
        // mirroring the per-ToR constraint at the pod granularity. The per-ToR
        // constraint is unchanged (so this is a superset of the original ILP).
        let ilp_input = SegmentILPInput {
            fragmented_segments,
            segments,
            num_tors,
            initial_allocation,
            tor_capacity: hosts_per_tor,
            target_lambda: threshold,
            nonfrag_workers_per_tor,
            block_size: self.config.block_size,
            pod_config: Some(PodConstraintConfig {
                num_pods,
                tors_per_pod,
                pod_lambda: threshold,
            }),
        };

        let ilp_start = Instant::now();
        let solution = match solve_segment_migration_ilp(&ilp_input) {
            Ok(sol) => sol,
            Err(e) => {
                eprintln!("[MonkeyTree3] ILP error: {}", e);
                return None;
            }
        };
        let ilp_duration = ilp_start.elapsed();
        println!("[MonkeyTree3] ILP solve time: {:.3}ms", ilp_duration.as_secs_f64() * 1000.0);

        if solution.status != SolveStatus::Optimal {
            println!("[MonkeyTree3] ILP status: {:?}", solution.status);
            return None;
        }
        if solution.num_moves == 0 {
            println!("[MonkeyTree3] No moves needed.");
            return None;
        }

        println!("[MonkeyTree3] ILP solved: {} moves", solution.num_moves);

        let placements_map: HashMap<JobId, crate::utils::DHashMap<WorkerId, usize>> =
            placements.iter().map(|(k, v)| (*k, v.clone())).collect();

        let migrations = compute_segment_migrations(
            &ilp_input, &solution, &placements_map, hosts_per_tor,
        );

        if migrations.is_empty() {
            return None;
        }

        let total_workers: usize = migrations.iter().map(|m| m.worker_to_host.len()).sum();
        println!("[MonkeyTree3] Migration plan: {} jobs, {} workers", migrations.len(), total_workers);

        self.inject_migration_routes(&placements, &migrations, topo);

        Some(MigrationPlan { jobs: migrations })
    }

    fn inject_migration_routes(
        &self,
        current_placements: &crate::utils::DHashMap<JobId, crate::utils::DHashMap<WorkerId, usize>>,
        migrations: &[crate::simulator::system::JobMigration],
        topo: &FatTree<FatTreePerfectRouter>,
    ) {
        inject_migration_routes_for_fat_tree(current_placements, migrations, topo);
    }
}

impl<S, FS> SystemModule<FatTree<FatTreePerfectRouter>, S, FS> for FatTreeMonkeyTree3
where
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_job_scheduled(
        &mut self, _now_us: u64, _ctx: &MLContext, _job_id: JobId, job: &MLJob,
        _topo: &FatTree<FatTreePerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {
        self.perfect.record_job(job);
    }

    fn on_job_completed(
        &mut self, _now_us: u64, _ctx: &MLContext, job_id: JobId, _job: &MLJob,
        _topo: &FatTree<FatTreePerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {
        self.perfect.remove_job(job_id);
    }

    fn on_reconfigure(
        &mut self, _now_us: u64, ctx: &MLContext,
        topo: &FatTree<FatTreePerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) -> Option<MigrationPlan> {
        self.perfect.recompute_routes_pub(ctx, topo);
        self.check_and_plan_migration(ctx, topo)
    }

    fn on_migration_end(
        &mut self, _now_us: u64, ctx: &MLContext, job_id: JobId, job: &MLJob,
        topo: &FatTree<FatTreePerfectRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {
        self.perfect.remove_job(job_id);
        self.perfect.record_job(job);
        self.perfect.recompute_routes_pub(ctx, topo);
    }
}
