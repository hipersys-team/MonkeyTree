//! MonkeyTree + Perfect Routing Combined System Module
//!
//! Combines MonkeyTree's fragmentation monitoring and ILP-based migration
//! with perfect routing using bipartite edge coloring for optimal spine assignment.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

use petgraph::graph::NodeIndex;

use crate::network::flow::FlowId;
use crate::network::routing::{Path, PathCell};
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::ml_worker::WorkerId;
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::system::{SystemModule, MigrationPlan, MIGRATION_FLOW_IDX_BASE};
use crate::spine::topology::{SpineTreeTopology, SpineTreeRouter, SpineTree};

use super::fragmentation::{compute_segment_fragmentation, SegmentId, JobSegment, print_segment_fragmentation_summary};
use super::ilp::{SegmentILPInput, solve_segment_migration_ilp, compute_segment_migrations, SolveStatus};
use super::system::MonkeyTreeConfig;
use super::perfect_matching::{compute_edge_coloring, collapse_colors};

// -----------------------------------------------------------------------------
// SpinePerfectRouter: uses edge coloring for optimal spine assignment
// -----------------------------------------------------------------------------

/// Router that uses bipartite edge coloring to assign flows to spine switches.
///
/// The algorithm:
/// 1. Build a bipartite graph where:
///    - Left vertices = source ToRs
///    - Right vertices = destination ToRs
///    - Edges = flows between different ToRs
/// 2. Compute an edge coloring (each color = a spine switch)
/// 3. If max_degree > num_spines, collapse colors using modulo
#[derive(Debug, Clone, Default)]
pub struct SpinePerfectRouter {
    context: Option<MLContext>,
    /// (job_id, job_flow_idx) → spine index to use
    flow_to_spine: HashMap<(JobId, usize), usize>,
    /// Cache of already-routed flows
    path_cache: HashMap<FlowId, PathCell>,
}

impl SpinePerfectRouter {
    pub fn new() -> Self {
        Self {
            context: None,
            flow_to_spine: HashMap::new(),
            path_cache: HashMap::new(),
        }
    }

    /// Clear all routing assignments
    pub fn clear(&mut self) {
        self.flow_to_spine.clear();
        self.path_cache.clear();
    }

    /// Inject a spine assignment for a specific (job_id, job_flow_idx)
    pub fn inject_spine_assignment(&mut self, job_id: JobId, job_flow_idx: usize, spine_idx: usize) {
        self.flow_to_spine.insert((job_id, job_flow_idx), spine_idx);
    }

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

impl SpineTreeRouter for SpinePerfectRouter {
    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn route(&mut self, topo: &impl SpineTreeTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        // Fast path - already cached
        if let Some(cell) = self.path_cache.get(&flow_id) {
            return cell.clone();
        }

        let src_leaf = topo.get_host_leaf(src);
        let dst_leaf = topo.get_host_leaf(dst);

        let mut nodes: Vec<NodeIndex> = Vec::with_capacity(5);
        nodes.push(src);
        nodes.push(src_leaf);

        if src_leaf != dst_leaf {
            // Need to go through a spine - look up which one
            let ctx = self.context.as_ref().expect("SpinePerfectRouter context not set");
            let map_ref = ctx.waiting_flows.borrow();
            let (job_id, job_flow_idx, _iter_idx, _src_w, _dst_w, _send_eid, _recv_eid) = map_ref
                .get(&flow_id)
                .copied()
                .unwrap_or_else(|| panic!("SpinePerfectRouter: missing mapping for flow_id {}", flow_id));
            drop(map_ref);

            let key = (job_id, job_flow_idx);
            let spine_idx = self.flow_to_spine.get(&key).copied().unwrap_or_else(|| {
                // Fallback to hash-based selection if no assignment exists
                // This shouldn't happen in normal operation but provides safety
                let hash = (job_id as usize).wrapping_mul(31).wrapping_add(job_flow_idx);
                hash % topo.num_spines()
            });

            let spine = topo.get_spine(spine_idx);
            nodes.push(spine);
            nodes.push(dst_leaf);
        }

        nodes.push(dst);

        let link_path = self.convert_node_path_to_links(topo, nodes);
        let cell = PathCell { path: Rc::new(RefCell::new(link_path)) };
        self.path_cache.insert(flow_id, cell.clone());
        cell
    }

    fn complete_flow(&mut self, flow_id: FlowId) {
        self.path_cache.remove(&flow_id);
    }
}

// -----------------------------------------------------------------------------
// Internal state: derive flows from active jobs
// -----------------------------------------------------------------------------

/// Flow template info - stores worker IDs, not host indices.
/// Host positions are looked up from ctx.placements at route computation time.
#[derive(Debug, Clone)]
pub struct FlowTemplateSpec {
    pub job_id: JobId,
    pub job_flow_idx: usize,
    pub src_worker_id: WorkerId,
    pub dst_worker_id: WorkerId,
    /// Whether this is a ring flow (AllReduce, StridedRing) or all-to-all flow.
    /// Ring flows get priority routing via bipartite edge coloring.
    pub is_ring_flow: bool,  // Kept as bool for routing decisions
}

#[derive(Debug, Default, Clone)]
pub struct JobInfo {
    pub flows: Vec<FlowTemplateSpec>,
}

#[derive(Debug, Default)]
pub struct PerfectCore {
    pub jobs: HashMap<JobId, JobInfo>,
}

impl PerfectCore {
    pub fn record_job(&mut self, job: &MLJob) {
        let mut info = JobInfo { flows: Vec::new() };

        for (&src_worker_id, worker) in job.workers.iter() {
            for ev in &worker.template_events {
                if ev.kind == crate::simulator::ml_worker::WorkerEventKind::FlowSend {
                    let send = ev.flow_send.as_ref().expect("FlowSend missing payload");
                    let dst_worker_id = send.dst_worker;
                    let flow_idx = *job
                        .send_template_to_flow_idx
                        .get(&(src_worker_id, ev.template_id))
                        .expect("missing send_template_to_flow_idx mapping");
                    info.flows.push(FlowTemplateSpec {
                        job_id: job.id,
                        job_flow_idx: flow_idx,
                        src_worker_id,
                        dst_worker_id,
                        is_ring_flow: send.is_ring_flow(),
                    });
                }
            }
        }

        info.flows.sort_by_key(|f| (f.job_id, f.job_flow_idx));
        self.jobs.insert(job.id, info);
    }

    pub fn remove_job(&mut self, job_id: JobId) {
        self.jobs.remove(&job_id);
    }
}

// -----------------------------------------------------------------------------
// MonkeyTreePerfect: Combined system module
// -----------------------------------------------------------------------------

/// MonkeyTree + Perfect Routing combined system module
///
/// Combines:
/// - MonkeyTree's fragmentation monitoring and ILP-based migration
/// - Perfect routing via bipartite edge coloring for optimal spine assignment
#[derive(Debug, Default)]
pub struct MonkeyTreePerfect {
    config: MonkeyTreeConfig,
    /// Job tracking and flow state
    core: PerfectCore,
}

impl MonkeyTreePerfect {
    pub fn new(config: MonkeyTreeConfig) -> Self {
        Self {
            config,
            core: PerfectCore::default(),
        }
    }

    pub fn with_threshold(threshold: usize) -> Self {
        Self::new(MonkeyTreeConfig {
            fragmentation_threshold: threshold,
            block_size: 1,
        })
    }

    // -------------------------------------------------------------------------
    // Perfect routing logic using edge coloring
    // -------------------------------------------------------------------------

    fn recompute_routes(&mut self, ctx: &MLContext, topo: &SpineTree<SpinePerfectRouter>) {
        let placements = ctx.placements.borrow();
        let hosts_per_leaf = topo.hosts_per_leaf;
        let num_spines = topo.num_spines;
        let num_tors = topo.num_leaves;

        // Separate ring flows (priority: bipartite edge coloring) from 
        // other flows (pipeline activations, all-to-all)
        // Ring flows use (src_tor, dst_tor) for edge coloring
        // Other flows use (src_host, dst_host) for consistent routing per host pair
        let mut ring_flows: Vec<(usize, usize, (JobId, usize))> = Vec::new();
        let mut other_flows: Vec<(usize, usize, (JobId, usize))> = Vec::new();

        for (&jid, info) in self.core.jobs.iter() {
            let job_placements = match placements.get(&jid) {
                Some(p) => p,
                None => continue,
            };

            for f in &info.flows {
                let src_host = match job_placements.get(&f.src_worker_id) {
                    Some(&h) => h,
                    None => continue,
                };
                let dst_host = match job_placements.get(&f.dst_worker_id) {
                    Some(&h) => h,
                    None => continue,
                };

                let src_tor = src_host / hosts_per_leaf;
                let dst_tor = dst_host / hosts_per_leaf;

                // Only include cross-ToR flows
                if src_tor != dst_tor {
                    if f.is_ring_flow {
                        // Ring flows: use ToR indices for edge coloring
                        ring_flows.push((src_tor, dst_tor, (jid, f.job_flow_idx)));
                    } else {
                        // Other flows: use host indices for per-host-pair routing
                        other_flows.push((src_host, dst_host, (jid, f.job_flow_idx)));
                    }
                }
            }
        }

        let mut router = topo.router.borrow_mut();
        router.clear();

        if ring_flows.is_empty() && other_flows.is_empty() {
            // No cross-ToR flows, nothing to do
            return;
        }

        // 1. Route ring flows using bipartite edge coloring
        if !ring_flows.is_empty() {
            let edges: Vec<(usize, usize, usize)> = ring_flows
                .iter()
                .enumerate()
                .map(|(idx, &(src, dst, _))| (src, dst, idx))
                .collect();

            let coloring = compute_edge_coloring(num_tors, edges);

            println!(
                "[MonkeyTreePerfect] Ring flows: {} flows, {} colors needed, {} spines available",
                ring_flows.len(),
                coloring.num_colors,
                num_spines
            );

            // Collapse colors if we have more than num_spines
            let final_coloring = if coloring.num_colors > num_spines {
                println!(
                    "[MonkeyTreePerfect] Collapsing {} colors into {} spines",
                    coloring.num_colors,
                    num_spines
                );
                collapse_colors(&coloring, num_spines)
            } else {
                coloring.edge_to_color.clone()
            };

            // Install spine assignments for ring flows
            for (edge_idx, &(_, _, (job_id, job_flow_idx))) in ring_flows.iter().enumerate() {
                if let Some(&spine_idx) = final_coloring.get(&edge_idx) {
                    router.inject_spine_assignment(job_id, job_flow_idx, spine_idx);
                }
            }
        }

        // 2. Route other flows (all-to-all, pipeline activations) by (src_host, dst_host)
        // All flows with the same source and destination host get the same spine
        if !other_flows.is_empty() {
            // Group flows by (src_host, dst_host) and assign each unique pair to a spine.
            // Per-source-leaf round-robin: each leaf spreads its outgoing all-to-all
            // flows evenly across ALL spines, starting from a staggered (per-leaf)
            // offset so leaves don't all pile onto spine 0. A single global counter
            // (next_spine % num_spines) aliases against the strided-ring flow pattern
            // at certain spine counts, creating per-leaf hotspots; keying the counter
            // on the source leaf guarantees even spread regardless of num_spines.
            let mut pair_to_spine: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
            let mut leaf_counter: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            
            for &(src_host, dst_host, _) in other_flows.iter() {
                pair_to_spine.entry((src_host, dst_host)).or_insert_with(|| {
                    let src_leaf = src_host / hosts_per_leaf;
                    let cnt = leaf_counter.entry(src_leaf).or_insert(0);
                    let spine = (src_leaf + *cnt) % num_spines;
                    *cnt += 1;
                    spine
                });
            }
            
            println!(
                "[MonkeyTreePerfect] Other flows: {} flows, {} unique host pairs, {} spines",
                other_flows.len(),
                pair_to_spine.len(),
                num_spines
            );

            // Assign each flow based on its (src_host, dst_host) pair
            for &(src_host, dst_host, (job_id, job_flow_idx)) in other_flows.iter() {
                let spine_idx = pair_to_spine[&(src_host, dst_host)];
                router.inject_spine_assignment(job_id, job_flow_idx, spine_idx);
            }
        }
    }

    // -------------------------------------------------------------------------
    // MonkeyTree migration logic
    // -------------------------------------------------------------------------

    fn check_and_plan_migration(
        &self,
        ctx: &MLContext,
        topo: &SpineTree<SpinePerfectRouter>,
    ) -> Option<MigrationPlan> {
        // Use segment-based fragmentation for pipeline-aware optimization
        let frag = compute_segment_fragmentation(ctx, topo);

        println!(
            "[MonkeyTreePerfect] Fragmentation: max={}, threshold={}, fragmented_segments={}",
            frag.max_fragmentation,
            self.config.fragmentation_threshold,
            frag.fragmented_segments.len()
        );

        if frag.max_fragmentation <= self.config.fragmentation_threshold {
            return None;
        }

        // Print detailed fragmentation state
        print_segment_fragmentation_summary(&frag, ctx.time_us.get());
        
        println!("[MonkeyTreePerfect] Threshold exceeded! Solving ILP...");

        let placements = ctx.placements.borrow();
        let hosts_per_leaf = topo.hosts_per_leaf;
        let num_tors = topo.num_leaves;

        let fragmented_segments: Vec<SegmentId> = frag.fragmented_segments.iter().copied().collect();

        // Build segment map
        let segments: HashMap<SegmentId, JobSegment> = frag.segments.iter()
            .map(|s| (s.id, s.clone()))
            .collect();

        // Initial allocation per segment
        let mut initial_allocation: HashMap<(SegmentId, usize), usize> = HashMap::new();
        for segment in &frag.segments {
            if let Some(worker_hosts) = placements.get(&segment.id.job_id) {
                for &wid in &segment.worker_ids {
                    if let Some(&host) = worker_hosts.get(&wid) {
                        let tor = host / hosts_per_leaf;
                        *initial_allocation.entry((segment.id, tor)).or_insert(0) += 1;
                    }
                }
            }
        }

        // Non-fragmented workers per ToR
        let mut nonfrag_workers_per_tor = vec![0; num_tors];
        for segment in &frag.segments {
            if !frag.fragmented_segments.contains(&segment.id) {
                if let Some(worker_hosts) = placements.get(&segment.id.job_id) {
                    for &wid in &segment.worker_ids {
                        if let Some(&host) = worker_hosts.get(&wid) {
                            let tor = host / hosts_per_leaf;
                            nonfrag_workers_per_tor[tor] += 1;
                        }
                    }
                }
            }
        }

        println!("[MonkeyTreePerfect] ILP input: {} fragmented segments, {} total segments",
            fragmented_segments.len(), segments.len());
        for seg_id in &fragmented_segments {
            if let Some(seg) = segments.get(seg_id) {
                println!("  Fragmented: {:?} workers={} rings={}", seg_id, seg.worker_ids.len(), seg.ring_count);
            }
        }
        
        let ilp_input = SegmentILPInput {
            fragmented_segments,
            segments,
            num_tors,
            initial_allocation,
            tor_capacity: hosts_per_leaf,
            target_lambda: self.config.fragmentation_threshold,
            nonfrag_workers_per_tor,
            block_size: self.config.block_size,
            pod_config: None,
        };

        let ilp_start = Instant::now();
        let solution = match solve_segment_migration_ilp(&ilp_input) {
            Ok(sol) => sol,
            Err(e) => {
                eprintln!("[MonkeyTreePerfect] ILP error: {}", e);
                return None;
            }
        };
        let ilp_duration = ilp_start.elapsed();

        println!("[MonkeyTreePerfect] ILP solve time: {:.3}ms", ilp_duration.as_secs_f64() * 1000.0);

        if solution.status != SolveStatus::Optimal {
            println!("[MonkeyTreePerfect] ILP status: {:?}", solution.status);
            return None;
        }

        if solution.num_moves == 0 {
            println!("[MonkeyTreePerfect] No moves needed.");
            return None;
        }

        println!("[MonkeyTreePerfect] ILP solved: {} moves", solution.num_moves);

        let placements_map: HashMap<JobId, crate::utils::DHashMap<WorkerId, usize>> = 
            placements.iter().map(|(k, v)| (*k, v.clone())).collect();

        let migrations = compute_segment_migrations(&ilp_input, &solution, &placements_map, hosts_per_leaf);

        if migrations.is_empty() {
            return None;
        }

        let total_workers: usize = migrations.iter().map(|m| m.worker_to_host.len()).sum();
        println!(
            "[MonkeyTreePerfect] Migration plan: {} jobs, {} workers",
            migrations.len(),
            total_workers
        );

        // Inject migration flow routes
        self.inject_migration_routes(&placements, &migrations, topo);

        Some(MigrationPlan { jobs: migrations })
    }

    /// Inject spine assignments for migration flows.
    /// Each migration flow for worker W uses job_flow_idx = MIGRATION_FLOW_IDX_BASE + W.
    fn inject_migration_routes(
        &self,
        current_placements: &crate::utils::DHashMap<JobId, crate::utils::DHashMap<WorkerId, usize>>,
        migrations: &[crate::simulator::system::JobMigration],
        topo: &SpineTree<SpinePerfectRouter>,
    ) {
        let mut router = topo.router.borrow_mut();
        let hosts_per_leaf = topo.hosts_per_leaf;
        let num_spines = topo.num_spines;

        // Collect migration flows for edge coloring
        let mut migration_edges: Vec<(usize, usize, (JobId, usize))> = Vec::new();

        for job_migration in migrations {
            let job_id = job_migration.job_id;
            let current_hosts = match current_placements.get(&job_id) {
                Some(hosts) => hosts,
                None => continue,
            };

            for (&worker_id, &new_host) in job_migration.worker_to_host.iter() {
                let old_host = match current_hosts.get(&worker_id) {
                    Some(&h) => h,
                    None => continue,
                };

                if old_host == new_host {
                    continue;
                }

                let src_tor = old_host / hosts_per_leaf;
                let dst_tor = new_host / hosts_per_leaf;

                // Only need spine assignment for cross-ToR migrations
                if src_tor != dst_tor {
                    let migration_flow_idx = MIGRATION_FLOW_IDX_BASE + worker_id;
                    migration_edges.push((src_tor, dst_tor, (job_id, migration_flow_idx)));
                }
            }
        }

        if migration_edges.is_empty() {
            return;
        }

        // Build edges for edge coloring (using unique IDs for migration flows)
        let edges: Vec<(usize, usize, usize)> = migration_edges
            .iter()
            .enumerate()
            .map(|(idx, &(src, dst, _))| (src, dst, idx))
            .collect();

        // Compute edge coloring for migration flows
        let coloring = compute_edge_coloring(topo.num_leaves, edges);

        // Collapse if needed
        let final_coloring = if coloring.num_colors > num_spines {
            collapse_colors(&coloring, num_spines)
        } else {
            coloring.edge_to_color.clone()
        };

        // Inject spine assignments
        for (edge_idx, &(_, _, (job_id, migration_flow_idx))) in migration_edges.iter().enumerate() {
            if let Some(&spine_idx) = final_coloring.get(&edge_idx) {
                router.inject_spine_assignment(job_id, migration_flow_idx, spine_idx);
            }
        }
    }
}

impl<S, FS> SystemModule<SpineTree<SpinePerfectRouter>, S, FS> for MonkeyTreePerfect
where
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_init(
        &mut self,
        _ctx: &MLContext,
        _topo: &SpineTree<SpinePerfectRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // No initialization needed
    }
    fn on_job_scheduled(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        job: &MLJob,
        _topo: &SpineTree<SpinePerfectRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // Record job for routing
        self.core.record_job(job);
    }

    fn on_job_completed(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        job_id: JobId,
        _job: &MLJob,
        _topo: &SpineTree<SpinePerfectRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // Remove job from tracking
        self.core.remove_job(job_id);
    }

    fn on_reconfigure(
        &mut self,
        _now_us: u64,
        ctx: &MLContext,
        topo: &SpineTree<SpinePerfectRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) -> Option<MigrationPlan> {
        // 1. Recompute routes using edge coloring
        self.recompute_routes(ctx, topo);

        // 2. Check fragmentation and plan migration if needed
        self.check_and_plan_migration(ctx, topo)
    }

    fn on_migration_end(
        &mut self,
        _now_us: u64,
        ctx: &MLContext,
        job_id: JobId,
        job: &MLJob,
        topo: &SpineTree<SpinePerfectRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // Re-record the job with updated worker IDs after rank reassignment
        self.core.remove_job(job_id);
        self.core.record_job(job);
        
        // Recompute all routes after migration completes
        println!("[MonkeyTreePerfect] Recomputing routes after migration");
        self.recompute_routes(ctx, topo);
    }
}

