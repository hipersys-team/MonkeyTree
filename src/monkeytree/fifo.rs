//! FIFO Defragmentation + Perfect Routing System Module
//!
//! Similar to MonkeyTree, but instead of using an ILP to compute optimal migrations,
//! it simply sorts jobs by job ID and places them linearly through the cluster.
//! This is a simpler baseline that still uses perfect routing via edge coloring.

use std::collections::HashMap;

// Debug flag for FIFO migration tracking. Enable to trace placement decisions.
const DEBUG_FIFO: bool = false;

use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::ml_worker::WorkerId;
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::system::{SystemModule, MigrationPlan, JobMigration, MIGRATION_FLOW_IDX_BASE};
use crate::spine::topology::{SpineTree};
use crate::utils::DHashMap;

use super::fragmentation::compute_segment_fragmentation;
use super::system::MonkeyTreeConfig;
use super::perfect::{SpinePerfectRouter, PerfectCore};
use super::perfect_matching::{compute_edge_coloring, collapse_colors};

// -----------------------------------------------------------------------------
// FifoPerfect: FIFO defragmentation with perfect routing
// -----------------------------------------------------------------------------

/// FIFO Defragmentation + Perfect Routing system module
///
/// Combines:
/// - FIFO-based defragmentation: sorts jobs by ID and places linearly
/// - Perfect routing via bipartite edge coloring for optimal spine assignment
///
/// When fragmentation exceeds the threshold, this system:
/// 1. Sorts all active jobs by job ID
/// 2. Places workers linearly through the cluster (job 0 first, then job 1, etc.)
/// 3. Computes migrations needed to achieve this placement
#[derive(Debug, Default)]
pub struct FifoPerfect {
    config: MonkeyTreeConfig,
    /// Job tracking and flow state (shared with MonkeyTreePerfect)
    core: PerfectCore,
}

impl FifoPerfect {
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
    // Perfect routing logic (same as MonkeyTreePerfect)
    // -------------------------------------------------------------------------

    fn recompute_routes(&mut self, ctx: &MLContext, topo: &SpineTree<SpinePerfectRouter>) {
        let placements = ctx.placements.borrow();
        let hosts_per_leaf = topo.hosts_per_leaf;
        let num_spines = topo.num_spines;
        let num_tors = topo.num_leaves;

        // Separate ring flows from other flows
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

                if src_tor != dst_tor {
                    if f.is_ring_flow {
                        ring_flows.push((src_tor, dst_tor, (jid, f.job_flow_idx)));
                    } else {
                        other_flows.push((src_host, dst_host, (jid, f.job_flow_idx)));
                    }
                }
            }
        }

        let mut router = topo.router.borrow_mut();
        router.clear();

        if ring_flows.is_empty() && other_flows.is_empty() {
            return;
        }

        // Route ring flows using bipartite edge coloring
        if !ring_flows.is_empty() {
            let edges: Vec<(usize, usize, usize)> = ring_flows
                .iter()
                .enumerate()
                .map(|(idx, &(src, dst, _))| (src, dst, idx))
                .collect();

            let coloring = compute_edge_coloring(num_tors, edges);

            println!(
                "[FifoPerfect] Ring flows: {} flows, {} colors needed, {} spines available",
                ring_flows.len(),
                coloring.num_colors,
                num_spines
            );

            let final_coloring = if coloring.num_colors > num_spines {
                println!(
                    "[FifoPerfect] Collapsing {} colors into {} spines",
                    coloring.num_colors,
                    num_spines
                );
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

        // Route other flows by (src_host, dst_host)
        if !other_flows.is_empty() {
            let mut pair_to_spine: HashMap<(usize, usize), usize> = HashMap::new();
            let mut next_spine = 0;

            for &(src_host, dst_host, _) in other_flows.iter() {
                pair_to_spine.entry((src_host, dst_host)).or_insert_with(|| {
                    let spine = next_spine % num_spines;
                    next_spine += 1;
                    spine
                });
            }

            println!(
                "[FifoPerfect] Other flows: {} flows, {} unique host pairs, {} spines",
                other_flows.len(),
                pair_to_spine.len(),
                num_spines
            );

            for &(src_host, dst_host, (job_id, job_flow_idx)) in other_flows.iter() {
                let spine_idx = pair_to_spine[&(src_host, dst_host)];
                router.inject_spine_assignment(job_id, job_flow_idx, spine_idx);
            }
        }
    }

    // -------------------------------------------------------------------------
    // FIFO placement logic
    // -------------------------------------------------------------------------

    fn check_and_plan_migration(
        &self,
        ctx: &MLContext,
        topo: &SpineTree<SpinePerfectRouter>,
    ) -> Option<MigrationPlan> {
        // Check fragmentation using segment-based computation
        let frag = compute_segment_fragmentation(ctx, topo);

        println!(
            "[FifoPerfect] Fragmentation: max={}, threshold={}, fragmented_segments={}",
            frag.max_fragmentation,
            self.config.fragmentation_threshold,
            frag.fragmented_segments.len()
        );

        if frag.max_fragmentation <= self.config.fragmentation_threshold {
            return None;
        }

        println!("[FifoPerfect] Threshold exceeded! Computing FIFO placement...");

        let active_jobs = ctx.active_jobs.borrow();
        let current_placements = ctx.placements.borrow();
        let hosts_per_leaf = topo.hosts_per_leaf;
        let num_hosts = topo.num_leaves * hosts_per_leaf;
        let block_size = self.config.block_size.max(1);

        // Debug: check for inconsistencies between active_jobs and placements
        if DEBUG_FIFO {
            println!("[FifoPerfect] DEBUG: active_jobs={}, placements={}",
                active_jobs.len(), current_placements.len());
            
            // Check for orphaned jobs (in placements but not active)
            for &job_id in current_placements.keys() {
                if !active_jobs.contains_key(&job_id) {
                    println!("[FifoPerfect] WARNING: job {} in placements but not active", job_id);
                }
            }
            // Check for unplaced jobs (active but not in placements)
            for &job_id in active_jobs.keys() {
                if !current_placements.contains_key(&job_id) {
                    println!("[FifoPerfect] WARNING: job {} active but not in placements", job_id);
                }
            }
        }

        // Build host ownership map and verify no corruption (multiple jobs on same host)
        let mut host_to_job: Vec<Option<JobId>> = vec![None; num_hosts];
        for (&job_id, worker_hosts) in current_placements.iter() {
            for &host in worker_hosts.values() {
                if host < num_hosts {
                    if let Some(other_job) = host_to_job[host] {
                        panic!("[FifoPerfect] Host {} claimed by both job {} and {}", host, other_job, job_id);
                    }
                    host_to_job[host] = Some(job_id);
                }
            }
        }

        // Get all active job IDs sorted for deterministic linear placement
        let mut job_ids: Vec<JobId> = active_jobs.keys().copied().collect();
        job_ids.sort_unstable();

        // Compute new placement: assign workers linearly by job ID order
        let mut next_host = 0usize;
        let mut new_placements: HashMap<JobId, DHashMap<WorkerId, usize>> = HashMap::new();

        for &job_id in &job_ids {
            // Get worker IDs from current placements, sorted for deterministic placement
            let current_job_placements = match current_placements.get(&job_id) {
                Some(p) => p,
                None => continue,
            };

            let mut worker_ids: Vec<WorkerId> = current_job_placements.keys().copied().collect();
            worker_ids.sort_unstable();

            let num_workers = worker_ids.len();
            
            // Align to block boundary
            if block_size > 1 && next_host % block_size != 0 {
                next_host = ((next_host / block_size) + 1) * block_size;
            }

            // Check if we have enough space
            if next_host + num_workers > num_hosts {
                panic!(
                    "[FifoPerfect] FATAL: Not enough hosts for job {} ({} workers needed, {} available). \
                    This should be impossible if placements are consistent. \
                    Total workers so far: {}, num_hosts: {}",
                    job_id,
                    num_workers,
                    num_hosts - next_host,
                    next_host,
                    num_hosts
                );
            }

            // Assign workers to consecutive hosts
            let mut worker_to_host: DHashMap<WorkerId, usize> = DHashMap::default();
            for (i, &wid) in worker_ids.iter().enumerate() {
                worker_to_host.insert(wid, next_host + i);
            }

            new_placements.insert(job_id, worker_to_host);
            next_host += num_workers;

            // Align to block boundary for next job
            if block_size > 1 && next_host % block_size != 0 {
                next_host = ((next_host / block_size) + 1) * block_size;
            }
        }

        // Compare new placement with current placement to find migrations
        let mut migrations: Vec<JobMigration> = Vec::new();

        for (job_id, new_worker_hosts) in new_placements.iter() {
            let current_hosts = match current_placements.get(job_id) {
                Some(h) => h,
                None => continue,
            };

            // Find workers that need to move
            let mut worker_to_host: DHashMap<WorkerId, usize> = DHashMap::default();
            for (&wid, &new_host) in new_worker_hosts.iter() {
                let old_host = current_hosts.get(&wid).copied().unwrap_or(new_host);
                if old_host != new_host {
                    worker_to_host.insert(wid, new_host);
                }
            }

            if !worker_to_host.is_empty() {
                migrations.push(JobMigration {
                    job_id: *job_id,
                    worker_to_host,
                });
            }
        }

        if migrations.is_empty() {
            println!("[FifoPerfect] No migrations needed (already optimal)");
            return None;
        }

        let total_workers: usize = migrations.iter().map(|m| m.worker_to_host.len()).sum();
        println!(
            "[FifoPerfect] Migration plan: {} jobs, {} workers",
            migrations.len(),
            total_workers
        );

        // Validate migration plan: ensure no target host is occupied by a non-migrating job.
        // This catches bugs where the FIFO layout assigns a job to a host that's still
        // occupied by another job that isn't moving.
        let migrating_jobs: std::collections::HashSet<JobId> = migrations.iter()
            .map(|m| m.job_id)
            .collect();
        
        for job_mig in &migrations {
            for (&_worker_id, &target_host) in job_mig.worker_to_host.iter() {
                if let Some(current_occupant) = host_to_job[target_host] {
                    if current_occupant != job_mig.job_id && !migrating_jobs.contains(&current_occupant) {
                        // Debug info before panicking
                        if DEBUG_FIFO {
                            let in_active = active_jobs.contains_key(&current_occupant);
                            let in_placements = current_placements.contains_key(&current_occupant);
                            println!("[FifoPerfect] Invalid plan: job {} -> host {} occupied by job {} (active={}, placed={})",
                                job_mig.job_id, target_host, current_occupant, in_active, in_placements);
                        }
                        panic!("[FifoPerfect] Invalid migration: job {} targets host {} occupied by non-migrating job {}",
                            job_mig.job_id, target_host, current_occupant);
                    }
                }
            }
        }

        // Inject migration flow routes
        self.inject_migration_routes(&current_placements, &migrations, topo);

        Some(MigrationPlan { jobs: migrations })
    }

    /// Inject spine assignments for migration flows.
    fn inject_migration_routes(
        &self,
        current_placements: &DHashMap<JobId, DHashMap<WorkerId, usize>>,
        migrations: &[JobMigration],
        topo: &SpineTree<SpinePerfectRouter>,
    ) {
        let mut router = topo.router.borrow_mut();
        let hosts_per_leaf = topo.hosts_per_leaf;
        let num_spines = topo.num_spines;

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

                if src_tor != dst_tor {
                    let migration_flow_idx = MIGRATION_FLOW_IDX_BASE + worker_id;
                    migration_edges.push((src_tor, dst_tor, (job_id, migration_flow_idx)));
                }
            }
        }

        if migration_edges.is_empty() {
            return;
        }

        let edges: Vec<(usize, usize, usize)> = migration_edges
            .iter()
            .enumerate()
            .map(|(idx, &(src, dst, _))| (src, dst, idx))
            .collect();

        let coloring = compute_edge_coloring(topo.num_leaves, edges);

        let final_coloring = if coloring.num_colors > num_spines {
            collapse_colors(&coloring, num_spines)
        } else {
            coloring.edge_to_color.clone()
        };

        for (edge_idx, &(_, _, (job_id, migration_flow_idx))) in migration_edges.iter().enumerate() {
            if let Some(&spine_idx) = final_coloring.get(&edge_idx) {
                router.inject_spine_assignment(job_id, migration_flow_idx, spine_idx);
            }
        }
    }
}

impl<S, FS> SystemModule<SpineTree<SpinePerfectRouter>, S, FS> for FifoPerfect
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

        // 2. Check fragmentation and plan FIFO migration if needed
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
        
        println!("[FifoPerfect] Recomputing routes after migration");
        self.recompute_routes(ctx, topo);
    }
}
