#![allow(unused_imports)]
use network_sim::{
    network::alloc::{BandwidthAllocator},
    network::alloc::MaxMin,
    network::mltcp::MLTCP,
    network::flow::FlowId,
    network::routing::PathCell,
    simulator::{MLSimulator, ml_job::MLJobBuilder, WorkerEvent},
    simulator::{FifoScheduler, ImmediateFlowScheduler, NoopSystemModule, SystemModule, FlowScheduler, JobScheduler},
    simulator::ml_simulator::{MLContext, MLEventKind},
    simulator::ml_job::{MLJob, JobId},
    simulator::system::MigrationPlan,
    spine::{SpineTree, SpineTreeRouter, SpineTreeTopology, SpineEcmpRouter, SpineCruxRouter, SpineCassiniSystemModule, CruxSystemModule, SGLBRouter, SGLBSystemModule, SGLBConfig},
    flow_scheduler::{CassiniFlowScheduler},
    job_schedulers::{SimpleScheduler},
    schedulers::{ClusterScheduler, BlockScheduler, CassiniBlockScheduler, FifoBlockScheduler, RandomBlockScheduler, PreloadedBlockScheduler, DEFAULT_BLOCK_SIZE},
    utils::{load_random_s_jobs, DHashMap, fetch_job, job_loader::load_default_registry},
    network::topology::Topology,
    monkeytree::{MonkeyTreeSystem, MonkeyTreeEcmp, MonkeyTreeCrux, MonkeyTreePerfect, MonkeyTreeSGLB, FifoPerfect, SpinePerfectRouter, MonkeyTreeConfig},
    spine::PerfectRoutingSystem,
};

use std::env;
use std::fs;
use std::path::Path;
use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use network_sim::utils::load_homogenous_jobs;
use petgraph::graph::NodeIndex;

/// Parsed initial-state snapshot with iteration counts per job.
struct InitialStateWithIters {
    /// (job_id, num_workers, job_type, num_iterations)
    jobs: Vec<(usize, usize, String, usize)>,
    /// job_id -> sorted list of host indices
    placements: HashMap<usize, Vec<usize>>,
}

/// Parse a snapshot file produced by build_initial_state.py.
///
/// Format:
///   num_jobs
///   job_id: num_workers job_type num_iterations
///   ...
///   <blank line>
///   Placement
///   host_index: job_id
///   ...
fn parse_initial_state_full(path: &str) -> InitialStateWithIters {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read initial state file {}: {}", path, e));
    let mut lines = content.lines();

    let first = lines.next().expect("Empty initial state file").trim().to_string();
    if first == "Topology" {
        for line in &mut lines {
            if line.trim().is_empty() { break; }
        }
        // next line is num_jobs
        lines.next();
    }
    // else first was already the num_jobs line — consumed

    let mut jobs = Vec::new();
    for line in &mut lines {
        let t = line.trim();
        if t.is_empty() { break; }
        let mut parts = t.split(':');
        let job_id: usize = parts.next().and_then(|s| s.trim().parse().ok()).unwrap();
        let rhs = parts.next().unwrap().trim();
        let mut rhs_parts = rhs.split_whitespace();
        let nw: usize = rhs_parts.next().and_then(|s| s.parse().ok()).unwrap();
        let jtype = rhs_parts.next().unwrap_or("one_layer_moe").to_string();
        let iters: usize = rhs_parts.next().and_then(|s| s.parse().ok()).unwrap_or(100);
        jobs.push((job_id, nw, jtype, iters));
    }

    let hdr = lines.next().unwrap_or("").trim().to_string();
    assert!(hdr.eq_ignore_ascii_case("Placement"), "Expected 'Placement', got '{}'", hdr);

    let mut placements: HashMap<usize, Vec<usize>> = HashMap::new();
    for line in lines {
        let t = line.trim();
        if t.is_empty() { continue; }
        let mut parts = t.split(':');
        let host: usize = parts.next().and_then(|s| s.trim().parse().ok()).unwrap();
        let jid: usize = parts.next().and_then(|s| s.trim().parse().ok()).unwrap();
        placements.entry(jid).or_default().push(host);
    }
    for hosts in placements.values_mut() {
        hosts.sort_unstable();
    }

    // Establish a deterministic FIFO (job-id) ordering of preloaded jobs.
    // Non-migrating systems keep the original (fragmented) host assignments;
    // MonkeyTree systems re-pack placements contiguously in this order (see
    // fifo_compact_placement).
    jobs.sort_by_key(|&(job_id, _, _, _)| job_id);

    InitialStateWithIters { jobs, placements }
}

/// Re-pack preloaded placements so jobs are laid out contiguously from host 0 in
/// job-id (FIFO) order, discarding the seed's original (fragmented) assignments.
///
/// Job identities, sizes, types, and iteration counts are preserved -- only the
/// host assignments change. This gives MonkeyTree systems an already-compacted
/// starting state (the defragmented layout they would converge to), so they
/// don't pay a large one-time defragmentation ILP at simulation start.
fn fifo_compact_placement(state: &mut InitialStateWithIters) {
    let mut next_host = 0usize;
    let mut new_placements: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(job_id, num_workers, _, _) in &state.jobs {
        let hosts: Vec<usize> = (next_host..next_host + num_workers).collect();
        next_host += num_workers;
        new_placements.insert(job_id, hosts);
    }
    state.placements = new_placements;
}

/// Set up preloaded placements and add preloaded jobs to the simulator.
/// Returns the number of preloaded jobs.
fn add_preloaded_jobs<T, S, FS, M, A>(
    ml_sim: &mut MLSimulator<T, S, FS, M, A>,
    state: &InitialStateWithIters,
) -> usize
where
    T: Topology,
    S: JobScheduler,
    FS: FlowScheduler,
    M: SystemModule<T, S, FS>,
    A: BandwidthAllocator,
{
    for &(job_id, num_workers, ref job_type, num_iterations) in &state.jobs {
        let job = fetch_job(job_id, job_type, num_workers, num_iterations);
        ml_sim.add_job_arrival(0, job);
    }
    state.jobs.len()
}

/// Compute ideal job duration in microseconds (no network congestion).
/// Uses the job definition's collective operations to compute accurate network time.
fn compute_ideal_duration_us(kind: &str, num_workers: usize, num_iterations: usize, bandwidth_bps: f64) -> u64 {
    // Load the job registry
    let registry = load_default_registry().expect("Failed to load job registry");
    
    if let Some(job_def) = registry.get(kind) {
        job_def.ideal_duration_us(num_workers, num_iterations, bandwidth_bps)
    } else {
        // Unknown job type - return a large value so slowdown check doesn't trigger
        u64::MAX
    }
}

/// Load job definitions from trace file
/// Returns: job_id -> (arrival_time, kind, num_workers, num_iterations)
fn load_trace(trace_path: &str) -> DHashMap<usize, (usize, String, usize, usize)> {
    let content = fs::read_to_string(trace_path).expect("Failed to read trace file");
    let mut lines = content.lines();
    let _num_jobs = lines
        .next()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .expect("trace must start with number of jobs");

    let mut job_defs: DHashMap<usize, (usize, String, usize, usize)> = DHashMap::default();
    let mut job_id = 0;
    for line in &mut lines {
        let t = line.trim();
        if t.is_empty() { break; }
        let mut parts = t.split(' ');
        let arrival_time = parts
            .next()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .expect("Invalid arrival time in trace");
        let kind = parts
            .next()
            .map(|s| s.to_string())
            .expect("Invalid job type in trace");
        let num_workers = parts
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Invalid num_workers in trace");
        let num_iterations = parts
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Invalid num_iterations in trace (format: arrival_time job_type num_workers num_iterations)");

        job_defs.insert(job_id, (arrival_time, kind, num_workers, num_iterations));
        job_id += 1;
    }
    job_defs
}

/// Add jobs to simulator and run simulation
/// Returns Some((job_id, slowdown)) if terminated early due to max_slowdown, None otherwise
fn run_simulation<T, S, FS, M, A>(
    mut ml_sim: MLSimulator<T, S, FS, M, A>,
    job_defs: &DHashMap<usize, (usize, String, usize, usize)>,
    max_slowdown: Option<f64>,
    bandwidth_bps: f64,
    post_migration_delay_us: u64,
    preload_count: usize,
) -> Option<(usize, f64)>
where
    T: Topology,
    S: JobScheduler,
    FS: FlowScheduler,
    M: SystemModule<T, S, FS>,
    A: BandwidthAllocator,
{
    // Set post-migration delay if configured
    if post_migration_delay_us > 0 {
        ml_sim.set_post_migration_delay_us(post_migration_delay_us);
    }

    // Pre-compute ideal durations for each job
    let ideal_durations: DHashMap<usize, u64> = job_defs.iter()
        .map(|(&job_id, (_, kind, num_workers, num_iterations))| {
            (job_id, compute_ideal_duration_us(kind, *num_workers, *num_iterations, bandwidth_bps))
        })
        .collect();

    let total_jobs = job_defs.len() + preload_count;
    
    for (&job_id, (arrival_time, kind, num_workers, num_iterations)) in job_defs.iter() {
        let job = fetch_job(job_id, kind, *num_workers, *num_iterations);
        ml_sim.add_job_arrival(*arrival_time as u64, job);
    }

    let mut completed_jobs = 0;
    
    while let Some(event_kind) = ml_sim.advance_next_step() {
        // Check for job completions
        if event_kind == MLEventKind::JobComplete {
            completed_jobs += 1;
            
            // Check if all jobs have completed
            if completed_jobs >= total_jobs {
                println!("All {} jobs completed, terminating simulation", total_jobs);
                return None;
            }
            
            if let Some(threshold) = max_slowdown {
                // Check all completed jobs for slowdown violations
                for (job_id, job) in ml_sim.get_all_jobs().iter() {
                    if job.completion_time_us.is_some() {
                        let submit_time = job.submit_time_us;
                        let completion_time = job.completion_time_us.unwrap();
                        let total_runtime = completion_time.saturating_sub(submit_time);
                        
                        if let Some(&ideal) = ideal_durations.get(job_id) {
                            if ideal > 0 && ideal < u64::MAX {
                                let slowdown = total_runtime as f64 / ideal as f64;
                                if slowdown > threshold {
                                    println!("TERMINATED: Job {} exceeded max slowdown ({:.2} > {:.2})",
                                        job_id, slowdown, threshold);
                                    return Some((*job_id, slowdown));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

fn main() {
    let trace_path = env::args().nth(1).expect("Usage: golden_spine <TRACE_FILE> <SYSTEM> [MAX_JOBS] [BLOCK_SIZE] [MAX_SLOWDOWN] [NUM_SPINES] [NUM_GPUS] [THRESHOLD] [POST_MIGRATION_DELAY_US] [GPUS_PER_TOR] [INITIAL_SNAPSHOT] [SCHEDULER]");
    let system_arg = env::args().nth(2).expect("Usage: golden_spine <TRACE_FILE> <SYSTEM> [MAX_JOBS] [BLOCK_SIZE] [MAX_SLOWDOWN] [NUM_SPINES] [NUM_GPUS] [THRESHOLD] [POST_MIGRATION_DELAY_US] [GPUS_PER_TOR] [INITIAL_SNAPSHOT] [SCHEDULER]");
    // A `_nocompact` suffix on the system name disables the MonkeyTree FIFO
    // re-packing of preloaded jobs. With it, MonkeyTree inherits the seed's
    // fragmented placement and must migrate its way out, rather than starting
    // from an already-compacted layout. The suffix is stripped so the rest of
    // the binary treats it as the base system (e.g. monkeytree_perfect).
    let fifo_compact = !system_arg.ends_with("_nocompact");
    let system = system_arg.trim_end_matches("_nocompact").to_string();
    let max_jobs: usize = env::args().nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0); // 0 means load all jobs
    let block_size: usize = env::args().nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BLOCK_SIZE); // Default: 8 GPUs per server
    let max_slowdown: Option<f64> = env::args().nth(5)
        .and_then(|s| s.parse().ok())
        .filter(|&v| v > 0.0); // 0 or negative means no limit
    let num_spines: usize = env::args().nth(6)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16); // Default: 16 spine switches
    let num_gpus: usize = env::args().nth(7)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024); // Default: 512 GPUs (64 servers, 8 ToRs)
    let threshold_arg: usize = env::args().nth(8)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0); // 0 means use num_spines as default
    let post_migration_delay_us: u64 = env::args().nth(9)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0); // 0 means no delay; set to e.g. 500000 for 500ms (~10 iterations)
    let gpus_per_tor: usize = env::args().nth(10)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64); // Default: 64 GPUs per ToR (8 servers * 8 GPUs)
    let initial_snapshot_path: Option<String> = env::args().nth(11)
        .filter(|s| !s.is_empty() && s != "0" && s != "none");
    let scheduler_type: String = env::args().nth(12)
        .unwrap_or_else(|| "block".to_string()); // block (default best-fit), fifo, random

    // GPU cluster topology
    let num_servers = num_gpus / block_size;
    let num_leaves = num_gpus / gpus_per_tor;
    let hosts_per_leaf = gpus_per_tor;  // "hosts" in the topology = GPUs
    // Threshold defaults to num_spines if not specified (0)
    let threshold: usize = if threshold_arg > 0 { threshold_arg } else { num_spines };
    
    // Validate topology
    assert!(num_gpus % gpus_per_tor == 0, 
        "num_gpus ({}) must be divisible by gpus_per_tor ({})", num_gpus, gpus_per_tor);

    let bandwidth_bps = 400.0e9; // 400 Gbps - must match topology

    // Print topology header for snapshot extraction
    println!("Topology {} {} {}", hosts_per_leaf, num_leaves, num_spines);
    println!("Num GPUs: {} ({} servers, {} ToRs, {} GPUs/ToR)", num_gpus, num_servers, num_leaves, gpus_per_tor);
    println!("Block size: {} (GPUs per server)", block_size);
    println!("Num spines: {}", num_spines);
    println!("System: {}", system);
    if let Some(ms) = max_slowdown {
        println!("Max slowdown: {:.2} (will terminate if exceeded)", ms);
    }
    if system.starts_with("monkeytree") || system == "fifo_perfect" {
        println!("Fragmentation threshold: {}", threshold);
    }
    if post_migration_delay_us > 0 {
        println!("Post-migration delay: {}us ({:.1}ms)", post_migration_delay_us, post_migration_delay_us as f64 / 1000.0);
    }
    if scheduler_type != "block" {
        println!("Scheduler: {}", scheduler_type);
    }

    // Load trace first (common to all systems)
    let mut job_defs = load_trace(&trace_path);
    
    // Limit jobs if max_jobs is specified and > 0
    if max_jobs > 0 && job_defs.len() > max_jobs {
        println!("Limiting to {} jobs (out of {} in trace)", max_jobs, job_defs.len());
        let keys_to_remove: Vec<_> = job_defs.keys()
            .filter(|&&id| id >= max_jobs)
            .copied()
            .collect();
        for key in keys_to_remove {
            job_defs.remove(&key);
        }
    }
    println!("Loaded {} jobs from trace", job_defs.len());

    // Parse initial state if provided
    let initial_state = initial_snapshot_path.as_ref().map(|p| {
        println!("Loading initial state from: {}", p);
        let mut state = parse_initial_state_full(p);
        println!("  Preloaded jobs: {}", state.jobs.len());
        let occupied: usize = state.placements.values().map(|v| v.len()).sum();
        println!("  Occupied hosts: {} / {} ({:.1}%)", occupied, num_gpus, 100.0 * occupied as f64 / num_gpus as f64);
        // MonkeyTree systems start from an already-compacted layout: re-pack the
        // preloaded jobs contiguously in FIFO (job-id) order, overriding the
        // seed's fragmented host assignments. Non-migrating systems keep the
        // original fragmented placement so they experience the real fragmentation.
        if fifo_compact && system.starts_with("monkeytree") {
            fifo_compact_placement(&mut state);
            println!("  [MonkeyTree] re-packed seed into FIFO-contiguous placement (job-id order, from host 0)");
        } else if !fifo_compact && system.starts_with("monkeytree") {
            println!("  [MonkeyTree] FIFO compaction DISABLED (_nocompact): inheriting fragmented seed placement");
        }
        state
    });

    // Shift trace job IDs above preloaded IDs to avoid collisions
    if let Some(ref state) = initial_state {
        let max_preload_id = state.jobs.iter().map(|(id, _, _, _)| *id).max().unwrap_or(0);
        let id_offset = max_preload_id + 1;
        let old_defs: Vec<_> = job_defs.iter().map(|(&k, v)| (k, v.clone())).collect();
        job_defs.clear();
        for (old_id, val) in old_defs {
            job_defs.insert(old_id + id_offset, val);
        }
    }

    /// Helper: create a PreloadedBlockScheduler and populate it from initial state.
    fn make_scheduler(hosts_per_leaf: usize, block_size: usize, state: &Option<InitialStateWithIters>) -> PreloadedBlockScheduler {
        let mut scheduler = PreloadedBlockScheduler::new(hosts_per_leaf, block_size);
        if let Some(ref s) = state {
            for &(job_id, _, _, _) in &s.jobs {
                if let Some(hosts) = s.placements.get(&job_id) {
                    scheduler.set_preloaded_placement(job_id, hosts.clone());
                }
            }
        }
        scheduler
    }

    match system.as_str() {
        "ecmp" => {
            match scheduler_type.as_str() {
                // Naive baseline: jobs land on uniformly-random free hosts
                // (no rack/locality awareness at all), still routed with ECMP
                // and with no MonkeyTree migration.
                "random" => {
                    let router = SpineEcmpRouter::new(42);
                    let spine_tree = SpineTree::new(hosts_per_leaf, num_leaves, num_spines, 400.0e9, router);
                    let mut scheduler = RandomBlockScheduler::new(block_size, 42);
                    if let Some(ref s) = initial_state {
                        for &(job_id, _, _, _) in &s.jobs {
                            if let Some(hosts) = s.placements.get(&job_id) {
                                scheduler.set_preloaded_placement(job_id, hosts.clone());
                            }
                        }
                    }
                    let flow_scheduler = ImmediateFlowScheduler::new();
                    let system_module = NoopSystemModule::default();
                    let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
                    let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
                    run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
                }
                _ => {
                    let router = SpineEcmpRouter::new(42);
                    let spine_tree = SpineTree::new(
                        hosts_per_leaf,
                        num_leaves,
                        num_spines,
                        400.0e9,
                        router,
                    );
                    let scheduler = make_scheduler(hosts_per_leaf, block_size, &initial_state);
                    let flow_scheduler = ImmediateFlowScheduler::new();
                    let system_module = NoopSystemModule::default();
                    let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
                    let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
                    run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
                }
            }
        }
        "crux" => {
            let router = SpineCruxRouter::new();
            let spine_tree = SpineTree::new(hosts_per_leaf, num_leaves, num_spines, 400.0e9, router);
            let scheduler = make_scheduler(hosts_per_leaf, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let system_module = CruxSystemModule::default();
            let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "cassini" => {
            // Cassini: network-aware job scheduling with time-shifted flows
            // Uses CassiniBlockScheduler to generate multiple placement candidates
            // SpineCassiniSystemModule evaluates candidates for network compatibility
            let router = SpineEcmpRouter::new(42);
            let spine_tree = SpineTree::new(
                hosts_per_leaf,
                num_leaves,
                num_spines,
                400.0e9,
                router,
            );
            let scheduler = CassiniBlockScheduler::new(hosts_per_leaf, block_size);
            let flow_scheduler = CassiniFlowScheduler::new();
            let system_module = SpineCassiniSystemModule::new(); // No static_placement - evaluate candidates
            let ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, 0);
        }
        "cassini_lite" => {
            // Cassini Lite: uses CassiniBlockScheduler's multi-candidate generation
            // and simple contention-based scoring, but NO geometric abstraction,
            // NO affinity graph optimization, and NO time-shifting.
            // This isolates the benefit of considering multiple placement candidates.
            let router = SpineEcmpRouter::new(42);
            let spine_tree = SpineTree::new(
                hosts_per_leaf,
                num_leaves,
                num_spines,
                400.0e9,
                router,
            );
            let scheduler = CassiniBlockScheduler::new(hosts_per_leaf, block_size);
            let flow_scheduler = ImmediateFlowScheduler::new(); // No time-shifting
            let system_module = NoopSystemModule::default();    // No Cassini evaluation
            let ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, 0);
        }
        "monkeytree" | "monkeytree_ecmp" => {
            let router = SpineEcmpRouter::new(42);
            let spine_tree = SpineTree::new(hosts_per_leaf, num_leaves, num_spines, 400.0e9, router);
            let scheduler = make_scheduler(hosts_per_leaf, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let config = MonkeyTreeConfig { fragmentation_threshold: threshold, block_size };
            let system_module = MonkeyTreeSystem::<SpineEcmpRouter>::new(config);
            let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "monkeytree_crux" => {
            let router = SpineCruxRouter::new();
            let spine_tree = SpineTree::new(hosts_per_leaf, num_leaves, num_spines, 400.0e9, router);
            let scheduler = make_scheduler(hosts_per_leaf, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let config = MonkeyTreeConfig { fragmentation_threshold: threshold, block_size };
            let system_module = MonkeyTreeCrux::new(config);
            let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "monkeytree_perfect" => {
            match scheduler_type.as_str() {
                "fifo" => {
                    let router = SpinePerfectRouter::new();
                    let spine_tree = SpineTree::new(hosts_per_leaf, num_leaves, num_spines, 400.0e9, router);
                    let mut scheduler = FifoBlockScheduler::new(block_size);
                    if let Some(ref s) = initial_state {
                        for &(job_id, _, _, _) in &s.jobs {
                            if let Some(hosts) = s.placements.get(&job_id) {
                                scheduler.set_preloaded_placement(job_id, hosts.clone());
                            }
                        }
                    }
                    let flow_scheduler = ImmediateFlowScheduler::new();
                    let config = MonkeyTreeConfig { fragmentation_threshold: threshold, block_size };
                    let system_module = MonkeyTreePerfect::new(config);
                    let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
                    let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
                    run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
                }
                "random" => {
                    let router = SpinePerfectRouter::new();
                    let spine_tree = SpineTree::new(hosts_per_leaf, num_leaves, num_spines, 400.0e9, router);
                    let mut scheduler = RandomBlockScheduler::new(block_size, 42);
                    if let Some(ref s) = initial_state {
                        for &(job_id, _, _, _) in &s.jobs {
                            if let Some(hosts) = s.placements.get(&job_id) {
                                scheduler.set_preloaded_placement(job_id, hosts.clone());
                            }
                        }
                    }
                    let flow_scheduler = ImmediateFlowScheduler::new();
                    let config = MonkeyTreeConfig { fragmentation_threshold: threshold, block_size };
                    let system_module = MonkeyTreePerfect::new(config);
                    let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
                    let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
                    run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
                }
                _ => {
                    // "block" or default: existing best-fit block scheduler
                    let router = SpinePerfectRouter::new();
                    let spine_tree = SpineTree::new(hosts_per_leaf, num_leaves, num_spines, 400.0e9, router);
                    let scheduler = make_scheduler(hosts_per_leaf, block_size, &initial_state);
                    let flow_scheduler = ImmediateFlowScheduler::new();
                    let config = MonkeyTreeConfig { fragmentation_threshold: threshold, block_size };
                    let system_module = MonkeyTreePerfect::new(config);
                    let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
                    let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
                    run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
                }
            }
        }
        "fifo_perfect" => {
            let router = SpinePerfectRouter::new();
            let spine_tree = SpineTree::new(hosts_per_leaf, num_leaves, num_spines, 400.0e9, router);
            let scheduler = make_scheduler(hosts_per_leaf, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let config = MonkeyTreeConfig { fragmentation_threshold: threshold, block_size };
            let system_module = FifoPerfect::new(config);
            let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "perfect" => {
            let router = SpinePerfectRouter::new();
            let spine_tree = SpineTree::new(hosts_per_leaf, num_leaves, num_spines, 400.0e9, router);
            let scheduler = make_scheduler(hosts_per_leaf, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let system_module = PerfectRoutingSystem::new();
            let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "sglb" => {
            let k = num_spines / 2;
            let config = SGLBConfig::with_remap_interval(k, 10_000);
            let router = SGLBRouter::new(config.clone());
            let spine_tree = SpineTree::new(hosts_per_leaf, num_leaves, num_spines, 400.0e9, router);
            let scheduler = make_scheduler(hosts_per_leaf, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let system_module = SGLBSystemModule::new(config);
            println!("SGLB K={}, remap_interval=10ms", k);
            let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "monkeytree_sglb" => {
            let k = num_spines / 2;
            let sglb_config = SGLBConfig::with_remap_interval(k, 10_000);
            let router = SGLBRouter::new(sglb_config.clone());
            let spine_tree = SpineTree::new(hosts_per_leaf, num_leaves, num_spines, 400.0e9, router);
            let scheduler = make_scheduler(hosts_per_leaf, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let mt_config = MonkeyTreeConfig { fragmentation_threshold: threshold, block_size };
            let system_module = MonkeyTreeSGLB::new(mt_config, sglb_config);
            let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "mltcp" => {
            let router = SpineEcmpRouter::new(42);
            let spine_tree = SpineTree::new(hosts_per_leaf, num_leaves, num_spines, 400.0e9, router);
            let scheduler = make_scheduler(hosts_per_leaf, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let system_module = NoopSystemModule::default();
            let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MLTCP);
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "monkeytree_mltcp" => {
            let router = SpineEcmpRouter::new(42);
            let spine_tree = SpineTree::new(hosts_per_leaf, num_leaves, num_spines, 400.0e9, router);
            let scheduler = make_scheduler(hosts_per_leaf, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let config = MonkeyTreeConfig { fragmentation_threshold: threshold, block_size };
            let system_module = MonkeyTreeSystem::<SpineEcmpRouter>::new(config);
            let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MLTCP);
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        _ => panic!("Unknown system type: {}. Valid options: ecmp, crux, cassini, cassini_lite, monkeytree, monkeytree_ecmp, monkeytree_crux, monkeytree_perfect, monkeytree_sglb, fifo_perfect, perfect, sglb, mltcp, monkeytree_mltcp", system),
    }
}
