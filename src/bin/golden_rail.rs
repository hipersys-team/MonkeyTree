#![allow(unused_imports)]
use network_sim::{
    network::alloc::MaxMin,
    network::mltcp::MLTCP,
    network::topology::Topology,
    simulator::{MLSimulator, ml_job::MLJobBuilder, WorkerEvent},
    simulator::{FifoScheduler, ImmediateFlowScheduler, NoopSystemModule, SystemModule, FlowScheduler, JobScheduler},
    simulator::ml_simulator::{MLContext, MLEventKind},
    simulator::ml_job::{MLJob, JobId},
    simulator::system::MigrationPlan,
    rail::{RailTree, RailTopology, RailTreeRouter, RailEcmpRouter, RailCruxRouter, RailCruxSystemModule, RailPerfectRouter, RailPerfectRoutingSystem, RailSGLBRouter, RailSGLBSystem},
    spine::SGLBConfig,
    schedulers::{PreloadedRailBlockScheduler, DEFAULT_BLOCK_SIZE},
    utils::{DHashMap, fetch_job, job_loader::load_default_registry},
    monkeytree::{MonkeyTreeConfig, RailMonkeyTreeSystem, RailMonkeyTreeCrux, RailMonkeyTreePerfect},
};

use std::env;
use std::fs;
use std::collections::HashMap;
use petgraph::graph::NodeIndex;

/// Parsed initial-state snapshot with iteration counts per job.
struct InitialStateWithIters {
    jobs: Vec<(usize, usize, String, usize)>,
    placements: HashMap<usize, Vec<usize>>,
}

fn parse_initial_state_full(path: &str) -> InitialStateWithIters {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read initial state file {}: {}", path, e));
    let mut lines = content.lines();

    let first = lines.next().expect("Empty initial state file").trim().to_string();
    if first == "Topology" {
        for line in &mut lines {
            if line.trim().is_empty() { break; }
        }
        lines.next();
    }

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

    InitialStateWithIters { jobs, placements }
}

fn add_preloaded_jobs<T, S, FS, M, A>(
    ml_sim: &mut MLSimulator<T, S, FS, M, A>,
    state: &InitialStateWithIters,
) -> usize
where
    T: Topology,
    S: JobScheduler,
    FS: FlowScheduler,
    M: SystemModule<T, S, FS>,
    A: network_sim::network::alloc::BandwidthAllocator,
{
    for &(job_id, num_workers, ref job_type, num_iterations) in &state.jobs {
        let job = fetch_job(job_id, job_type, num_workers, num_iterations);
        ml_sim.add_job_arrival(0, job);
    }
    state.jobs.len()
}

fn compute_ideal_duration_us(kind: &str, num_workers: usize, num_iterations: usize, bandwidth_bps: f64) -> u64 {
    let registry = load_default_registry().expect("Failed to load job registry");
    if let Some(job_def) = registry.get(kind) {
        job_def.ideal_duration_us(num_workers, num_iterations, bandwidth_bps)
    } else {
        u64::MAX
    }
}

fn load_trace(trace_path: &str) -> DHashMap<usize, (usize, String, usize, usize)> {
    let content = fs::read_to_string(trace_path).expect("Failed to read trace file");
    let mut lines = content.lines();
    let _num_jobs = lines.next().and_then(|s| s.trim().parse::<usize>().ok()).expect("trace must start with number of jobs");

    let mut job_defs: DHashMap<usize, (usize, String, usize, usize)> = DHashMap::default();
    let mut job_id = 0;
    for line in &mut lines {
        let t = line.trim();
        if t.is_empty() { break; }
        let mut parts = t.split(' ');
        let arrival_time = parts.next().and_then(|s| s.trim().parse::<usize>().ok()).expect("Invalid arrival time");
        let kind = parts.next().map(|s| s.to_string()).expect("Invalid job type");
        let num_workers = parts.next().and_then(|s| s.parse::<usize>().ok()).expect("Invalid num_workers");
        let num_iterations = parts.next().and_then(|s| s.parse::<usize>().ok()).expect("Invalid num_iterations");
        job_defs.insert(job_id, (arrival_time, kind, num_workers, num_iterations));
        job_id += 1;
    }
    job_defs
}

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
    A: network_sim::network::alloc::BandwidthAllocator,
{
    if post_migration_delay_us > 0 {
        ml_sim.set_post_migration_delay_us(post_migration_delay_us);
    }

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
        if event_kind == MLEventKind::JobComplete {
            completed_jobs += 1;
            if completed_jobs >= total_jobs {
                println!("All {} jobs completed, terminating simulation", total_jobs);
                return None;
            }
            if let Some(threshold) = max_slowdown {
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
    let usage = "Usage: golden_rail <TRACE_FILE> <SYSTEM> [MAX_JOBS] [BLOCK_SIZE] [MAX_SLOWDOWN] [NUM_SPINES] [NUM_GPUS] [THRESHOLD] [POST_MIGRATION_DELAY_US] [BLOCKS_PER_POD] [NUM_PODS] [GPUS_PER_BLOCK]";
    let trace_path = env::args().nth(1).expect(usage);
    let system = env::args().nth(2).expect(usage);
    let max_jobs: usize = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let block_size: usize = env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_BLOCK_SIZE);
    let max_slowdown: Option<f64> = env::args().nth(5).and_then(|s| s.parse().ok()).filter(|&v| v > 0.0);
    let num_spines: usize = env::args().nth(6).and_then(|s| s.parse().ok()).unwrap_or(16);
    let num_gpus: usize = env::args().nth(7).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let threshold_arg: usize = env::args().nth(8).and_then(|s| s.parse().ok()).unwrap_or(0);
    let post_migration_delay_us: u64 = env::args().nth(9).and_then(|s| s.parse().ok()).unwrap_or(0);
    let blocks_per_pod: usize = env::args().nth(10).and_then(|s| s.parse().ok()).unwrap_or(8);
    let num_pods: usize = env::args().nth(11).and_then(|s| s.parse().ok()).unwrap_or(0);
    let gpus_per_block: usize = env::args().nth(12).and_then(|s| s.parse().ok()).unwrap_or(8);
    let initial_snapshot_path: Option<String> = env::args().nth(13)
        .filter(|s| !s.is_empty() && s != "0" && s != "none");

    // Derive topology parameters
    let actual_num_pods = if num_pods > 0 {
        num_pods
    } else {
        num_gpus / (blocks_per_pod * gpus_per_block)
    };

    let total_gpus = actual_num_pods * blocks_per_pod * gpus_per_block;
    assert_eq!(total_gpus, num_gpus,
        "num_gpus ({}) != num_pods ({}) * blocks_per_pod ({}) * gpus_per_block ({})",
        num_gpus, actual_num_pods, blocks_per_pod, gpus_per_block);

    let threshold: usize = if threshold_arg > 0 { threshold_arg } else { num_spines };

    let host_bw = 400.0e9;
    let rail_bw = 400.0e9;
    let spine_bw = 400.0e9;
    let bandwidth_bps = rail_bw;

    println!("RailTopology {} pods, {} blocks/pod, {} GPUs/block, {} spines",
        actual_num_pods, blocks_per_pod, gpus_per_block, num_spines);
    println!("Total GPUs: {} ({} servers)", total_gpus, total_gpus / gpus_per_block);
    println!("Block size: {} (GPUs per server)", block_size);
    println!("System: {}", system);
    if let Some(ms) = max_slowdown {
        println!("Max slowdown: {:.2}", ms);
    }
    if system.starts_with("monkeytree") {
        println!("Fragmentation threshold: {}", threshold);
    }
    if post_migration_delay_us > 0 {
        println!("Post-migration delay: {}us ({:.1}ms)", post_migration_delay_us, post_migration_delay_us as f64 / 1000.0);
    }

    let mut job_defs = load_trace(&trace_path);

    if max_jobs > 0 && job_defs.len() > max_jobs {
        println!("Limiting to {} jobs (out of {} in trace)", max_jobs, job_defs.len());
        let keys_to_remove: Vec<_> = job_defs.keys().filter(|&&id| id >= max_jobs).copied().collect();
        for key in keys_to_remove { job_defs.remove(&key); }
    }
    println!("Loaded {} jobs from trace", job_defs.len());

    let initial_state = initial_snapshot_path.as_ref().map(|p| {
        println!("Loading initial state from: {}", p);
        let state = parse_initial_state_full(p);
        println!("  Preloaded jobs: {}", state.jobs.len());
        state
    });

    if let Some(ref state) = initial_state {
        let max_preload_id = state.jobs.iter().map(|(id, _, _, _)| *id).max().unwrap_or(0);
        let id_offset = max_preload_id + 1;
        let old_defs: Vec<_> = job_defs.iter().map(|(&k, v)| (k, v.clone())).collect();
        job_defs.clear();
        for (old_id, val) in old_defs { job_defs.insert(old_id + id_offset, val); }
    }

    fn make_scheduler(blocks_per_pod: usize, block_size: usize, state: &Option<InitialStateWithIters>) -> PreloadedRailBlockScheduler {
        let mut scheduler = PreloadedRailBlockScheduler::new(blocks_per_pod, block_size);
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
            let router = RailEcmpRouter::new(42);
            let topo = RailTree::new(gpus_per_block, blocks_per_pod, actual_num_pods, num_spines, host_bw, rail_bw, spine_bw, router);
            let scheduler = make_scheduler(blocks_per_pod, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let system_module = NoopSystemModule::default();
            let mut ml_sim = MLSimulator::new(topo, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "crux" => {
            let router = RailCruxRouter::new();
            let topo = RailTree::new(gpus_per_block, blocks_per_pod, actual_num_pods, num_spines, host_bw, rail_bw, spine_bw, router);
            let scheduler = make_scheduler(blocks_per_pod, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let system_module = RailCruxSystemModule::default();
            let mut ml_sim = MLSimulator::new(topo, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "perfect" => {
            let router = RailPerfectRouter::new();
            let topo = RailTree::new(gpus_per_block, blocks_per_pod, actual_num_pods, num_spines, host_bw, rail_bw, spine_bw, router);
            let scheduler = make_scheduler(blocks_per_pod, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let system_module = RailPerfectRoutingSystem::new();
            let mut ml_sim = MLSimulator::new(topo, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "sglb" => {
            let k = (num_spines / 2).max(1);
            let config = SGLBConfig::with_remap_interval(k, 10_000);
            let router = RailSGLBRouter::new(config.clone());
            let topo = RailTree::new(gpus_per_block, blocks_per_pod, actual_num_pods, num_spines, host_bw, rail_bw, spine_bw, router);
            let scheduler = make_scheduler(blocks_per_pod, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let system_module = RailSGLBSystem::new(config);
            println!("Rail SGLB K={}, remap_interval=10ms", k);
            let mut ml_sim = MLSimulator::new(topo, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "monkeytree" | "monkeytree_ecmp" => {
            let router = RailEcmpRouter::new(42);
            let topo = RailTree::new(gpus_per_block, blocks_per_pod, actual_num_pods, num_spines, host_bw, rail_bw, spine_bw, router);
            let scheduler = make_scheduler(blocks_per_pod, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let config = MonkeyTreeConfig { fragmentation_threshold: threshold, block_size };
            let system_module = RailMonkeyTreeSystem::<RailEcmpRouter>::new(config);
            let mut ml_sim = MLSimulator::new(topo, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "monkeytree_crux" => {
            let router = RailCruxRouter::new();
            let topo = RailTree::new(gpus_per_block, blocks_per_pod, actual_num_pods, num_spines, host_bw, rail_bw, spine_bw, router);
            let scheduler = make_scheduler(blocks_per_pod, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let config = MonkeyTreeConfig { fragmentation_threshold: threshold, block_size };
            let system_module = RailMonkeyTreeCrux::new(config);
            let mut ml_sim = MLSimulator::new(topo, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        "monkeytree_perfect" => {
            let router = RailPerfectRouter::new();
            let topo = RailTree::new(gpus_per_block, blocks_per_pod, actual_num_pods, num_spines, host_bw, rail_bw, spine_bw, router);
            let scheduler = make_scheduler(blocks_per_pod, block_size, &initial_state);
            let flow_scheduler = ImmediateFlowScheduler::new();
            let config = MonkeyTreeConfig { fragmentation_threshold: threshold, block_size };
            let system_module = RailMonkeyTreePerfect::new(config);
            let mut ml_sim = MLSimulator::new(topo, scheduler, flow_scheduler, system_module, MaxMin::new());
            let pc = initial_state.as_ref().map(|s| add_preloaded_jobs(&mut ml_sim, s)).unwrap_or(0);
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us, pc);
        }
        _ => panic!("Unknown system: {}. Valid: ecmp, crux, perfect, sglb, monkeytree, monkeytree_ecmp, monkeytree_crux, monkeytree_perfect", system),
    }
}
