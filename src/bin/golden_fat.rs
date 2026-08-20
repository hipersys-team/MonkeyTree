#![allow(unused_imports)]
use network_sim::{
    network::alloc::MaxMin,
    network::mltcp::MLTCP,
    network::topology::{Topology, FatTree, FatTreeTopology},
    network::routing::FatTreeRouter,
    simulator::{MLSimulator, WorkerEvent},
    simulator::{FifoScheduler, ImmediateFlowScheduler, NoopSystemModule, SystemModule, FlowScheduler, JobScheduler},
    simulator::ml_simulator::{MLContext, MLEventKind},
    simulator::ml_job::{MLJob, JobId},
    simulator::system::MigrationPlan,
    routing::EcmpRouter,
    fat::{FatTreePerfectRouter, FatTreePerfectSystem, FatTreeCruxRouter, FatTreeCruxSystem, FatTreeMonkeyTreePerfect, FatTreeMonkeyTree3, FatTreeSGLBRouter, FatTreeSGLBSystem},
    spine::SGLBConfig,
    monkeytree::MonkeyTreeConfig,
    schedulers::{BlockScheduler, FatTreeBlockScheduler, DEFAULT_BLOCK_SIZE},
    utils::{DHashMap, fetch_job, job_loader::load_default_registry},
};

use std::env;
use std::fs;

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
    let _num_jobs = lines.next().and_then(|s| s.trim().parse::<usize>().ok())
        .expect("trace must start with number of jobs");

    let mut job_defs: DHashMap<usize, (usize, String, usize, usize)> = DHashMap::default();
    let mut job_id = 0;
    for line in &mut lines {
        let t = line.trim();
        if t.is_empty() { break; }
        let mut parts = t.split(' ');
        let arrival_time = parts.next().and_then(|s| s.parse().ok()).expect("Invalid arrival time");
        let kind = parts.next().map(|s| s.to_string()).expect("Invalid job type");
        let num_workers = parts.next().and_then(|s| s.parse().ok()).expect("Invalid num_workers");
        let num_iterations = parts.next().and_then(|s| s.parse().ok()).expect("Invalid num_iterations");
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
        .map(|(&id, (_, kind, nw, ni))| (id, compute_ideal_duration_us(kind, *nw, *ni, bandwidth_bps)))
        .collect();

    let total_jobs = job_defs.len();
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
                for (jid, job) in ml_sim.get_all_jobs().iter() {
                    if let Some(ct) = job.completion_time_us {
                        let runtime = ct.saturating_sub(job.submit_time_us);
                        if let Some(&ideal) = ideal_durations.get(jid) {
                            if ideal > 0 && ideal < u64::MAX {
                                let sd = runtime as f64 / ideal as f64;
                                if sd > threshold {
                                    println!("TERMINATED: Job {} exceeded max slowdown ({:.2} > {:.2})", jid, sd, threshold);
                                    return Some((*jid, sd));
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
    let args: Vec<String> = env::args().collect();
    let trace_path = args.get(1).expect(
        "Usage: golden_fat <TRACE> <SYSTEM> [MAX_JOBS] [BLOCK_SIZE] [MAX_SLOWDOWN] \
         [NUM_PODS] [NUM_GPUS] [THRESHOLD] [POST_MIGRATION_DELAY_US] \
         [GPUS_PER_TOR] [TORS_PER_POD] [AGGS_PER_POD] [NUM_CORES]");
    let system = args.get(2).expect("Missing SYSTEM argument");
    let max_jobs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let block_size: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_BLOCK_SIZE);
    let max_slowdown: Option<f64> = args.get(5).and_then(|s| s.parse().ok()).filter(|&v| v > 0.0);
    let num_pods: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(4);
    let num_gpus: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let threshold: usize = args.get(8).and_then(|s| s.parse().ok()).unwrap_or(0);
    let post_migration_delay_us: u64 = args.get(9).and_then(|s| s.parse().ok()).unwrap_or(0);
    let gpus_per_tor: usize = args.get(10).and_then(|s| s.parse().ok()).unwrap_or(64);
    let tors_per_pod: usize = args.get(11).and_then(|s| s.parse().ok()).unwrap_or(4);
    let aggs_per_pod: usize = args.get(12).and_then(|s| s.parse().ok()).unwrap_or(16);
    let num_cores: usize = args.get(13).and_then(|s| s.parse().ok()).unwrap_or(16);

    let bandwidth_bps = 400.0e9;

    assert_eq!(num_gpus, num_pods * tors_per_pod * gpus_per_tor,
        "num_gpus ({}) != num_pods ({}) * tors_per_pod ({}) * gpus_per_tor ({})",
        num_gpus, num_pods, tors_per_pod, gpus_per_tor);

    println!("FatTree Topology: {} pods, {} tors/pod, {} gpus/tor, {} aggs/pod, {} cores",
        num_pods, tors_per_pod, gpus_per_tor, aggs_per_pod, num_cores);
    println!("Total GPUs: {}, Block size: {}", num_gpus, block_size);
    println!("System: {}", system);
    if let Some(ms) = max_slowdown {
        println!("Max slowdown: {:.2}", ms);
    }
    if system.starts_with("monkeytree") {
        let t = if threshold > 0 { threshold } else { aggs_per_pod };
        println!("Fragmentation threshold: {}", t);
    }

    let mut job_defs = load_trace(trace_path);
    if max_jobs > 0 && job_defs.len() > max_jobs {
        println!("Limiting to {} jobs (out of {} in trace)", max_jobs, job_defs.len());
        let to_remove: Vec<_> = job_defs.keys().filter(|&&id| id >= max_jobs).copied().collect();
        for k in to_remove { job_defs.remove(&k); }
    }
    println!("Loaded {} jobs from trace", job_defs.len());

    match system.as_str() {
        "ecmp" => {
            let router = EcmpRouter::new(42);
            let topo = FatTree::new(gpus_per_tor, tors_per_pod, aggs_per_pod, num_cores, num_pods, bandwidth_bps, router);
            let scheduler = BlockScheduler::new(gpus_per_tor, block_size);
            let ml_sim = MLSimulator::new(topo, scheduler, ImmediateFlowScheduler::new(), NoopSystemModule::default(), MaxMin::new());
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us);
        }
        "perfect" => {
            let router = FatTreePerfectRouter::new();
            let topo = FatTree::new(gpus_per_tor, tors_per_pod, aggs_per_pod, num_cores, num_pods, bandwidth_bps, router);
            let scheduler = BlockScheduler::new(gpus_per_tor, block_size);
            let system_module = FatTreePerfectSystem::new();
            let ml_sim = MLSimulator::new(topo, scheduler, ImmediateFlowScheduler::new(), system_module, MaxMin::new());
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us);
        }
        "crux" => {
            let router = FatTreeCruxRouter::new();
            let topo = FatTree::new(gpus_per_tor, tors_per_pod, aggs_per_pod, num_cores, num_pods, bandwidth_bps, router);
            let scheduler = BlockScheduler::new(gpus_per_tor, block_size);
            let system_module = FatTreeCruxSystem::new();
            let ml_sim = MLSimulator::new(topo, scheduler, ImmediateFlowScheduler::new(), system_module, MaxMin::new());
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us);
        }
        "monkeytree_perfect" => {
            let router = FatTreePerfectRouter::new();
            let topo = FatTree::new(gpus_per_tor, tors_per_pod, aggs_per_pod, num_cores, num_pods, bandwidth_bps, router);
            let scheduler = BlockScheduler::new(gpus_per_tor, block_size);
            let frag_threshold = if threshold > 0 { threshold } else { aggs_per_pod };
            let config = MonkeyTreeConfig {
                fragmentation_threshold: frag_threshold,
                block_size,
            };
            let system_module = FatTreeMonkeyTreePerfect::new(config);
            let ml_sim = MLSimulator::new(topo, scheduler, ImmediateFlowScheduler::new(), system_module, MaxMin::new());
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us);
        }
        "monkeytree3" => {
            let router = FatTreePerfectRouter::new();
            let topo = FatTree::new(gpus_per_tor, tors_per_pod, aggs_per_pod, num_cores, num_pods, bandwidth_bps, router);
            let scheduler = FatTreeBlockScheduler::new(gpus_per_tor, tors_per_pod, block_size);
            let frag_threshold = if threshold > 0 { threshold } else { aggs_per_pod };
            let config = MonkeyTreeConfig {
                fragmentation_threshold: frag_threshold,
                block_size,
            };
            let system_module = FatTreeMonkeyTree3::new(config);
            let ml_sim = MLSimulator::new(topo, scheduler, ImmediateFlowScheduler::new(), system_module, MaxMin::new());
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us);
        }
        "sglb" => {
            let k = aggs_per_pod / 2;
            let config = SGLBConfig::with_remap_interval(k.max(1), 10_000);
            let router = FatTreeSGLBRouter::new(config.clone());
            let topo = FatTree::new(gpus_per_tor, tors_per_pod, aggs_per_pod, num_cores, num_pods, bandwidth_bps, router);
            let scheduler = BlockScheduler::new(gpus_per_tor, block_size);
            let system_module = FatTreeSGLBSystem::new(config);
            let ml_sim = MLSimulator::new(topo, scheduler, ImmediateFlowScheduler::new(), system_module, MaxMin::new());
            run_simulation(ml_sim, &job_defs, max_slowdown, bandwidth_bps, post_migration_delay_us);
        }
        _ => panic!("Unknown system: {}. Valid: ecmp, perfect, crux, monkeytree_perfect, monkeytree3, sglb", system),
    }
}
