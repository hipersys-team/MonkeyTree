#![allow(unused_imports)]
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

use network_sim::{
    network::alloc::{BandwidthAllocator},
    network::alloc::MaxMin,
    simulator::{MLSimulator, ml_job::MLJobBuilder, WorkerEvent, FlowScheduler, JobScheduler, SystemModule},
    schedulers::SnapshotScheduler,
    spine::{SpineTree, SpineTreeRouter, SpineEcmpRouter, SpineCassiniSystemModule},
    flow_scheduler::{CassiniFlowScheduler},
    ImmediateFlowScheduler,
};
use network_sim::utils::job_def::fetch_job;

fn parse_snapshot_file<P: AsRef<Path>>(path: P) -> (HashMap<usize, (usize, String)>, HashMap<usize, Vec<usize>>) {
    let content = fs::read_to_string(path).expect("Failed to read snapshot file");
    let mut lines = content.lines();

    let _num_jobs = lines
        .next()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .expect("Snapshot must start with number of jobs");

    let mut job_defs: HashMap<usize, (usize, String)> = HashMap::new();
    for line in &mut lines {
        let t = line.trim();
        if t.is_empty() { break; }
        let mut parts = t.split(':');
        let job_id = parts
            .next()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .expect("Invalid job_id in snapshot");
        let rhs = parts.next().expect("Missing job spec after ':'").trim();
        let mut rhs_parts = rhs.split_whitespace();
        let num_workers = rhs_parts
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Invalid num_workers in snapshot");
        let job_type = rhs_parts
            .next()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "canon".to_string());
        job_defs.insert(job_id, (num_workers, job_type));
    }

    let placement_hdr = lines.next().unwrap_or("").trim();
    assert!(placement_hdr.eq_ignore_ascii_case("Placement"), "Expected 'Placement' section");

    let mut job_to_hosts: HashMap<usize, Vec<usize>> = HashMap::new();
    for line in lines {
        let t = line.trim();
        if t.is_empty() { continue; }
        let (host_idx_opt, job_id_opt) = if let Some(rest) = t.strip_prefix("host ") {
            let mut it = rest.split("->");
            let h = it.next().and_then(|s| s.trim().parse::<usize>().ok());
            let j = it
                .next()
                .map(|s| s.trim())
                .and_then(|s| s.strip_prefix("job ").or(Some(s)))
                .and_then(|s| s.parse::<isize>().ok())
                .and_then(|v| if v >= 0 { Some(v as usize) } else { None });
            (h, j)
        } else {
            let mut parts = t.split(':');
            let h = parts.next().and_then(|s| s.trim().parse::<usize>().ok());
            let j = parts.next().and_then(|s| s.trim().parse::<usize>().ok());
            (h, j)
        };
        if let (Some(host_idx), Some(job_id)) = (host_idx_opt, job_id_opt) {
            job_to_hosts.entry(job_id).or_default().push(host_idx);
        }
    }

    for hosts in job_to_hosts.values_mut() {
        hosts.sort_unstable();
    }

    (job_defs, job_to_hosts)
}

fn build_and_submit_job<R, JS, FS, SM, A>(
    ml_sim: &mut MLSimulator<SpineTree<R>, JS, FS, SM, A>,
    job_id: usize,
    num_workers: usize,
    job_type: &str,
) where
    R: SpineTreeRouter,
    JS: JobScheduler,
    FS: FlowScheduler,
    SM: SystemModule<SpineTree<R>, JS, FS>,
    A: BandwidthAllocator,
{
    let num_iterations = 100; // Default for snapshot tool
    let job = fetch_job(job_id, job_type, num_workers, num_iterations);
    ml_sim.add_job_arrival(0, job);
    println!("JobDefinition {} {} {}", num_workers, job_type, num_iterations);
}

fn main() {
    let snapshot_path = env::args().nth(1).expect("Usage: spine_snapshot_crux <SNAPSHOT_FILE>");

    // Use a fixed seed for deterministic ECMP selection across runs
    let router = SpineEcmpRouter::new(42);
    let spine_tree = SpineTree::new(
        6,
        6,
        3,
        50.0e9,
        router,
    );

    let (job_defs, job_to_hosts) = parse_snapshot_file(snapshot_path);

    let mut scheduler = SnapshotScheduler::new();
    for (&job_id, hosts) in job_to_hosts.iter() {
        if let Some(&(num_workers, ref _job_type)) = job_defs.get(&job_id) {
            let desired: Vec<usize> = hosts.iter().copied().take(num_workers).collect();
            scheduler.set_job_placement(job_id, desired);
        }
    }

    let flow_scheduler = CassiniFlowScheduler::new();
    // Enable static placement mode: Cassini will not choose placements, it will
    // compute and apply schedules based on the provided snapshot placements.
    let system_module = SpineCassiniSystemModule::default().with_static_placement();
    let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());

    for (&job_id, &(num_workers, ref job_type)) in job_defs.iter() {
        build_and_submit_job(&mut ml_sim, job_id, num_workers, job_type);
    }

    while let Some(_kind) = ml_sim.advance_next_step() {}
}


