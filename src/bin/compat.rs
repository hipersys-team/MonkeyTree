#![allow(unused_imports)]
use std::{env, fs};
use network_sim::{
    simulator::{MLSimulator, ml_job::MLJobBuilder, WorkerEvent},
    simulator::{ImmediateFlowScheduler, NoopSystemModule},
    spine::{SpineTree, SpineEcmpRouter},
    network::alloc::{BandwidthAllocator},
    network::alloc::MaxMin,
    network::mltcp::MLTCP,
    network::mltcp_topo::MLTCPTopo,
    network::mltcp_topo_approx::MLTCPTopoApprox,
    network::mltcp_topo_bytes::MLTCPTopoBytes,
    schedulers::SnapshotScheduler,
};

// Simple runtime-selectable allocator wrapper (no external CLI crate required)
enum CompatAllocator {
    MaxMin(MaxMin),
    MLTCPTopo(MLTCPTopo),
    MLTCPTopoBytes(MLTCPTopoBytes),
    MLTCPTopoApprox(MLTCPTopoApprox),
}

impl BandwidthAllocator for CompatAllocator {
    fn set_context(&mut self, context: &network_sim::simulator::ml_simulator::MLContext) {
        match self {
            CompatAllocator::MaxMin(inner) => inner.set_context(context),
            CompatAllocator::MLTCPTopo(inner) => inner.set_context(context),
            CompatAllocator::MLTCPTopoBytes(inner) => inner.set_context(context),
            CompatAllocator::MLTCPTopoApprox(inner) => inner.set_context(context),
        }
    }

    fn allocate(
        &self,
        topo: &impl network_sim::network::topology::Topology,
        active_desc: &indexmap::IndexMap<network_sim::network::flow::FlowId, network_sim::network::flow::FlowDesc>,
        active_state: &indexmap::IndexMap<network_sim::network::flow::FlowId, network_sim::network::flow::FlowState>,
    ) -> Vec<f64> {
        match self {
            CompatAllocator::MaxMin(inner) => inner.allocate(topo, active_desc, active_state),
            CompatAllocator::MLTCPTopo(inner) => inner.allocate(topo, active_desc, active_state),
            CompatAllocator::MLTCPTopoBytes(inner) => inner.allocate(topo, active_desc, active_state),
            CompatAllocator::MLTCPTopoApprox(inner) => inner.allocate(topo, active_desc, active_state),
        }
    }
}

fn make_allocator(name: &str) -> CompatAllocator {
    match name {
        "minmax" => CompatAllocator::MaxMin(MaxMin::new()),
        "mltcp_topo" => CompatAllocator::MLTCPTopo(MLTCPTopo::new()),
        "mltcp_topo_bytes" => CompatAllocator::MLTCPTopoBytes(MLTCPTopoBytes::new()),
        "mltcp_topo_approx" => CompatAllocator::MLTCPTopoApprox(MLTCPTopoApprox::new()),
        other => panic!(
            "Unknown allocator '{}'. Use one of: minmax, mltcp_topo, mltcp_topo_bytes, mltcp_topo_approx",
            other
        ),
    }
}

fn parse_job_file(contents: &str) -> (usize, Vec<Vec<(f64, f64)>>) {
    let mut lines = contents.lines().filter(|l| !l.trim().is_empty());
    let num_jobs: usize = lines
        .next()
        .expect("job file missing num_jobs")
        .trim()
        .parse()
        .expect("invalid num_jobs");

    let mut jobs: Vec<Vec<(f64, f64)>> = Vec::new();
    for _ in 0..num_jobs {
        let line = lines.next().expect("missing job line");
        let mut it = line.split_whitespace();
        let num_peaks: usize = it
            .next()
            .expect("missing num_peaks")
            .parse()
            .expect("invalid num_peaks");
        let mut pairs = Vec::with_capacity(num_peaks);
        for _ in 0..num_peaks {
            let comp_us: f64 = it
                .next()
                .expect("missing comp_time")
                .parse()
                .expect("invalid comp_time");
            let comm_us: f64 = it
                .next()
                .expect("missing comm_time")
                .parse()
                .expect("invalid comm_time");
            pairs.push((comp_us, comm_us));
        }
        jobs.push(pairs);
    }
    (num_jobs, jobs)
}

fn main() {
    // Usage: compat <JOB_FILE> [ALLOCATOR]
    // ALLOCATOR in {minmax, mltcp_topo, mltcp_topo_bytes, mltcp_topo_approx}, default: mltcp_topo
    let path = env::args().nth(1).expect("Usage: compat <JOB_FILE> [ALLOCATOR]");
    let alloc_name = env::args().nth(2).unwrap_or_else(|| "mltcp_topo".to_string());
    let contents = fs::read_to_string(&path).expect("failed to read job file");
    let (_declared_jobs, jobs_spec) = parse_job_file(&contents);

    // Topology: 2 ToRs (leaves), 6 hosts per ToR -> up to 6 two-worker jobs split across ToRs
    let hosts_per_leaf = 6usize;
    let num_leaves = 2usize;
    let num_spines = 1usize;
    let link_bandwidth_bps: f64 = 50.0e9; // bps

    let router = SpineEcmpRouter::new(42);
    let spine_tree = SpineTree::new(
        hosts_per_leaf,
        num_leaves,
        num_spines,
        link_bandwidth_bps,
        router,
    );

    // Preconfigure placements so each job's two workers land on different ToRs.
    let mut scheduler = SnapshotScheduler::new();
    let max_jobs = jobs_spec.len().min(hosts_per_leaf);
    for j in 0..max_jobs {
        let w0_host = j; // leaf 0
        let w1_host = j + hosts_per_leaf; // leaf 1
        scheduler.set_job_placement(j, vec![w0_host, w1_host]);
    }

    let flow_scheduler = ImmediateFlowScheduler::new();
    let system_module = NoopSystemModule::default();
    let allocator = make_allocator(&alloc_name);
    let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, allocator);

    // Build and enqueue jobs
    let bytes_per_us_at_line_rate = link_bandwidth_bps / 8.0 / 1_000_000.0; // B/us
    for j in 0..max_jobs {
        let peaks = &jobs_spec[j];
        let mut builder = MLJobBuilder::new(j, 0, 2, 1000).with_name(format!("Job {}", j));

        // Build full event lists per worker
        let mut events_w0 = Vec::new();
        let mut events_w1 = Vec::new();

        let mut next_id0 = 0usize;
        let mut next_id1 = 0usize;
        let mut deps0: Vec<usize> = Vec::new();
        let mut deps1: Vec<usize> = Vec::new();
        for (comp_us_f, comm_us_f) in peaks.iter().copied() {
            // Round compute time to nearest ms, with a minimum of 0
            let comp_us = comp_us_f.max(0.0).round() as u64;
            // Worker 0
            let comp0 = next_id0; next_id0 += 1;
            events_w0.push(WorkerEvent::new_compute(comp0, comp_us, deps0.clone()));
            // Convert communication time (ms) to bytes at line rate, rounded to nearest byte
            let size_bytes = (comm_us_f.max(0.0) * bytes_per_us_at_line_rate).round() as u64;
            let send0 = next_id0; next_id0 += 1;
            let recv0 = next_id0; next_id0 += 1;
            events_w0.push(WorkerEvent::new_flow_send(send0, 1, size_bytes, vec![comp0]));
            events_w0.push(WorkerEvent::new_flow_receive(recv0, 1, size_bytes, vec![comp0]));
            deps0 = vec![send0, recv0];

            // Worker 1
            let comp1 = next_id1; next_id1 += 1;
            events_w1.push(WorkerEvent::new_compute(comp1, comp_us, deps1.clone()));
            let send1 = next_id1; next_id1 += 1;
            let recv1 = next_id1; next_id1 += 1;
            events_w1.push(WorkerEvent::new_flow_send(send1, 0, size_bytes, vec![comp1]));
            events_w1.push(WorkerEvent::new_flow_receive(recv1, 0, size_bytes, vec![comp1]));
            deps1 = vec![send1, recv1];
        }

        builder = builder
            .add_worker_with_events(0, events_w0)
            .add_worker_with_events(1, events_w1);

        let job = builder.build();
        ml_sim.add_job_arrival(j as u64, job);
    }

    while let Some(_kind) = ml_sim.advance_next_step() {}
}
