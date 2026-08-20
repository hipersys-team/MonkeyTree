//! SGLB (Spine Group Load Balancing) test binary.
//!
//! Runs a simulation with SGLB routing on a spine-leaf topology.

#![allow(unused_imports)]
use network_sim::{
    simulator::{MLSimulator, ml_job::MLJobBuilder, WorkerEvent},
    simulator::{FifoScheduler, ImmediateFlowScheduler, SystemModule, FlowScheduler, JobScheduler},
    spine::{SpineTree, SpineTreeRouter, SGLBRouter, SGLBSystemModule, SGLBConfig},
    flow_scheduler::CassiniFlowScheduler,
    job_schedulers::SimpleScheduler,
    network::alloc::{BandwidthAllocator, MaxMin},
};

use network_sim::utils::load_homogenous_jobs;

fn main() {
    // Configure SGLB with K=4 (top 4 spines eligible)
    // Default uses job-event based remapping (remap_interval_us = 0)
    let config = SGLBConfig::with_k(4);
    let router = SGLBRouter::new(config.clone());
    
    // Create spine-leaf topology: 6 hosts/leaf, 6 leaves, 8 spines, 50 Gbps links
    let spine_tree = SpineTree::new(
        6,   // hosts_per_leaf
        6,   // num_leaves
        8,   // num_spines
        50.0e9,  // bandwidth (50 Gbps)
        router,
    );

    let scheduler = FifoScheduler::new();
    let flow_scheduler = ImmediateFlowScheduler::new();
    let system_module = SGLBSystemModule::new(config);

    let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());

    // Load test jobs
    load_homogenous_jobs(&mut ml_sim, 50);

    println!("[SGLB Test] Starting simulation...");
    
    let mut step_count = 0;
    while let Some(_kind) = ml_sim.advance_next_step() {
        step_count += 1;
    }
    
    println!("[SGLB Test] Simulation complete. Total steps: {}", step_count);
}
