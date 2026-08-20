#![allow(unused_imports)]
use network_sim::{
    network::alloc::{BandwidthAllocator},
    network::alloc::MaxMin,
    simulator::{MLSimulator, ml_job::MLJobBuilder, WorkerEvent},
    simulator::{FifoScheduler, ImmediateFlowScheduler, NoopSystemModule, SystemModule, FlowScheduler, JobScheduler},
    spine::{SpineTree, SpineTreeRouter, SpineEcmpRouter, SpineCruxRouter, SpineCassiniSystemModule},
    flow_scheduler::{CassiniFlowScheduler},
    job_schedulers::{SimpleScheduler},
    utils::load_random_s_jobs,
};

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use network_sim::utils::load_homogenous_jobs;




fn main() {
    let hosts_per_leaf = 6;
    let num_leaves = 24;
    let num_spines = 6;

    // Print topology header for snapshot extraction
    println!("Topology {} {} {}", hosts_per_leaf, num_leaves, num_spines);

    let router = SpineEcmpRouter::new(42);
    let spine_tree = SpineTree::new(
        hosts_per_leaf,
        num_leaves,
        num_spines,
        50.0e9,
        router,
    );

    let scheduler = FifoScheduler::new();
    let flow_scheduler = ImmediateFlowScheduler::new();
    let system_module = NoopSystemModule::default();

    let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());

    load_homogenous_jobs(&mut ml_sim, 100);

    while let Some(_kind) = ml_sim.advance_next_step() {}
}
