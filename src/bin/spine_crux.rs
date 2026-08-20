#![allow(unused_imports)]
use network_sim::{
    simulator::{MLSimulator, ml_job::MLJobBuilder, WorkerEvent},
    simulator::{FifoScheduler, ImmediateFlowScheduler, SystemModule, FlowScheduler, JobScheduler},
    spine::{SpineTree, SpineTreeRouter, SpineCruxRouter, CruxSystemModule},
    flow_scheduler::{CassiniFlowScheduler},
    job_schedulers::{SimpleScheduler},
    network::alloc::{BandwidthAllocator},
    network::alloc::MaxMin,
};

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use network_sim::utils::load_homogenous_jobs;



// load_jobs moved to utils::job_gen as load_homogenous_jobs

fn main() {
    let router = SpineCruxRouter::new();
    let spine_tree = SpineTree::new(
        6,
        6,
        3,
        50.0e9,
        router,
    );

    let scheduler = FifoScheduler::new();
    let flow_scheduler = ImmediateFlowScheduler::new();
    let system_module = CruxSystemModule::default();

    let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());

    load_homogenous_jobs(&mut ml_sim, 50);

    while let Some(_kind) = ml_sim.advance_next_step() {}
}
