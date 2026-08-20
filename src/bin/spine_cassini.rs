#![allow(unused_imports)]
use network_sim::{
    simulator::{MLSimulator, ml_job::MLJobBuilder, WorkerEvent},
    simulator::{FifoScheduler, ImmediateFlowScheduler, NoopSystemModule, SystemModule, FlowScheduler, JobScheduler},
    spine::{SpineTree, SpineTreeRouter, SpineEcmpRouter, SpineCruxRouter, SpineCassiniSystemModule},
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
    let router = SpineEcmpRouter::new(42);
    //let router = SpineCruxRouter::new();
    let spine_tree = SpineTree::new(
        8,
        16,
        8,
        50.0e9,
        router,
    );

    let scheduler = SimpleScheduler::new();
    //let flow_scheduler = ImmediateFlowScheduler::new();
    //let system_module = NoopSystemModule::default();
    let flow_scheduler = CassiniFlowScheduler::new();
    let system_module = SpineCassiniSystemModule::new();

    let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());

    load_homogenous_jobs(&mut ml_sim, 50);

    while let Some(_kind) = ml_sim.advance_next_step() {}
}
