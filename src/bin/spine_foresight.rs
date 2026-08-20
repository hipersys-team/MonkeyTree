#![allow(unused_imports)]
use network_sim::{
    network::alloc::{BandwidthAllocator},
    network::alloc::MaxMin,
    simulator::{MLSimulator, ml_job::MLJobBuilder, WorkerEvent, JobScheduler, FlowScheduler, SystemModule},
    simulator::{FifoScheduler},
    spine::{SpineTree, SpineTreeRouter, SpineSystemRouter, Foresight},
    flow_scheduler::ReleaseFlowScheduler,
};

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;



fn load_jobs<R, S, FS, SM, A>(ml_sim: &mut MLSimulator<SpineTree<R>, S, FS, SM, A>, num_jobs: usize) where R: SpineTreeRouter, S: JobScheduler, FS: FlowScheduler, SM: SystemModule<SpineTree<R>, S, FS>, A: BandwidthAllocator {
    let mut rng = StdRng::seed_from_u64(42);
    for i in 0..num_jobs {
        let num_workers = rng.gen_range(2..=4);
        if num_workers == 2 {
            let job = MLJobBuilder::new(i, 0, 2, 100)
            .with_name(format!("Job {}", i))
            .add_worker_with_events(0, vec![
                WorkerEvent::new_compute(0, 800, vec![]),
                WorkerEvent::new_flow_send(1, 1, 5_000_000_000, vec![0]),
                WorkerEvent::new_flow_receive(2, 1, 5_000_000_000, vec![0]),
            ])
            .add_worker_with_events(1, vec![
                WorkerEvent::new_compute(0, 800, vec![]),
                WorkerEvent::new_flow_send(1, 0, 5_000_000_000, vec![0]),
                WorkerEvent::new_flow_receive(2, 0, 5_000_000_000, vec![0]),
            ])
            .build();
            ml_sim.add_job_arrival(i as u64, job);
            println!("JobDefinition {}", num_workers);
        } else if num_workers == 3 {
            let job = MLJobBuilder::new(i, 0, 3, 100)
            .with_name(format!("Job {}", i))
            .add_worker_with_events(0, vec![
                WorkerEvent::new_compute(0, 800, vec![]),
                WorkerEvent::new_flow_send(1, 1, 5_000_000_000, vec![0]),
                WorkerEvent::new_flow_receive(2, 2, 5_000_000_000, vec![0]),
            ])
            .add_worker_with_events(1, vec![
                WorkerEvent::new_compute(0, 800, vec![]),
                WorkerEvent::new_flow_send(1, 2, 5_000_000_000, vec![0]),
                WorkerEvent::new_flow_receive(2, 0, 5_000_000_000, vec![0]),
            ])
            .add_worker_with_events(2, vec![
                WorkerEvent::new_compute(0, 800, vec![]),
                WorkerEvent::new_flow_send(1, 0, 5_000_000_000, vec![0]),
                WorkerEvent::new_flow_receive(2, 1, 5_000_000_000, vec![0]),
            ])
            .build();
            ml_sim.add_job_arrival(i as u64, job);
            println!("JobDefinition {}", num_workers);
        } else {
            let job = MLJobBuilder::new(i, 0, 4, 100)
            .with_name(format!("Job {}", i))
            .add_worker_with_events(0, vec![
                WorkerEvent::new_compute(0, 800, vec![]),
                WorkerEvent::new_flow_send(1, 1, 5_000_000_000, vec![0]),
                WorkerEvent::new_flow_receive(2, 3, 5_000_000_000, vec![0]),
            ])
            .add_worker_with_events(1, vec![
                WorkerEvent::new_compute(0, 800, vec![]),
                WorkerEvent::new_flow_send(1, 2, 5_000_000_000, vec![0]),
                WorkerEvent::new_flow_receive(2, 0, 5_000_000_000, vec![0]),
            ])
            .add_worker_with_events(2, vec![
                WorkerEvent::new_compute(0, 800, vec![]),
                WorkerEvent::new_flow_send(1, 3, 5_000_000_000, vec![0]),
                WorkerEvent::new_flow_receive(2, 1, 5_000_000_000, vec![0]),
            ])
            .add_worker_with_events(3, vec![
                WorkerEvent::new_compute(0, 800, vec![]),
                WorkerEvent::new_flow_send(1, 0, 5_000_000_000, vec![0]),
                WorkerEvent::new_flow_receive(2, 2, 5_000_000_000, vec![0]),
            ])
            .build();
            ml_sim.add_job_arrival(i as u64, job);
            println!("JobDefinition {}", num_workers);
        }

    }
}

fn main() {
    let router = SpineSystemRouter::new();
    //let router = SpineCruxRouter::new();
    let spine_tree = SpineTree::new(
        3,
        3,
        3,
        50.0e9,
        router,
    );

    let scheduler = FifoScheduler::new();
    let flow_scheduler = ReleaseFlowScheduler::new();
    let system_module = Foresight::new();

    let mut ml_sim = MLSimulator::new(spine_tree, scheduler, flow_scheduler, system_module, MaxMin::new());

    load_jobs(&mut ml_sim, 50);

    while let Some(_kind) = ml_sim.advance_next_step() {}
}
