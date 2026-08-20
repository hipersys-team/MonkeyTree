#![allow(unused_imports)]
use network_sim::{
    network::{
        FatTree,
        routing::FatTreeRouter,
    },
    network::alloc::{BandwidthAllocator},
    network::alloc::MaxMin,
    simulator::{MLSimulator, ml_job::MLJobBuilder, WorkerEvent},
    simulator::{FifoScheduler, ImmediateFlowScheduler, NoopSystemModule},
    routing::{EcmpRouter, CruxRouter},
};

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;



fn load_jobs<R, A>(ml_sim: &mut MLSimulator<FatTree<R>, FifoScheduler, ImmediateFlowScheduler, NoopSystemModule, A>, num_jobs: usize) where R: FatTreeRouter, A: BandwidthAllocator {
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
    let router = EcmpRouter::new(42);
    //let router = CruxRouter::new();
    let fat_tree = FatTree::new(
        3,
        3,
        3,
        9,
        6,
        50.0e9,
        router,
    );

    let scheduler = FifoScheduler::new();
    let flow_scheduler = ImmediateFlowScheduler::new();
    let system_module = NoopSystemModule::default();

    let mut ml_sim = MLSimulator::new(fat_tree, scheduler, flow_scheduler, system_module, MaxMin::new());

    load_jobs(&mut ml_sim, 50);

    while let Some(_kind) = ml_sim.advance_next_step() {
        //println!("{kind:?} @ {} us", ml_sim.now_us);
        //ml_sim.dump_running_jobs();
        //ml_sim.dump_cluster_state();
    }
}
