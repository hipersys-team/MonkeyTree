use crate::simulator::{MLSimulator, ml_job::MLJobBuilder, WorkerEvent, JobScheduler, FlowScheduler, SystemModule};
use crate::spine::{SpineTree, SpineTreeRouter};
use crate::network::alloc::BandwidthAllocator;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use crate::utils::job_def::fetch_job;

// Generates a homogeneous mix of 2-4 worker jobs with fixed compute/flow patterns
pub fn load_homogenous_jobs<R, JS, FS, SM, A>(
    ml_sim: &mut MLSimulator<SpineTree<R>, JS, FS, SM, A>,
    num_jobs: usize,
) where
    R: SpineTreeRouter,
    JS: JobScheduler,
    FS: FlowScheduler,
    SM: SystemModule<SpineTree<R>, JS, FS>,
    A: BandwidthAllocator,
{
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
            println!("JobDefinition {} canon", num_workers);
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
            println!("JobDefinition {} canon", num_workers);
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
            println!("JobDefinition {} canon", num_workers);
        }
    }
}

// Generates a homogeneous mix of 2-4 worker jobs with fixed compute/flow patterns
pub fn load_variable_singlepeak_jobs<R, JS, FS, SM, A>(
    ml_sim: &mut MLSimulator<SpineTree<R>, JS, FS, SM, A>,
    seed: u64,
    num_jobs: usize,
    bandwidth: f64,
) where
    R: SpineTreeRouter,
    JS: JobScheduler,
    FS: FlowScheduler,
    SM: SystemModule<SpineTree<R>, JS, FS>,
    A: BandwidthAllocator,
{
    let mut rng = StdRng::seed_from_u64(seed);
    for i in 0..num_jobs {
        let num_workers = rng.gen_range(2..=4);
        let peak_time = rng.gen_range(50..=1550);
        let compute_time = 1600 - peak_time;
        let flow_size = (peak_time as f64 / 1000.0 * bandwidth) as u64;
        if num_workers == 2 {
            let job = MLJobBuilder::new(i, 0, 2, 100)
                .with_name(format!("Job {}", i))
                .add_worker_with_events(0, vec![
                    WorkerEvent::new_compute(0, compute_time, vec![]),
                    WorkerEvent::new_flow_send(1, 1, flow_size, vec![0]),
                    WorkerEvent::new_flow_receive(2, 1, flow_size, vec![0]),
                ])
                .add_worker_with_events(1, vec![
                    WorkerEvent::new_compute(0, compute_time, vec![]),
                    WorkerEvent::new_flow_send(1, 0, flow_size, vec![0]),
                    WorkerEvent::new_flow_receive(2, 0, flow_size, vec![0]),
                ])
                .build();
            ml_sim.add_job_arrival(i as u64, job);
            println!("JobDefinition {} singlepeak", num_workers);
        } else if num_workers == 3 {
            let job = MLJobBuilder::new(i, 0, 3, 100)
                .with_name(format!("Job {}", i))
                .add_worker_with_events(0, vec![
                    WorkerEvent::new_compute(0, compute_time, vec![]),
                    WorkerEvent::new_flow_send(1, 1, flow_size, vec![0]),
                    WorkerEvent::new_flow_receive(2, 2, flow_size, vec![0]),
                ])
                .add_worker_with_events(1, vec![
                    WorkerEvent::new_compute(0, compute_time, vec![]),
                    WorkerEvent::new_flow_send(1, 2, flow_size, vec![0]),
                    WorkerEvent::new_flow_receive(2, 0, flow_size, vec![0]),
                ])
                .add_worker_with_events(2, vec![
                    WorkerEvent::new_compute(0, compute_time, vec![]),
                    WorkerEvent::new_flow_send(1, 0, flow_size, vec![0]),
                    WorkerEvent::new_flow_receive(2, 1, flow_size, vec![0]),
                ])
                .build();
            ml_sim.add_job_arrival(i as u64, job);
            println!("JobDefinition {} singlepeak", num_workers);
        } else {
            let job = MLJobBuilder::new(i, 0, 4, 100)
                .with_name(format!("Job {}", i))
                .add_worker_with_events(0, vec![
                    WorkerEvent::new_compute(0, compute_time, vec![]),
                    WorkerEvent::new_flow_send(1, 1, flow_size, vec![0]),
                    WorkerEvent::new_flow_receive(2, 3, flow_size, vec![0]),
                ])
                .add_worker_with_events(1, vec![
                    WorkerEvent::new_compute(0, compute_time, vec![]),
                    WorkerEvent::new_flow_send(1, 2, flow_size, vec![0]),
                    WorkerEvent::new_flow_receive(2, 0, flow_size, vec![0]),
                ])
                .add_worker_with_events(2, vec![
                    WorkerEvent::new_compute(0, compute_time, vec![]),
                    WorkerEvent::new_flow_send(1, 3, flow_size, vec![0]),
                    WorkerEvent::new_flow_receive(2, 1, flow_size, vec![0]),
                ])
                .add_worker_with_events(3, vec![
                    WorkerEvent::new_compute(0, compute_time, vec![]),
                    WorkerEvent::new_flow_send(1, 0, flow_size, vec![0]),
                    WorkerEvent::new_flow_receive(2, 2, flow_size, vec![0]),
                ])
                .build();
            ml_sim.add_job_arrival(i as u64, job);
            println!("JobDefinition {} singlepeak", num_workers);
        }
    }
}

// Generates a random mix of s1, s2, s3 jobs with 2-4 workers
pub fn load_random_s_jobs<R, JS, FS, SM, A>(
    ml_sim: &mut MLSimulator<SpineTree<R>, JS, FS, SM, A>,
    seed: u64,
    num_jobs: usize,
) where
    R: SpineTreeRouter,
    JS: JobScheduler,
    FS: FlowScheduler,
    SM: SystemModule<SpineTree<R>, JS, FS>,
    A: BandwidthAllocator,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let kinds = ["s1", "s2", "s3"];
    for i in 0..num_jobs {
        let num_workers = rng.gen_range(2..=4);
        let kind = kinds[rng.gen_range(0..kinds.len())];
        let num_iterations = 100; // Default for this utility function
        let job = fetch_job(i, kind, num_workers, num_iterations);
        ml_sim.add_job_arrival(i as u64, job);
        println!("JobDefinition {} {} {}", num_workers, kind, num_iterations);
    }
}