//! All-to-All Ring collective communication pattern.
//!
//! In All-to-All Ring, instead of doing all N*(N-1) flows simultaneously,
//! we perform N-1 rounds where in each round, every worker sends to a neighbor
//! at a specific distance. This creates a phased approach where:
//! - Round 1: worker i sends to (i + 1) % N
//! - Round 2: worker i sends to (i + 2) % N
//! - ...
//! - Round N-1: worker i sends to (i + N-1) % N
//!
//! Each round completes before the next begins, creating a dependency chain.

use crate::simulator::WorkerEvent;
use crate::simulator::ml_worker::FlowKind;
use super::CollectiveEvents;

/// Generates All-to-All Ring collective events for all workers.
///
/// Instead of simultaneous all-to-all, this creates N-1 rounds of ring-style
/// exchanges. Each round has all workers sending to a neighbor at distance `round`.
///
/// # Arguments
/// * `num_workers` - Number of workers participating in the collective
/// * `data_size_per_pair` - Bytes each worker sends to each other worker
/// * `start_event_id` - First event ID to use for generated events
/// * `dependencies` - Event IDs that must complete before the first round starts
///
/// # Returns
/// A `CollectiveEvents` struct containing the generated events for each worker.
///
/// # Event structure per worker
/// For a worker `i` in an n-worker collective:
/// - Round 1: send to (i+1)%n, receive from (i-1+n)%n
/// - Round 2: send to (i+2)%n, receive from (i-2+n)%n  (depends on round 1)
/// - ...
/// - Round N-1: send to (i+N-1)%n, receive from (i-(N-1)+n)%n
pub fn generate_all_to_all_ring_events(
    num_workers: usize,
    data_size_per_pair: u64,
    start_event_id: usize,
    dependencies: Vec<usize>,
) -> CollectiveEvents {
    let mut worker_events: Vec<Vec<WorkerEvent>> = vec![vec![]; num_workers];
    let mut completion_event_ids: Vec<Vec<usize>> = vec![vec![]; num_workers];
    
    if num_workers <= 1 {
        return CollectiveEvents {
            worker_events,
            events_per_worker: 0,
            completion_event_ids,
        };
    }
    
    let num_rounds = num_workers - 1;
    // Each round has 2 events per worker (1 send + 1 receive)
    let events_per_worker = 2 * num_rounds;
    
    let mut current_event_id = start_event_id;
    
    // Track the dependency for each round per worker
    // Initially, all workers depend on the provided dependencies
    let mut round_deps: Vec<Vec<usize>> = vec![dependencies.clone(); num_workers];
    
    for round in 1..=num_rounds {
        // In this round, worker i sends to (i + round) % N
        // and receives from (i - round + N) % N
        
        // First, generate all events for this round
        let mut round_event_ids: Vec<Vec<usize>> = vec![vec![]; num_workers];
        
        for worker_id in 0..num_workers {
            let dst_worker = (worker_id + round) % num_workers;
            let src_worker = (worker_id + num_workers - round) % num_workers;
            
            // Send event
            let send_id = current_event_id;
            current_event_id += 1;
            
            worker_events[worker_id].push(WorkerEvent::new_flow_send_with_kind(
                send_id,
                dst_worker,
                data_size_per_pair,
                round_deps[worker_id].clone(),
                FlowKind::AllToAll,
            ));
            round_event_ids[worker_id].push(send_id);
            
            // Receive event
            let recv_id = current_event_id;
            current_event_id += 1;
            
            worker_events[worker_id].push(WorkerEvent::new_flow_receive_with_kind(
                recv_id,
                src_worker,
                data_size_per_pair,
                round_deps[worker_id].clone(),
                FlowKind::AllToAll,
            ));
            round_event_ids[worker_id].push(recv_id);
        }
        
        // Update dependencies for next round - each worker depends on their own round events
        for worker_id in 0..num_workers {
            round_deps[worker_id] = round_event_ids[worker_id].clone();
        }
    }
    
    // The completion events are the last round's events for each worker
    for worker_id in 0..num_workers {
        completion_event_ids[worker_id] = round_deps[worker_id].clone();
    }
    
    CollectiveEvents {
        worker_events,
        events_per_worker,
        completion_event_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_all_to_all_ring_event_count() {
        // 4 workers: 3 rounds * 2 events = 6 events per worker
        let result = generate_all_to_all_ring_events(4, 1000, 0, vec![]);
        
        assert_eq!(result.worker_events.len(), 4);
        assert_eq!(result.events_per_worker, 6);
        
        for events in &result.worker_events {
            assert_eq!(events.len(), 6);
        }
    }
    
    #[test]
    fn test_all_to_all_ring_two_workers() {
        // 2 workers: 1 round, each sends to the other
        let result = generate_all_to_all_ring_events(2, 5_000_000_000, 0, vec![]);
        
        assert_eq!(result.worker_events.len(), 2);
        assert_eq!(result.events_per_worker, 2);
        
        // Worker 0: sends to 1, receives from 1
        let w0 = &result.worker_events[0];
        assert_eq!(w0.len(), 2);
        assert_eq!(w0[0].flow_send.as_ref().unwrap().dst_worker, 1);
        assert_eq!(w0[1].flow_receive.as_ref().unwrap().src_worker, 1);
        
        // Worker 1: sends to 0, receives from 0
        let w1 = &result.worker_events[1];
        assert_eq!(w1.len(), 2);
        assert_eq!(w1[0].flow_send.as_ref().unwrap().dst_worker, 0);
        assert_eq!(w1[1].flow_receive.as_ref().unwrap().src_worker, 0);
    }
    
    #[test]
    fn test_all_to_all_ring_three_workers_topology() {
        // 3 workers: 2 rounds
        // Round 1: 0→1, 1→2, 2→0
        // Round 2: 0→2, 1→0, 2→1
        let result = generate_all_to_all_ring_events(3, 1000, 0, vec![]);
        
        assert_eq!(result.events_per_worker, 4);  // 2 rounds * 2 events
        
        // Check worker 0
        let w0 = &result.worker_events[0];
        // Round 1: send to 1, recv from 2
        assert_eq!(w0[0].flow_send.as_ref().unwrap().dst_worker, 1);
        assert_eq!(w0[1].flow_receive.as_ref().unwrap().src_worker, 2);
        // Round 2: send to 2, recv from 1
        assert_eq!(w0[2].flow_send.as_ref().unwrap().dst_worker, 2);
        assert_eq!(w0[3].flow_receive.as_ref().unwrap().src_worker, 1);
        
        // Check worker 1
        let w1 = &result.worker_events[1];
        // Round 1: send to 2, recv from 0
        assert_eq!(w1[0].flow_send.as_ref().unwrap().dst_worker, 2);
        assert_eq!(w1[1].flow_receive.as_ref().unwrap().src_worker, 0);
        // Round 2: send to 0, recv from 2
        assert_eq!(w1[2].flow_send.as_ref().unwrap().dst_worker, 0);
        assert_eq!(w1[3].flow_receive.as_ref().unwrap().src_worker, 2);
    }
    
    #[test]
    fn test_all_to_all_ring_round_dependencies() {
        // Each round should depend on the previous round completing
        let result = generate_all_to_all_ring_events(4, 1000, 100, vec![99]);
        
        // Worker 0's events
        let w0 = &result.worker_events[0];
        
        // Round 1 events (IDs 100, 101) should depend on initial deps [99]
        assert_eq!(w0[0].dependencies, vec![99]);  // send
        assert_eq!(w0[1].dependencies, vec![99]);  // recv
        
        // Round 2 events (IDs 108, 109) should depend on round 1 [100, 101]
        assert_eq!(w0[2].dependencies, vec![100, 101]);  // send
        assert_eq!(w0[3].dependencies, vec![100, 101]);  // recv
        
        // Round 3 events should depend on round 2
        assert_eq!(w0[4].dependencies, vec![108, 109]);
        assert_eq!(w0[5].dependencies, vec![108, 109]);
    }
    
    #[test]
    fn test_all_to_all_ring_completion_events() {
        // Completion events should be the last round's events
        let result = generate_all_to_all_ring_events(3, 1000, 0, vec![]);
        
        // 3 workers, 2 rounds, last round events should be completion
        for worker_id in 0..3 {
            let events = &result.worker_events[worker_id];
            let completion = &result.completion_event_ids[worker_id];
            
            // Last 2 events should be the completion events
            assert_eq!(completion.len(), 2);
            assert!(completion.contains(&events[events.len() - 2].id));
            assert!(completion.contains(&events[events.len() - 1].id));
        }
    }
    
    #[test]
    fn test_all_to_all_ring_single_worker() {
        // Single worker has no one to communicate with
        let result = generate_all_to_all_ring_events(1, 1000, 0, vec![]);
        
        assert_eq!(result.worker_events.len(), 1);
        assert_eq!(result.events_per_worker, 0);
        assert!(result.worker_events[0].is_empty());
    }
    
    #[test]
    fn test_all_to_all_ring_data_size() {
        // Verify data sizes are preserved
        let data_size = 1_000_000_000u64;
        let result = generate_all_to_all_ring_events(4, data_size, 0, vec![]);
        
        for events in &result.worker_events {
            for event in events {
                if let Some(send) = &event.flow_send {
                    assert_eq!(send.size_bytes, data_size);
                }
                if let Some(recv) = &event.flow_receive {
                    assert_eq!(recv.size_bytes, data_size);
                }
            }
        }
    }
    
    #[test]
    fn test_all_to_all_ring_covers_all_pairs() {
        // Each worker should send to every other worker exactly once across all rounds
        let num_workers = 5;
        let result = generate_all_to_all_ring_events(num_workers, 1000, 0, vec![]);
        
        for worker_id in 0..num_workers {
            let events = &result.worker_events[worker_id];
            
            // Collect all destinations
            let mut destinations: Vec<usize> = events.iter()
                .filter_map(|e| e.flow_send.as_ref().map(|s| s.dst_worker))
                .collect();
            destinations.sort();
            
            // Should have sent to all other workers
            let mut expected: Vec<usize> = (0..num_workers)
                .filter(|&i| i != worker_id)
                .collect();
            expected.sort();
            
            assert_eq!(destinations, expected, "Worker {} didn't send to all others", worker_id);
        }
    }
}
