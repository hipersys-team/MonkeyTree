//! AllReduce collective communication pattern.
//!
//! AllReduce combines values from all workers and distributes the result back
//! to all workers. This implementation models the communication pattern of a
//! ring-based AllReduce, where each worker exchanges data with neighbors.
//!
//! Total data per worker: 2 * (n-1) / n * model_size
//! This accounts for both reduce-scatter and all-gather phases.
//!
//! Ring topology: 0 → 1 → 2 → ... → N-1 → 0

use crate::simulator::WorkerEvent;
use crate::simulator::ml_worker::FlowKind;
use super::CollectiveEvents;

/// Generates AllReduce collective events for all workers.
///
/// Models a ring-based AllReduce where each worker sends to the next worker
/// and receives from the previous worker. The data size is scaled by
/// 2 * (n-1) / n to match the total data transferred in a real AllReduce.
///
/// # Arguments
/// * `num_workers` - Number of workers participating in the collective
/// * `model_size` - Model size in bytes (will be scaled by 2*(n-1)/n)
/// * `start_event_id` - First event ID to use for generated events
/// * `dependencies` - Event IDs that must complete before the collective starts
///
/// # Returns
/// A `CollectiveEvents` struct containing the generated events for each worker.
///
/// # Event structure per worker
/// For a worker `i` in an n-worker ring:
/// - 1 FlowSend event: sends to worker (i + 1) % n
/// - 1 FlowReceive event: receives from worker (i - 1 + n) % n
/// - Both depend on `dependencies` and execute in parallel
pub fn generate_allreduce_events(
    num_workers: usize,
    model_size: u64,
    start_event_id: usize,
    dependencies: Vec<usize>,
) -> CollectiveEvents {
    let mut worker_events: Vec<Vec<WorkerEvent>> = Vec::with_capacity(num_workers);
    let mut completion_event_ids: Vec<Vec<usize>> = Vec::with_capacity(num_workers);
    
    // Each worker has exactly 2 events: 1 send + 1 receive
    let events_per_worker = if num_workers > 1 { 2 } else { 0 };
    
    // Calculate actual data size: 2 * (n-1) / n * model_size
    // This is the total data transferred per worker in a ring AllReduce
    let n = num_workers as f64;
    let data_size = if num_workers > 1 {
        ((2.0 * (n - 1.0) / n) * model_size as f64) as u64
    } else {
        0
    };
    
    let mut current_event_id = start_event_id;
    
    for worker_id in 0..num_workers {
        let mut events = Vec::with_capacity(events_per_worker);
        let mut final_event_ids = Vec::new();
        
        if num_workers > 1 {
            // Next worker in ring (send destination)
            let next = (worker_id + 1) % num_workers;
            // Previous worker in ring (receive source)
            let prev = if worker_id == 0 { num_workers - 1 } else { worker_id - 1 };
            
            // Send to next worker (FlowKind::Ring for priority routing)
            let send_id = current_event_id;
            current_event_id += 1;
            events.push(WorkerEvent::new_flow_send_with_kind(
                send_id,
                next,
                data_size,
                dependencies.clone(),
                FlowKind::Ring,
            ));
            final_event_ids.push(send_id);
            
            // Receive from previous worker
            let recv_id = current_event_id;
            current_event_id += 1;
            events.push(WorkerEvent::new_flow_receive_with_kind(
                recv_id,
                prev,
                data_size,
                dependencies.clone(),
                FlowKind::Ring,
            ));
            final_event_ids.push(recv_id);
        }
        
        worker_events.push(events);
        completion_event_ids.push(final_event_ids);
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
    fn test_allreduce_event_count() {
        // 4 workers: each sends to 1, receives from 1 = 2 events each
        let result = generate_allreduce_events(4, 1_000_000_000, 0, vec![]);
        
        assert_eq!(result.worker_events.len(), 4);
        assert_eq!(result.events_per_worker, 2);
        
        for events in &result.worker_events {
            assert_eq!(events.len(), 2);
        }
    }
    
    #[test]
    fn test_allreduce_data_size() {
        // 4 workers, model_size = 1GB
        // Expected data: 2 * (4-1) / 4 * 1GB = 2 * 0.75 * 1GB = 1.5GB
        let model_size = 1_000_000_000u64;
        let result = generate_allreduce_events(4, model_size, 0, vec![]);
        
        let expected_data = (2.0 * 3.0 / 4.0 * model_size as f64) as u64;
        
        let w0 = &result.worker_events[0];
        assert_eq!(w0[0].flow_send.as_ref().unwrap().size_bytes, expected_data);
        assert_eq!(w0[1].flow_receive.as_ref().unwrap().size_bytes, expected_data);
    }
    
    #[test]
    fn test_allreduce_topology() {
        let result = generate_allreduce_events(4, 5_000_000_000, 0, vec![]);
        
        // Worker 0: sends to 1, receives from 3
        let w0 = &result.worker_events[0];
        assert_eq!(w0[0].flow_send.as_ref().unwrap().dst_worker, 1);
        assert_eq!(w0[1].flow_receive.as_ref().unwrap().src_worker, 3);
        
        // Worker 1: sends to 2, receives from 0
        let w1 = &result.worker_events[1];
        assert_eq!(w1[0].flow_send.as_ref().unwrap().dst_worker, 2);
        assert_eq!(w1[1].flow_receive.as_ref().unwrap().src_worker, 0);
        
        // Worker 2: sends to 3, receives from 1
        let w2 = &result.worker_events[2];
        assert_eq!(w2[0].flow_send.as_ref().unwrap().dst_worker, 3);
        assert_eq!(w2[1].flow_receive.as_ref().unwrap().src_worker, 1);
        
        // Worker 3: sends to 0, receives from 2
        let w3 = &result.worker_events[3];
        assert_eq!(w3[0].flow_send.as_ref().unwrap().dst_worker, 0);
        assert_eq!(w3[1].flow_receive.as_ref().unwrap().src_worker, 2);
    }
    
    #[test]
    fn test_allreduce_two_workers() {
        // 2 workers, model_size = 1GB
        // Expected data: 2 * (2-1) / 2 * 1GB = 2 * 0.5 * 1GB = 1GB
        let model_size = 1_000_000_000u64;
        let result = generate_allreduce_events(2, model_size, 0, vec![]);
        
        let expected_data = model_size; // 2 * 1/2 * model_size = model_size
        
        // Worker 0: sends to 1, receives from 1
        let w0 = &result.worker_events[0];
        assert_eq!(w0[0].flow_send.as_ref().unwrap().dst_worker, 1);
        assert_eq!(w0[0].flow_send.as_ref().unwrap().size_bytes, expected_data);
        assert_eq!(w0[1].flow_receive.as_ref().unwrap().src_worker, 1);
        
        // Worker 1: sends to 0, receives from 0
        let w1 = &result.worker_events[1];
        assert_eq!(w1[0].flow_send.as_ref().unwrap().dst_worker, 0);
        assert_eq!(w1[1].flow_receive.as_ref().unwrap().src_worker, 0);
    }
    
    #[test]
    fn test_allreduce_dependencies() {
        let deps = vec![42, 43];
        let result = generate_allreduce_events(3, 1000, 100, deps.clone());
        
        for worker_events in &result.worker_events {
            for event in worker_events {
                assert_eq!(event.dependencies, deps);
            }
        }
    }
    
    #[test]
    fn test_allreduce_single_worker() {
        // Edge case: single worker has no one to communicate with
        let result = generate_allreduce_events(1, 1000, 0, vec![]);
        
        assert_eq!(result.worker_events.len(), 1);
        assert_eq!(result.events_per_worker, 0);
        assert!(result.worker_events[0].is_empty());
    }
}
