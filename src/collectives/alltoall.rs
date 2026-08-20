//! All-to-All collective communication pattern.
//!
//! In All-to-All, each worker sends a fixed amount of data to every other worker
//! simultaneously. This is commonly used in:
//! - Transformer attention layers (exchanging key/value tensors)
//! - Expert parallelism (MoE routing)
//! - Distributed matrix transpose operations

use crate::simulator::WorkerEvent;
use crate::simulator::ml_worker::FlowKind;
use super::CollectiveEvents;

/// Generates All-to-All collective events for all workers.
///
/// Each worker sends `data_size_per_pair` bytes to every other worker simultaneously,
/// and receives the same amount from every other worker.
///
/// # Arguments
/// * `num_workers` - Number of workers participating in the collective
/// * `data_size_per_pair` - Bytes each worker sends to each other worker
/// * `start_event_id` - First event ID to use for generated events
/// * `dependencies` - Event IDs that must complete before the collective starts
///                    (same dependencies applied to all workers)
///
/// # Returns
/// A `CollectiveEvents` struct containing the generated events for each worker.
///
/// # Event structure per worker
/// For a worker `i` in an n-worker collective:
/// - (n-1) FlowSend events: sends to workers 0..i-1, i+1..n-1
/// - (n-1) FlowReceive events: receives from workers 0..i-1, i+1..n-1
/// - All sends and receives depend on `dependencies` and execute in parallel
pub fn generate_alltoall_events(
    num_workers: usize,
    data_size_per_pair: u64,
    start_event_id: usize,
    dependencies: Vec<usize>,
) -> CollectiveEvents {
    let mut worker_events: Vec<Vec<WorkerEvent>> = Vec::with_capacity(num_workers);
    let mut completion_event_ids: Vec<Vec<usize>> = Vec::with_capacity(num_workers);
    
    // Calculate how many events each worker needs:
    // - (n-1) sends + (n-1) receives = 2*(n-1) events per worker
    let events_per_worker = if num_workers > 1 { 2 * (num_workers - 1) } else { 0 };
    
    let mut current_event_id = start_event_id;
    
    for worker_id in 0..num_workers {
        let mut events = Vec::with_capacity(events_per_worker);
        let mut final_event_ids = Vec::new();
        
        // Generate sends to all other workers
        for dst_worker in 0..num_workers {
            if dst_worker == worker_id {
                continue; // Don't send to self
            }
            
            let event_id = current_event_id;
            current_event_id += 1;
            
            events.push(WorkerEvent::new_flow_send_with_kind(
                event_id,
                dst_worker,
                data_size_per_pair,
                dependencies.clone(),
                FlowKind::AllToAll,
            ));
            final_event_ids.push(event_id);
        }
        
        // Generate receives from all other workers
        for src_worker in 0..num_workers {
            if src_worker == worker_id {
                continue; // Don't receive from self
            }
            
            let event_id = current_event_id;
            current_event_id += 1;
            
            events.push(WorkerEvent::new_flow_receive_with_kind(
                event_id,
                src_worker,
                data_size_per_pair,
                dependencies.clone(),
                FlowKind::AllToAll,
            ));
            final_event_ids.push(event_id);
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
    fn test_alltoall_event_count() {
        // 4 workers: each sends to 3 others, receives from 3 others = 6 events each
        let result = generate_alltoall_events(4, 1000, 0, vec![]);
        
        assert_eq!(result.worker_events.len(), 4);
        assert_eq!(result.events_per_worker, 6);
        
        for events in &result.worker_events {
            assert_eq!(events.len(), 6);
        }
    }
    
    #[test]
    fn test_alltoall_two_workers() {
        // 2 workers: each sends to 1, receives from 1 = 2 events each
        let result = generate_alltoall_events(2, 5_000_000_000, 0, vec![]);
        
        assert_eq!(result.worker_events.len(), 2);
        assert_eq!(result.events_per_worker, 2);
        
        // Worker 0 should send to worker 1 and receive from worker 1
        let w0_events = &result.worker_events[0];
        assert_eq!(w0_events.len(), 2);
        assert!(w0_events[0].flow_send.as_ref().unwrap().dst_worker == 1);
        assert!(w0_events[1].flow_receive.as_ref().unwrap().src_worker == 1);
        
        // Worker 1 should send to worker 0 and receive from worker 0
        let w1_events = &result.worker_events[1];
        assert_eq!(w1_events.len(), 2);
        assert!(w1_events[0].flow_send.as_ref().unwrap().dst_worker == 0);
        assert!(w1_events[1].flow_receive.as_ref().unwrap().src_worker == 0);
    }
    
    #[test]
    fn test_alltoall_dependencies() {
        // Events should inherit the provided dependencies
        let deps = vec![42, 43];
        let result = generate_alltoall_events(3, 1000, 100, deps.clone());
        
        for worker_events in &result.worker_events {
            for event in worker_events {
                assert_eq!(event.dependencies, deps);
            }
        }
    }
    
    #[test]
    fn test_alltoall_event_ids_sequential() {
        let result = generate_alltoall_events(3, 1000, 10, vec![]);
        
        // Total events: 3 workers * 4 events each = 12 events
        // IDs should be 10, 11, 12, ..., 21
        let mut all_ids: Vec<usize> = result.worker_events
            .iter()
            .flat_map(|events| events.iter().map(|e| e.id))
            .collect();
        all_ids.sort();
        
        let expected: Vec<usize> = (10..22).collect();
        assert_eq!(all_ids, expected);
    }
    
    #[test]
    fn test_alltoall_single_worker() {
        // Edge case: single worker has no one to communicate with
        let result = generate_alltoall_events(1, 1000, 0, vec![]);
        
        assert_eq!(result.worker_events.len(), 1);
        assert_eq!(result.events_per_worker, 0);
        assert!(result.worker_events[0].is_empty());
    }
}
