//! Strided ring AllReduce collective communication pattern.
//!
//! A strided ring creates multiple independent rings where workers communicate
//! with peers that are `stride` positions apart rather than adjacent.
//!
//! With N workers and stride B (where B divides N evenly):
//! - Worker i sends to (i + B) % N
//! - Worker i receives from (i - B + N) % N
//! - This creates B independent rings, each with N/B workers
//! - Data size per worker = 2 * (ring_size - 1) / ring_size * model_size
//!
//! Example with N=8, B=2:
//! - Ring 0: workers 0 → 2 → 4 → 6 → 0 (ring_size = 4)
//! - Ring 1: workers 1 → 3 → 5 → 7 → 1 (ring_size = 4)
//! - Data per worker = 2 * 3/4 * model_size = 1.5 * model_size
//!
//! Use cases:
//! - Hierarchical AllReduce (intra-server vs inter-server communication)
//! - FSDP-style communication patterns
//! - Reducing cross-ToR traffic by keeping some rings local

use crate::simulator::WorkerEvent;
use crate::simulator::ml_worker::FlowKind;
use super::CollectiveEvents;

/// Generates strided ring AllReduce events for all workers.
///
/// The data size is calculated as: 2 * (ring_size - 1) / ring_size * model_size
/// where ring_size = num_workers / stride.
///
/// # Arguments
/// * `num_workers` - Number of workers participating (must be divisible by stride)
/// * `stride` - Distance between communicating workers (B in the formula)
/// * `model_size` - Model size in bytes (will be scaled by AllReduce formula)
/// * `start_event_id` - First event ID to use for generated events
/// * `dependencies` - Event IDs that must complete before the collective starts
///
/// # Returns
/// A `CollectiveEvents` struct containing the generated events for each worker.
///
/// # Panics
/// Panics if `num_workers` is not evenly divisible by `stride`.
///
/// # Event structure per worker
/// For a worker `i` with stride `B`:
/// - 1 FlowSend event: sends to worker (i + B) % N
/// - 1 FlowReceive event: receives from worker (i - B + N) % N
pub fn generate_strided_ring_events(
    num_workers: usize,
    stride: usize,
    model_size: u64,
    start_event_id: usize,
    dependencies: Vec<usize>,
) -> CollectiveEvents {
    assert!(
        stride > 0 && stride <= num_workers,
        "stride must be between 1 and num_workers"
    );
    assert!(
        num_workers % stride == 0,
        "num_workers ({}) must be divisible by stride ({})",
        num_workers,
        stride
    );

    let mut worker_events: Vec<Vec<WorkerEvent>> = Vec::with_capacity(num_workers);
    let mut completion_event_ids: Vec<Vec<usize>> = Vec::with_capacity(num_workers);

    // Each ring has ring_size = N / stride workers
    let ring_size = num_workers / stride;
    let events_per_worker = if ring_size > 1 { 2 } else { 0 };
    
    // Calculate actual data size using AllReduce formula for the ring size
    // data_size = 2 * (ring_size - 1) / ring_size * model_size
    let data_size = if ring_size > 1 {
        let rs = ring_size as f64;
        ((2.0 * (rs - 1.0) / rs) * model_size as f64) as u64
    } else {
        0
    };

    let mut current_event_id = start_event_id;

    for worker_id in 0..num_workers {
        let mut events = Vec::with_capacity(events_per_worker);
        let mut final_event_ids = Vec::new();

        if ring_size > 1 {
            // Next worker in strided ring (send destination)
            let next = (worker_id + stride) % num_workers;
            // Previous worker in strided ring (receive source)
            let prev = (worker_id + num_workers - stride) % num_workers;

            // Send to next worker in ring (FlowKind::Ring for priority routing)
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

            // Receive from previous worker in ring
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
    fn test_strided_ring_stride_1() {
        // stride=1 is equivalent to normal ring AllReduce
        let result = generate_strided_ring_events(4, 1, 1000, 0, vec![]);

        assert_eq!(result.worker_events.len(), 4);
        assert_eq!(result.events_per_worker, 2);

        // Worker 0: sends to 1, receives from 3
        let w0 = &result.worker_events[0];
        assert_eq!(w0[0].flow_send.as_ref().unwrap().dst_worker, 1);
        assert_eq!(w0[1].flow_receive.as_ref().unwrap().src_worker, 3);

        // Worker 1: sends to 2, receives from 0
        let w1 = &result.worker_events[1];
        assert_eq!(w1[0].flow_send.as_ref().unwrap().dst_worker, 2);
        assert_eq!(w1[1].flow_receive.as_ref().unwrap().src_worker, 0);
    }

    #[test]
    fn test_strided_ring_stride_2() {
        // 8 workers, stride=2: creates 2 rings of 4 workers each
        // Ring 0: 0 → 2 → 4 → 6 → 0
        // Ring 1: 1 → 3 → 5 → 7 → 1
        let result = generate_strided_ring_events(8, 2, 1000, 0, vec![]);

        assert_eq!(result.worker_events.len(), 8);
        assert_eq!(result.events_per_worker, 2);

        // Worker 0: sends to 2, receives from 6
        let w0 = &result.worker_events[0];
        assert_eq!(w0[0].flow_send.as_ref().unwrap().dst_worker, 2);
        assert_eq!(w0[1].flow_receive.as_ref().unwrap().src_worker, 6);

        // Worker 2: sends to 4, receives from 0
        let w2 = &result.worker_events[2];
        assert_eq!(w2[0].flow_send.as_ref().unwrap().dst_worker, 4);
        assert_eq!(w2[1].flow_receive.as_ref().unwrap().src_worker, 0);

        // Worker 1: sends to 3, receives from 7
        let w1 = &result.worker_events[1];
        assert_eq!(w1[0].flow_send.as_ref().unwrap().dst_worker, 3);
        assert_eq!(w1[1].flow_receive.as_ref().unwrap().src_worker, 7);

        // Worker 7: sends to 1, receives from 5
        let w7 = &result.worker_events[7];
        assert_eq!(w7[0].flow_send.as_ref().unwrap().dst_worker, 1);
        assert_eq!(w7[1].flow_receive.as_ref().unwrap().src_worker, 5);
    }

    #[test]
    fn test_strided_ring_stride_4() {
        // 8 workers, stride=4: creates 4 rings of 2 workers each
        // Ring 0: 0 → 4 → 0
        // Ring 1: 1 → 5 → 1
        // Ring 2: 2 → 6 → 2
        // Ring 3: 3 → 7 → 3
        let result = generate_strided_ring_events(8, 4, 1000, 0, vec![]);

        assert_eq!(result.worker_events.len(), 8);
        assert_eq!(result.events_per_worker, 2);

        // Worker 0: sends to 4, receives from 4
        let w0 = &result.worker_events[0];
        assert_eq!(w0[0].flow_send.as_ref().unwrap().dst_worker, 4);
        assert_eq!(w0[1].flow_receive.as_ref().unwrap().src_worker, 4);

        // Worker 4: sends to 0, receives from 0
        let w4 = &result.worker_events[4];
        assert_eq!(w4[0].flow_send.as_ref().unwrap().dst_worker, 0);
        assert_eq!(w4[1].flow_receive.as_ref().unwrap().src_worker, 0);
    }

    #[test]
    fn test_strided_ring_stride_equals_n() {
        // stride=N means ring size=1, no communication
        let result = generate_strided_ring_events(4, 4, 1000, 0, vec![]);

        assert_eq!(result.worker_events.len(), 4);
        assert_eq!(result.events_per_worker, 0);

        for events in &result.worker_events {
            assert!(events.is_empty());
        }
    }

    #[test]
    #[should_panic(expected = "must be divisible by stride")]
    fn test_strided_ring_invalid_stride() {
        // 7 workers is not divisible by 2
        generate_strided_ring_events(7, 2, 1000, 0, vec![]);
    }

    #[test]
    fn test_strided_ring_data_size_scaling() {
        // With 8 workers and stride=2, ring_size = 4
        // Data per worker = 2 * (4-1)/4 * model_size = 1.5 * model_size
        let model_size = 1_000_000_000u64;
        let result = generate_strided_ring_events(8, 2, model_size, 0, vec![]);
        
        let expected_data_size = ((2.0 * 3.0 / 4.0) * model_size as f64) as u64; // 1.5B

        for worker_events in &result.worker_events {
            for event in worker_events {
                if let Some(send) = &event.flow_send {
                    assert_eq!(send.size_bytes, expected_data_size);
                }
                if let Some(recv) = &event.flow_receive {
                    assert_eq!(recv.size_bytes, expected_data_size);
                }
            }
        }
    }
    
    #[test]
    fn test_strided_ring_stride_1_same_as_allreduce() {
        // With stride=1, ring_size = N, so data = 2 * (N-1)/N * model_size
        // This should match regular AllReduce
        let model_size = 1_000_000_000u64;
        let num_workers = 4;
        let result = generate_strided_ring_events(num_workers, 1, model_size, 0, vec![]);
        
        let n = num_workers as f64;
        let expected_data_size = ((2.0 * (n - 1.0) / n) * model_size as f64) as u64;

        let event = &result.worker_events[0][0];
        assert_eq!(event.flow_send.as_ref().unwrap().size_bytes, expected_data_size);
    }

    #[test]
    fn test_strided_ring_dependencies() {
        let deps = vec![100, 101];
        let result = generate_strided_ring_events(4, 2, 1000, 200, deps.clone());

        for worker_events in &result.worker_events {
            for event in worker_events {
                assert_eq!(event.dependencies, deps);
            }
        }
    }
}
