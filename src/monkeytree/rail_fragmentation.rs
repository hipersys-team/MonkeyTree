//! Fragmentation analysis for rail-optimized topologies.
//!
//! For rail topologies, fragmentation is computed **per-pod** rather than per-ToR.
//! A segment is "fragmented" if its workers span more than one pod.
//! Per-pod fragmentation = weighted count of fragmented segments present in that pod.

use std::collections::{HashMap, HashSet};

use crate::simulator::ml_simulator::MLContext;
use crate::simulator::ml_job::JobId;
use crate::simulator::ml_worker::WorkerId;
use crate::rail::{RailTree, RailTreeRouter};

use super::fragmentation::{
    SegmentId, JobSegment, SegmentFragmentation, ToRSegmentStats,
    build_segments_from_context, print_segment_fragmentation_summary,
};

/// Compute pod-level fragmentation for a rail-optimized topology.
///
/// This mirrors `compute_segment_fragmentation` but groups by pod instead of ToR.
/// The returned `SegmentFragmentation` uses pod indices in place of ToR indices
/// so the ILP solver can treat pods as the unit of placement.
pub fn compute_pod_fragmentation<R: RailTreeRouter>(
    ctx: &MLContext,
    topo: &RailTree<R>,
) -> SegmentFragmentation {
    let gpus_per_pod = topo.blocks_per_pod * topo.block_size;
    let num_pods = topo.num_pods;
    let placements = ctx.placements.borrow();

    let segments = build_segments_from_context(ctx);

    let mut worker_to_host: HashMap<(JobId, WorkerId), usize> = HashMap::new();
    for (job_id, worker_hosts) in placements.iter() {
        for (worker_id, &host) in worker_hosts.iter() {
            worker_to_host.insert((*job_id, *worker_id), host);
        }
    }

    // Identify fragmented segments (span multiple pods)
    let mut fragmented_segments = HashSet::new();
    for segment in &segments {
        let pods: HashSet<usize> = segment.worker_ids.iter()
            .filter_map(|w| worker_to_host.get(&(segment.id.job_id, *w)))
            .map(|host| host / gpus_per_pod)
            .collect();
        if pods.len() > 1 {
            fragmented_segments.insert(segment.id);
        }
    }

    let segment_by_id: HashMap<SegmentId, &JobSegment> = segments.iter()
        .map(|s| (s.id, s))
        .collect();

    // Compute per-pod stats (stored in the ToRSegmentStats struct, reusing the field names)
    let mut per_pod = Vec::with_capacity(num_pods);
    for pod_idx in 0..num_pods {
        let pod_host_start = pod_idx * gpus_per_pod;
        let pod_host_end = (pod_idx + 1) * gpus_per_pod;

        let mut segments_present = HashSet::new();
        let mut segment_worker_counts = HashMap::new();

        for segment in &segments {
            let workers_in_pod: usize = segment.worker_ids.iter()
                .filter(|w| {
                    worker_to_host.get(&(segment.id.job_id, **w))
                        .map(|&h| h >= pod_host_start && h < pod_host_end)
                        .unwrap_or(false)
                })
                .count();
            if workers_in_pod > 0 {
                segments_present.insert(segment.id);
                segment_worker_counts.insert(segment.id, workers_in_pod);
            }
        }

        let fragmented_count: usize = segments_present.iter()
            .filter(|s| fragmented_segments.contains(s))
            .filter_map(|s| segment_by_id.get(s))
            .map(|s| s.ring_count)
            .sum();

        per_pod.push(ToRSegmentStats {
            tor_index: pod_idx,
            segments_present,
            fragmented_segment_count: fragmented_count,
            segment_worker_counts,
        });
    }

    let max_fragmentation = per_pod.iter()
        .map(|s| s.fragmented_segment_count)
        .max()
        .unwrap_or(0);

    SegmentFragmentation {
        per_tor: per_pod,
        fragmented_segments,
        segments,
        max_fragmentation,
    }
}

/// Print pod-level fragmentation summary.
pub fn print_pod_fragmentation_summary(frag: &SegmentFragmentation, now_us: u64) {
    // Reuse the existing printer -- it says "ToR" in output but the indices are pods
    print_segment_fragmentation_summary(frag, now_us);
}
