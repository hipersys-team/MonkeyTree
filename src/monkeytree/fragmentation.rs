//! Fragmentation analysis utilities for MonkeyTree
//!
//! A **segment** is the unit of fragmentation analysis:
//! - For non-pipeline jobs: 1 segment = the whole job
//! - For pipeline jobs: num_stages segments, one per pipeline stage
//!
//! A segment is **fragmented** if its workers span more than one ToR (leaf switch).
//! Per-ToR fragmentation = weighted count of fragmented segments with workers on that ToR,
//! where each segment's contribution is weighted by its ring_count (e.g., DP replicas).

use std::collections::{HashMap, HashSet};
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::ml_job::JobId;
use crate::simulator::ml_worker::WorkerId;
use crate::spine::{SpineTree, SpineTreeRouter};

/// Identifier for a job segment.
/// For non-pipeline jobs, stage is None.
/// For pipeline jobs, stage is Some(stage_index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentId {
    pub job_id: JobId,
    pub stage: Option<usize>,
}

impl SegmentId {
    /// Create a segment ID for a non-pipeline job (whole job = one segment)
    pub fn whole_job(job_id: JobId) -> Self {
        Self { job_id, stage: None }
    }
    
    /// Create a segment ID for a pipeline stage
    pub fn pipeline_stage(job_id: JobId, stage: usize) -> Self {
        Self { job_id, stage: Some(stage) }
    }
}

/// A job segment with its properties for fragmentation/ILP analysis.
#[derive(Debug, Clone)]
pub struct JobSegment {
    pub id: SegmentId,
    /// Worker IDs belonging to this segment
    pub worker_ids: Vec<WorkerId>,
    /// Number of independent rings in this segment
    pub ring_count: usize,
}

impl JobSegment {
    /// Get the number of workers in this segment
    pub fn num_workers(&self) -> usize {
        self.worker_ids.len()
    }
}

/// Build segments from active jobs.
/// For pipeline jobs, creates one segment per stage.
/// For non-pipeline jobs, creates one segment for the whole job.
pub fn build_segments_from_context(ctx: &MLContext) -> Vec<JobSegment> {
    let active_jobs = ctx.active_jobs.borrow();
    let placements = ctx.placements.borrow();
    let mut segments = Vec::new();
    
    for (job_id, info) in active_jobs.iter() {
        if let Some(ref pipeline_info) = info.pipeline_stages {
            // Pipeline job: create one segment per stage
            for stage in 0..pipeline_info.num_stages {
                let worker_range = pipeline_info.stage_workers(stage);
                // Filter to workers that are actually placed (in case of partial placement)
                let worker_ids: Vec<WorkerId> = if let Some(worker_hosts) = placements.get(job_id) {
                    worker_range.filter(|w| worker_hosts.contains_key(w)).collect()
                } else {
                    vec![]
                };
                
                segments.push(JobSegment {
                    id: SegmentId::pipeline_stage(*job_id, stage),
                    worker_ids,
                    ring_count: pipeline_info.rings_per_stage,
                });
            }
        } else {
            // Non-pipeline job: one segment for the whole job
            let worker_ids: Vec<WorkerId> = if let Some(worker_hosts) = placements.get(job_id) {
                worker_hosts.keys().copied().collect()
            } else {
                vec![]
            };
            
            segments.push(JobSegment {
                id: SegmentId::whole_job(*job_id),
                worker_ids,
                ring_count: info.ring_count,
            });
        }
    }
    
    segments
}

/// Per-ToR fragmentation statistics (segment-based)
#[derive(Debug, Clone)]
pub struct ToRSegmentStats {
    pub tor_index: usize,
    /// Set of segment IDs with workers on this ToR
    pub segments_present: HashSet<SegmentId>,
    /// Weighted count of fragmented segments on this ToR (sum of ring_counts)
    pub fragmented_segment_count: usize,
    /// segment_id -> worker count on this ToR
    pub segment_worker_counts: HashMap<SegmentId, usize>,
}

/// Cluster-wide fragmentation snapshot (segment-based)
#[derive(Debug, Clone)]
pub struct SegmentFragmentation {
    pub per_tor: Vec<ToRSegmentStats>,
    /// Segments that span multiple ToRs
    pub fragmented_segments: HashSet<SegmentId>,
    /// All segments in the cluster
    pub segments: Vec<JobSegment>,
    /// Maximum weighted fragmentation on any single ToR (sum of ring_counts)
    pub max_fragmentation: usize,
}

/// Compute cluster-wide segment-based fragmentation statistics.
/// For pipeline jobs, each stage is a separate segment.
/// For non-pipeline jobs, the whole job is one segment.
pub fn compute_segment_fragmentation<R: SpineTreeRouter>(
    ctx: &MLContext,
    topo: &SpineTree<R>,
) -> SegmentFragmentation {
    let hosts_per_leaf = topo.hosts_per_leaf;
    let num_tors = topo.num_leaves;
    let placements = ctx.placements.borrow();
    
    // Build segments from active jobs
    let segments = build_segments_from_context(ctx);
    
    // Build a map from (job_id, worker_id) -> host for quick lookup
    let mut worker_to_host: HashMap<(JobId, WorkerId), usize> = HashMap::new();
    for (job_id, worker_hosts) in placements.iter() {
        for (worker_id, &host) in worker_hosts.iter() {
            worker_to_host.insert((*job_id, *worker_id), host);
        }
    }
    
    // 1. Identify fragmented segments (span multiple ToRs)
    let mut fragmented_segments = HashSet::new();
    for segment in &segments {
        let tors: HashSet<usize> = segment.worker_ids.iter()
            .filter_map(|w| worker_to_host.get(&(segment.id.job_id, *w)))
            .map(|host| host / hosts_per_leaf)
            .collect();
        if tors.len() > 1 {
            fragmented_segments.insert(segment.id);
        }
    }
    
    // 2. Build segment lookup by ID
    let segment_by_id: HashMap<SegmentId, &JobSegment> = segments.iter()
        .map(|s| (s.id, s))
        .collect();
    
    // 3. Compute per-ToR stats with ring_count weighting
    let mut per_tor = Vec::with_capacity(num_tors);
    for tor_idx in 0..num_tors {
        let tor_host_start = tor_idx * hosts_per_leaf;
        let tor_host_end = (tor_idx + 1) * hosts_per_leaf;
        
        let mut segments_present = HashSet::new();
        let mut segment_worker_counts = HashMap::new();
        
        for segment in &segments {
            let workers_on_tor: usize = segment.worker_ids.iter()
                .filter(|w| {
                    worker_to_host.get(&(segment.id.job_id, **w))
                        .map(|&h| h >= tor_host_start && h < tor_host_end)
                        .unwrap_or(false)
                })
                .count();
            if workers_on_tor > 0 {
                segments_present.insert(segment.id);
                segment_worker_counts.insert(segment.id, workers_on_tor);
            }
        }
        
        // Weight each fragmented segment by its ring_count
        let fragmented_count: usize = segments_present.iter()
            .filter(|s| fragmented_segments.contains(s))
            .filter_map(|s| segment_by_id.get(s))
            .map(|s| s.ring_count)
            .sum();
            
        per_tor.push(ToRSegmentStats {
            tor_index: tor_idx,
            segments_present,
            fragmented_segment_count: fragmented_count,
            segment_worker_counts,
        });
    }
    
    let max_fragmentation = per_tor.iter()
        .map(|s| s.fragmented_segment_count)
        .max()
        .unwrap_or(0);
    
    SegmentFragmentation {
        per_tor,
        fragmented_segments,
        segments,
        max_fragmentation,
    }
}

// ============================================================================
// Legacy job-based fragmentation (for backward compatibility)
// ============================================================================

/// Per-ToR fragmentation statistics (legacy, job-based)
#[deprecated(since = "0.2.0", note = "Use ToRSegmentStats for pipeline-aware fragmentation")]
#[derive(Debug, Clone)]
pub struct ToRFragmentationStats {
    pub tor_index: usize,
    /// Set of job IDs with workers on this ToR
    pub jobs_present: HashSet<JobId>,
    /// Weighted count of fragmented jobs on this ToR (sum of ring_counts)
    pub fragmented_job_count: usize,
    /// job_id -> worker count on this ToR
    pub job_worker_counts: HashMap<JobId, usize>,
}

/// Cluster-wide fragmentation snapshot (legacy, job-based)
#[deprecated(since = "0.2.0", note = "Use SegmentFragmentation for pipeline-aware fragmentation")]
#[derive(Debug, Clone)]
#[allow(deprecated)]
pub struct ClusterFragmentation {
    pub per_tor: Vec<ToRFragmentationStats>,
    /// Jobs that span multiple ToRs
    pub fragmented_jobs: HashSet<JobId>,
    /// Maximum weighted fragmentation on any single ToR (sum of ring_counts)
    pub max_fragmentation: usize,
}

/// Compute cluster-wide fragmentation statistics (legacy, job-based).
/// For pipeline-aware fragmentation, use compute_segment_fragmentation instead.
#[deprecated(since = "0.2.0", note = "Use compute_segment_fragmentation for pipeline-aware fragmentation")]
#[allow(deprecated)]
pub fn compute_cluster_fragmentation<R: SpineTreeRouter>(
    ctx: &MLContext,
    topo: &SpineTree<R>,
) -> ClusterFragmentation {
    let hosts_per_leaf = topo.hosts_per_leaf;
    let num_tors = topo.num_leaves;
    let placements = ctx.placements.borrow();
    let active_jobs = ctx.active_jobs.borrow();
    
    // Helper to get ring_count for a job (defaults to 1 if not found)
    let get_ring_count = |job_id: &JobId| -> usize {
        active_jobs.get(job_id)
            .map(|info| info.ring_count)
            .unwrap_or(1)
    };
    
    // 1. Identify fragmented jobs (span multiple ToRs)
    let mut fragmented_jobs = HashSet::new();
    for (job_id, worker_to_host) in placements.iter() {
        let tors: HashSet<usize> = worker_to_host.values()
            .map(|host| host / hosts_per_leaf)
            .collect();
        if tors.len() > 1 {
            fragmented_jobs.insert(*job_id);
        }
    }
    
    // 2. Compute per-ToR stats with ring_count weighting
    let mut per_tor = Vec::with_capacity(num_tors);
    for tor_idx in 0..num_tors {
        let tor_host_start = tor_idx * hosts_per_leaf;
        let tor_host_end = (tor_idx + 1) * hosts_per_leaf;
        
        let mut jobs_present = HashSet::new();
        let mut job_worker_counts = HashMap::new();
        
        for (job_id, worker_to_host) in placements.iter() {
            let workers_on_tor: usize = worker_to_host.values()
                .filter(|&&host| host >= tor_host_start && host < tor_host_end)
                .count();
            if workers_on_tor > 0 {
                jobs_present.insert(*job_id);
                job_worker_counts.insert(*job_id, workers_on_tor);
            }
        }
        
        // Weight each fragmented job by its ring_count
        let fragmented_count: usize = jobs_present.iter()
            .filter(|j| fragmented_jobs.contains(j))
            .map(|j| get_ring_count(j))
            .sum();
            
        per_tor.push(ToRFragmentationStats {
            tor_index: tor_idx,
            jobs_present,
            fragmented_job_count: fragmented_count,
            job_worker_counts,
        });
    }
    
    let max_fragmentation = per_tor.iter()
        .map(|s| s.fragmented_job_count)
        .max()
        .unwrap_or(0);
    
    ClusterFragmentation {
        per_tor,
        fragmented_jobs,
        max_fragmentation,
    }
}

/// Print a readable summary of segment-based fragmentation state.
/// Shows per-ToR fragmentation levels and which fragmented segments are on each ToR.
pub fn print_segment_fragmentation_summary(frag: &SegmentFragmentation, now_us: u64) {
    // Build segment lookup
    let segment_by_id: std::collections::HashMap<SegmentId, &JobSegment> = frag.segments.iter()
        .map(|s| (s.id, s))
        .collect();
    
    println!("{} FragmentationSummary (segment-based)", now_us);
    println!("  Fragmented segments: {:?}", 
        frag.fragmented_segments.iter().collect::<Vec<_>>());
    println!("  Max fragmentation (weighted): {}", frag.max_fragmentation);
    println!("  Per-ToR breakdown:");
    
    for tor_stats in &frag.per_tor {
        // Get all segments on this ToR with their worker counts
        // Mark fragmented segments with * and show their ring counts
        let mut segments_on_tor: Vec<String> = tor_stats.segments_present.iter()
            .map(|seg_id| {
                let workers = tor_stats.segment_worker_counts.get(seg_id).copied().unwrap_or(0);
                let is_fragmented = frag.fragmented_segments.contains(seg_id);
                let ring_count = segment_by_id.get(seg_id)
                    .map(|s| s.ring_count)
                    .unwrap_or(1);
                
                let seg_label = match seg_id.stage {
                    Some(stage) => format!("j{}s{}", seg_id.job_id, stage),
                    None => format!("j{}", seg_id.job_id),
                };
                
                if is_fragmented {
                    format!("{}*(w={},r={})", seg_label, workers, ring_count)
                } else {
                    format!("{}(w={})", seg_label, workers)
                }
            })
            .collect();
        // Sort for consistent output
        segments_on_tor.sort();
        
        println!("    ToR {}: frag_weight={}, segments=[{}]",
            tor_stats.tor_index,
            tor_stats.fragmented_segment_count,
            segments_on_tor.join(", "));
    }
}

/// Print a readable summary of cluster fragmentation state.
/// Shows per-ToR fragmentation levels and which fragmented jobs are on each ToR.
#[deprecated(since = "0.2.0", note = "Use print_segment_fragmentation_summary instead")]
#[allow(deprecated)]
pub fn print_fragmentation_summary(frag: &ClusterFragmentation, ctx: &MLContext, now_us: u64) {
    let active_jobs = ctx.active_jobs.borrow();
    
    println!("{} FragmentationSummary", now_us);
    println!("  Fragmented jobs: {:?}", 
        frag.fragmented_jobs.iter().copied().collect::<Vec<_>>());
    println!("  Max fragmentation (weighted): {}", frag.max_fragmentation);
    println!("  Per-ToR breakdown:");
    
    for tor_stats in &frag.per_tor {
        // Get all jobs on this ToR with their worker counts
        // Mark fragmented jobs with * and show their ring counts
        let mut jobs_on_tor: Vec<(JobId, String)> = tor_stats.jobs_present.iter()
            .map(|j| {
                let workers = tor_stats.job_worker_counts.get(j).copied().unwrap_or(0);
                let is_fragmented = frag.fragmented_jobs.contains(j);
                if is_fragmented {
                    let ring_count = active_jobs.get(j)
                        .map(|info| info.ring_count)
                        .unwrap_or(1);
                    (*j, format!("{}*(w={},r={})", j, workers, ring_count))
                } else {
                    (*j, format!("{}(w={})", j, workers))
                }
            })
            .collect();
        // Sort by job ID for consistent output
        jobs_on_tor.sort_by_key(|(id, _)| *id);
        let jobs_str: Vec<String> = jobs_on_tor.into_iter().map(|(_, s)| s).collect();
        
        println!("    ToR {}: frag_weight={}, jobs=[{}]",
            tor_stats.tor_index,
            tor_stats.fragmented_job_count,
            jobs_str.join(", "));
    }
}
