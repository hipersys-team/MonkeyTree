use crate::simulator::ml_job::MLJob;
use crate::simulator::ml_worker::{WorkerEvent, WorkerEventKind};
use crate::system_modules::cassini::types::{JobProfile, CommunicationPhase};
use std::collections::{HashMap, BTreeSet};

/// Profiles ML jobs to extract communication patterns for Cassini scheduling
pub struct JobProfiler {
    /// Angular resolution for bandwidth sampling (in degrees)
    angular_resolution_deg: f64,
    /// Baseline network throughput used to estimate flow durations (bytes/ms)
    baseline_throughput_bytes_per_us: u64,
}

impl JobProfiler {
    pub fn new() -> Self {
        Self {
            angular_resolution_deg: 5.0, // Default 5 degree resolution
            // Default to 1 Gbps: 1e9 bps = 125e6 B/s = 125 B/us
            baseline_throughput_bytes_per_us: 125,
        }
    }
    
    pub fn with_angular_resolution(mut self, resolution_deg: f64) -> Self {
        self.angular_resolution_deg = resolution_deg.max(0.1).min(90.0);
        self
    }
    
    /// Sets the baseline network throughput used to estimate flow durations
    /// Units: bytes per millisecond (B/ms)
    pub fn set_baseline_throughput_bytes_per_us(&mut self, throughput_b_per_us: u64) {
        // Avoid zero to prevent division by zero in estimations
        self.baseline_throughput_bytes_per_us = throughput_b_per_us.max(1);
    }
    
    /// Profiles a job to extract its communication pattern
    /// Analyzes the job's worker event templates to determine:
    /// - Total iteration time
    /// - Communication phases (Up/Down patterns)
    /// - Bandwidth demands
    pub fn profile_job(&self, job: &MLJob) -> Option<JobProfile> {
        if job.workers.is_empty() {
            return None;
        }
        
        // Get a representative worker to analyze the pattern
        // All workers should have similar patterns in most ML workloads
        let representative_worker = job.workers.values().next()?;
        let events = &representative_worker.template_events;
        
        if events.is_empty() {
            return None;
        }
        
        // Calculate total iteration time and extract communication phases
        let (iteration_time_us, communication_phases) = self.analyze_events(events);
        
        Some(JobProfile {
            job_id: job.id,
            iteration_time_us,
            communication_phases,
            num_workers: job.num_workers,
            name: job.name.clone(),
        })
    }
    
    /// Analyzes worker events to extract timing and communication patterns
    /// Accounts for the event DAG by computing earliest start/finish times and
    /// deriving a piecewise-constant bandwidth timeline from overlapping flows.
    fn analyze_events(&self, events: &[WorkerEvent]) -> (u64, Vec<CommunicationPhase>) {
        if events.is_empty() {
            return (1, vec![CommunicationPhase { duration_us: 1, bandwidth_demand: 0, is_up_phase: false }]);
        }

        // 1) Pre-compute per-event durations and static bandwidth demands
        #[derive(Clone, Copy)]
        struct EventTiming {
            duration_us: u64,
            bandwidth_demand: u64, // bytes/ms; 0 for compute
        }

        let mut event_timings: HashMap<usize, EventTiming> = HashMap::new();
        let mut event_lookup: HashMap<usize, &WorkerEvent> = HashMap::new();
        for ev in events {
            let timing = match ev.kind {
                WorkerEventKind::Compute => {
                    let d = ev.compute.as_ref().map(|c| c.duration_us).unwrap_or(0);
                    EventTiming { duration_us: d, bandwidth_demand: 0 }
                }
                WorkerEventKind::FlowSend => {
                    if let Some(fs) = &ev.flow_send {
                        let d = self.estimate_flow_duration(fs.size_bytes);
                        let bw = if d > 0 { fs.size_bytes / d } else { fs.size_bytes };
                        EventTiming { duration_us: d, bandwidth_demand: bw }
                    } else {
                        EventTiming { duration_us: 0, bandwidth_demand: 0 }
                    }
                }
                WorkerEventKind::FlowReceive => {
                    if let Some(fr) = &ev.flow_receive {
                        let d = self.estimate_flow_duration(fr.size_bytes);
                        let bw = if d > 0 { fr.size_bytes / d } else { fr.size_bytes };
                        EventTiming { duration_us: d, bandwidth_demand: bw }
                    } else {
                        EventTiming { duration_us: 0, bandwidth_demand: 0 }
                    }
                }
            };
            event_timings.insert(ev.id, timing);
            event_lookup.insert(ev.id, ev);
        }

        // 2) Compute earliest start/finish times via DP over the DAG
        #[derive(Clone, Copy, Default)]
        struct Times { start: u64, finish: u64 }
        let mut memo_times: HashMap<usize, Times> = HashMap::new();

        fn compute_times(
            ev_id: usize,
            event_lookup: &HashMap<usize, &WorkerEvent>,
            event_timings: &HashMap<usize, EventTiming>,
            memo_times: &mut HashMap<usize, Times>,
            visiting: &mut HashMap<usize, bool>,
        ) -> Times {
            if let Some(t) = memo_times.get(&ev_id) { return *t; }
            if *visiting.get(&ev_id).unwrap_or(&false) {
                // Cycle detected; treat as zero-latency to avoid infinite recursion
                return Times { start: 0, finish: event_timings.get(&ev_id).map(|t| t.duration_us).unwrap_or(0) };
            }
            visiting.insert(ev_id, true);
            let ev = match event_lookup.get(&ev_id) { Some(e) => *e, None => return Times { start: 0, finish: 0 } };
            let deps = &ev.dependencies;
            let mut earliest_start: u64 = 0;
            for dep_id in deps {
                let dep_times = compute_times(*dep_id, event_lookup, event_timings, memo_times, visiting);
                if dep_times.finish > earliest_start { earliest_start = dep_times.finish; }
            }
            let duration = event_timings.get(&ev_id).map(|t| t.duration_us).unwrap_or(0);
            let finish = earliest_start.saturating_add(duration);
            let result = Times { start: earliest_start, finish };
            visiting.insert(ev_id, false);
            memo_times.insert(ev_id, result);
            result
        }

        // Compute times for all events
        for ev in events {
            let mut visiting = HashMap::new();
            let _ = compute_times(ev.id, &event_lookup, &event_timings, &mut memo_times, &mut visiting);
        }

        // 3) Determine total iteration time = max finish over all events
        let mut total_time_us: u64 = 0;
        for t in memo_times.values() { if t.finish > total_time_us { total_time_us = t.finish; } }
        if total_time_us == 0 { total_time_us = 1; }

        // 4) Collect flow intervals with constant bandwidth demands
        #[derive(Clone, Copy)]
        struct FlowInterval { start: u64, end: u64, demand: u64 }
        let mut flow_intervals: Vec<FlowInterval> = Vec::new();
        for (ev_id, timing) in &event_timings {
            if timing.bandwidth_demand == 0 || timing.duration_us == 0 { continue; }
            if let Some(times) = memo_times.get(ev_id) {
                if times.finish > times.start {
                    flow_intervals.push(FlowInterval { start: times.start, end: times.finish, demand: timing.bandwidth_demand });
                }
            }
        }

        // 5) Build piecewise-constant bandwidth timeline using sweep over breakpoints
        let mut breakpoints: BTreeSet<u64> = BTreeSet::new();
        breakpoints.insert(0);
        breakpoints.insert(total_time_us);
        for iv in &flow_intervals {
            breakpoints.insert(iv.start);
            breakpoints.insert(iv.end);
        }
        let points: Vec<u64> = breakpoints.into_iter().collect();

        // Sort intervals by start and end to enable O(n) sweep
        let mut by_start = flow_intervals.clone();
        by_start.sort_by_key(|iv| iv.start);
        let mut by_end = flow_intervals.clone();
        by_end.sort_by_key(|iv| iv.end);

        let mut idx_start: usize = 0;
        let mut idx_end: usize = 0;
        let mut current_sum: u64 = 0;
        let mut communication_phases: Vec<CommunicationPhase> = Vec::new();

        for w in 0..(points.len().saturating_sub(1)) {
            let t = points[w];
            let t_next = points[w + 1];

            // Add intervals that start at t
            while idx_start < by_start.len() && by_start[idx_start].start == t {
                current_sum = current_sum.saturating_add(by_start[idx_start].demand);
                idx_start += 1;
            }
            // Remove intervals that end at t (they are not active in [t, t_next))
            while idx_end < by_end.len() && by_end[idx_end].end == t {
                current_sum = current_sum.saturating_sub(by_end[idx_end].demand);
                idx_end += 1;
            }

            let seg_len = t_next.saturating_sub(t);
            if seg_len == 0 { continue; }

            // Merge with previous if bandwidth is unchanged
            if let Some(last) = communication_phases.last_mut() {
                if last.bandwidth_demand == current_sum {
                    last.duration_us = last.duration_us.saturating_add(seg_len);
                    continue;
                }
            }

            communication_phases.push(CommunicationPhase {
                duration_us: seg_len,
                bandwidth_demand: current_sum,
                is_up_phase: current_sum > 0,
            });
        }

        if communication_phases.is_empty() {
            communication_phases.push(CommunicationPhase { duration_us: total_time_us, bandwidth_demand: 0, is_up_phase: false });
        }

        (total_time_us, communication_phases)
    }
    
    /// Estimates flow duration based on size
    /// This is a simplified model - real duration depends on network conditions
    fn estimate_flow_duration(&self, size_bytes: u64) -> u64 {
        // Duration in ms = size (bytes) / throughput (bytes/ms)
        size_bytes / self.baseline_throughput_bytes_per_us
    }
    
    /// Creates a default profile for jobs that couldn't be profiled
    /// Uses simple assumptions about iteration time and communication
    pub fn create_default_profile(&self, job: &MLJob, iteration_time_us: u64) -> JobProfile {
        // Create a simple two-phase pattern: compute then communicate
        let compute_phase = CommunicationPhase {
            duration_us: iteration_time_us * 7 / 10, // 70% compute
            bandwidth_demand: 0,
            is_up_phase: false,
        };
        
        let comm_phase = CommunicationPhase {
            duration_us: iteration_time_us * 3 / 10, // 30% communication
            bandwidth_demand: 1000, // Assume 1MB/ms during communication
            is_up_phase: true,
        };
        
        JobProfile {
            job_id: job.id,
            iteration_time_us,
            communication_phases: vec![compute_phase, comm_phase],
            num_workers: job.num_workers,
            name: job.name.clone(),
        }
    }
}

impl Default for JobProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::ml_worker::{WorkerEvent, WorkerEventKind, ComputeEvent, FlowSendEvent, FlowKind};
    use crate::simulator::ml_worker::{MLWorker, EventState};
    use crate::simulator::ml_job::MLJob;
    use std::collections::HashMap;
    
    #[test]
    fn test_simple_compute_communication_pattern() {
        let profiler = JobProfiler::new();
        let mut job = MLJob::new(1, 0, 2, 100);
        
        // Create a worker with a simple pattern: compute then send
        let mut worker = MLWorker::new(1, 0, 0, 100);
        
        // Add compute event
        let compute_event = WorkerEvent {
            id: 0,
            template_id: 0,
            kind: WorkerEventKind::Compute,
            compute: Some(ComputeEvent {
                duration_us: 100,
                name: Some("forward_pass".to_string()),
            }),
            flow_send: None,
            flow_receive: None,
            dependencies: vec![],
            state: EventState::Waiting,
        };
        
        // Add flow send event
        let send_event = WorkerEvent {
            id: 1,
            template_id: 1,
            kind: WorkerEventKind::FlowSend,
            compute: None,
            flow_send: Some(FlowSendEvent {
                dst_worker: 1,
                size_bytes: 1000000, // 1MB
                name: Some("gradient_sync".to_string()),
                flow_kind: FlowKind::Ring, // Gradient sync is typically a ring AllReduce
            }),
            flow_receive: None,
            dependencies: vec![0],
            state: EventState::Waiting,
        };
        
        worker.template_events = vec![compute_event, send_event];
        job.workers.insert(0, worker);
        
        let profile = profiler.profile_job(&job).unwrap();
        
        assert_eq!(profile.job_id, 1);
        assert_eq!(profile.num_workers, 2);
        assert!(profile.iteration_time_us > 100); // Should include compute + communication time
        assert_eq!(profile.communication_phases.len(), 2); // Compute phase + communication phase
        
        // First phase should be compute (Down phase)
        assert!(!profile.communication_phases[0].is_up_phase);
        assert_eq!(profile.communication_phases[0].bandwidth_demand, 0);
        
        // Second phase should be communication (Up phase)
        assert!(profile.communication_phases[1].is_up_phase);
        assert!(profile.communication_phases[1].bandwidth_demand > 0);
    }
    
    #[test]
    fn test_default_profile_creation() {
        let profiler = JobProfiler::new();
        let job = MLJob::new(1, 0, 4, 100);
        
        let profile = profiler.create_default_profile(&job, 1000);
        
        assert_eq!(profile.job_id, 1);
        assert_eq!(profile.iteration_time_us, 1000);
        assert_eq!(profile.num_workers, 4);
        assert_eq!(profile.communication_phases.len(), 2);
        
        // Should have compute phase followed by communication phase
        assert!(!profile.communication_phases[0].is_up_phase);
        assert!(profile.communication_phases[1].is_up_phase);
    }
}