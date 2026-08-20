use std::collections::{VecDeque, BTreeMap, HashMap, HashSet};
use crate::simulator::flow_scheduler::{FlowScheduler, QueuedFlow};
use crate::simulator::ml_simulator::MLContext;
use crate::system_modules::cassini::types::{CassiniSchedule};
use crate::flow_scheduler::release_scheduler::{FlowKey};

/// Cassini-aware flow scheduler that enforces time-based flow scheduling
/// according to computed time shifts from Cassini optimization
#[derive(Debug)]
pub struct CassiniFlowScheduler {
    /// Baseline time where schedule time 0 begins
    baseline_us: u64,
    
    /// Current Cassini schedule
    current_schedule: Option<CassiniSchedule>,
    
    /// Per-key state for flow management
    keys: BTreeMap<FlowKey, PerKeyState>,
    
    /// Ready output buffer between polls
    ready: VecDeque<QueuedFlow>,
    
    /// Time shifts per job (for quick lookup)
    job_time_shifts: HashMap<usize, u64>,
    /// Period (iteration time) per job in ms
    job_periods: HashMap<usize, u64>,
}

#[derive(Debug, Default)]
struct PerKeyState {
    /// Backlog of queued flows for this key
    queue: VecDeque<QueuedFlow>,
    /// Sorted, deduplicated release offsets in [0, period_us) with Cassini time shifts
    offsets: Vec<u64>,
    /// Number of occurrences already consumed since last baseline
    next_idx: u64,
    /// Period for this key (job iteration time)
    period_us: u64,
}

impl CassiniFlowScheduler {
    pub fn new() -> Self {
        Self {
            baseline_us: 0,
            current_schedule: None,
            keys: BTreeMap::new(),
            ready: VecDeque::new(),
            job_time_shifts: HashMap::new(),
            job_periods: HashMap::new(),
        }
    }
    
    /// Applies a new Cassini schedule and rebases to start at `now_us`
    pub fn apply_cassini_schedule(&mut self, now_us: u64, schedule: CassiniSchedule) {
        self.baseline_us = now_us;
        
        // Extract time shifts for quick lookup
        self.job_time_shifts.clear();
        for (job_id, time_shift) in &schedule.time_shifts {
            self.job_time_shifts.insert(*job_id, time_shift.shift_us);
        }
        // Extract per-job periods
        self.job_periods.clear();
        for (job_id, period) in &schedule.job_periods {
            self.job_periods.insert(*job_id, *period);
        }
        
        // Update per-key offsets and periods directly
        let scheduled_jobs: HashSet<usize> = self.job_time_shifts.keys().copied().collect();
        
        // Set offsets for scheduled jobs
        for (job_id, shift) in &self.job_time_shifts {
            let period = self.job_periods.get(job_id).copied().unwrap_or(0);
            let offset = if period == 0 { 0 } else { shift % period };
            let key = FlowKey { job_id: *job_id, job_flow_idx: 0 };
            let st = self.keys.entry(key).or_default();
            st.offsets = vec![offset];
            st.next_idx = 0;
            st.period_us = period;
        }
        
        // Clear offsets for keys not in the new schedule
        for (key, st) in self.keys.iter_mut() {
            if !scheduled_jobs.contains(&key.job_id) {
                st.offsets.clear();
                st.next_idx = 0;
                st.period_us = 0;
            }
        }
        
        self.current_schedule = Some(schedule);
    }
    
    /// Gets the current Cassini schedule
    pub fn get_current_schedule(&self) -> Option<&CassiniSchedule> {
        self.current_schedule.as_ref()
    }
    
    /// Clears the current Cassini schedule, reverting to immediate release behavior.
    pub fn clear_cassini_schedule(&mut self) {
        self.current_schedule = None;
        self.job_time_shifts.clear();
        self.job_periods.clear();
        self.baseline_us = 0;
        // Flush any queued flows so they can be immediately released on next poll
        for (_key, st) in self.keys.iter_mut() {
            while let Some(flow) = st.queue.pop_front() {
                self.ready.push_back(flow);
            }
            st.offsets.clear();
            st.next_idx = 0;
            st.period_us = 0;
        }
    }
    
    /// Checks if a flow should be released based on Cassini scheduling
    fn should_release_flow(&self, flow: &QueuedFlow, now_us: u64) -> bool {
        // If no Cassini schedule is active, use immediate release
        if self.current_schedule.is_none() {
            return true;
        }
        
        // Check if this job has a time shift
        let time_shift = self.job_time_shifts.get(&flow.job_id).copied().unwrap_or(0);
        let period = self.job_periods.get(&flow.job_id).copied().unwrap_or(0);
        
        // Calculate the current position in the job's schedule period
        if period == 0 {
            return true;
        }
        
        let elapsed_since_baseline = now_us.saturating_sub(self.baseline_us);
        let current_position = elapsed_since_baseline % period;
        
        // Check if we're at or past the release time for this job
        let release_position = time_shift % period;
        
        // Allow release if we're at or past the scheduled time
        current_position >= release_position
    }
    
    // Helper methods from ReleaseFlowScheduler adapted for Cassini
    
    fn total_occurrences_until_with_period(offsets: &[u64], period_us: u64, delta_us: u64) -> u64 {
        if period_us == 0 || offsets.is_empty() { return 0; }
        let loops = delta_us / period_us;
        let within = delta_us % period_us;
        let in_loop = offsets.partition_point(|&o| o <= within) as u64;
        loops as u64 * offsets.len() as u64 + in_loop
    }
    
    fn next_global_wake_after(&self, now_us: u64) -> Option<u64> {
        let mut best: Option<u64> = None;
        for (_key, st) in self.keys.iter() {
            if st.queue.is_empty() || st.offsets.is_empty() { continue; }
            if st.period_us == 0 { continue; }
            let next_idx = st.next_idx;
            let loop_idx = next_idx / st.offsets.len() as u64;
            let in_loop_idx = (next_idx % st.offsets.len() as u64) as usize;
            let offset = st.offsets[in_loop_idx];
            let abs_time = self.baseline_us + loop_idx * st.period_us + offset;
            if abs_time > now_us {
                best = match best {
                    None => Some(abs_time),
                    Some(current_best) => Some(current_best.min(abs_time)),
                };
            }
        }
        best
    }
}

impl Default for CassiniFlowScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowScheduler for CassiniFlowScheduler {
    fn enqueue_flow(&mut self, now_us: u64, flow: QueuedFlow) -> Option<u64> {
        let key = FlowKey { job_id: flow.job_id, job_flow_idx: flow.job_flow_idx };
        
        // If we should release this flow immediately (based on Cassini schedule), add to ready queue
        if self.should_release_flow(&flow, now_us) {
            self.ready.push_back(flow);
            return None;
        }
        
        // Otherwise, enqueue for later release
        let st = self.keys.entry(key).or_default();
        st.queue.push_back(flow);
        
        // Return wake time for next potential release
        self.next_global_wake_after(now_us)
    }
    
    fn poll_ready(&mut self, now_us: u64) -> (Vec<QueuedFlow>, Option<u64>) {
        // First, return any flows that are immediately ready
        let mut out = Vec::new();
        while let Some(flow) = self.ready.pop_front() {
            out.push(flow);
        }
        
        // Then check scheduled flows
        let delta = now_us.saturating_sub(self.baseline_us);
        
        // Collect flows to process to avoid borrowing issues
        let mut flows_to_process = Vec::new();
        
        for (key, st) in self.keys.iter_mut() {
            if st.queue.is_empty() || st.offsets.is_empty() { continue; }
            if st.period_us == 0 { continue; }
            
            // Calculate expected count using separate function to avoid borrow issues
            let expected_count = CassiniFlowScheduler::total_occurrences_until_with_period(&st.offsets, st.period_us, delta);
            
            while st.next_idx < expected_count && !st.queue.is_empty() {
                if let Some(flow) = st.queue.pop_front() {
                    flows_to_process.push((flow, *key));
                }
                st.next_idx += 1;
            }
        }
        
        // Process collected flows
        for (flow, key) in flows_to_process {
            if self.should_release_flow(&flow, now_us) {
                out.push(flow);
            } else {
                // Flow not ready yet, put it back
                if let Some(st) = self.keys.get_mut(&key) {
                    st.queue.push_front(flow);
                    st.next_idx = st.next_idx.saturating_sub(1);
                }
            }
        }
        
        let next_wake = self.next_global_wake_after(now_us);
        (out, next_wake)
    }
    
    fn is_idle(&self) -> bool {
        self.ready.is_empty() && 
        self.keys.values().all(|st| st.queue.is_empty())
    }

    fn on_migration_begin(&mut self, _now_us: u64, ctx: &MLContext, affected_jobs: &[crate::simulator::ml_job::JobId]) {
        if affected_jobs.is_empty() { return; }
        let affected: std::collections::HashSet<_> = affected_jobs.iter().copied().collect();
        let placements = ctx.placements.borrow();
        // remap ready buffer
        for f in self.ready.iter_mut() {
            if !affected.contains(&f.job_id) { continue; }
            if let Some(w2h) = placements.get(&f.job_id) {
                if let (Some(&src_h), Some(&dst_h)) = (w2h.get(&f.src_worker), w2h.get(&f.dst_worker)) {
                    f.src_host = src_h;
                    f.dst_host = dst_h;
                }
            }
        }
        // remap per-key queues
        for (key, st) in self.keys.iter_mut() {
            if !affected.contains(&key.job_id) { continue; }
            if let Some(w2h) = placements.get(&key.job_id) {
                for f in st.queue.iter_mut() {
                    if let (Some(&src_h), Some(&dst_h)) = (w2h.get(&f.src_worker), w2h.get(&f.dst_worker)) {
                        f.src_host = src_h;
                        f.dst_host = dst_h;
                    }
                }
            }
        }
    }

    fn on_migration_end(&mut self, now_us: u64) -> Option<u64> { self.next_global_wake_after(now_us) }
}