use std::collections::{VecDeque, BTreeMap, HashMap};

use crate::simulator::flow_scheduler::{FlowScheduler, QueuedFlow};
use crate::simulator::ml_simulator::MLContext;

/// Key used to group flows across iterations in a stable manner within a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FlowKey {
    pub job_id: crate::simulator::ml_job::JobId,
    pub job_flow_idx: usize,
}

/// One schedule entry representing a flow occurrence at a given offset.
#[derive(Debug, Clone)]
pub struct FlowReleaseSpec {
    pub job_id: crate::simulator::ml_job::JobId,
    pub job_flow_idx: usize,
    pub offset_us: u64,
}

/// Complete schedule that repeats every `period_us`.
#[derive(Debug, Clone)]
pub struct FlowReleaseSchedule {
    pub version: u64,
    pub period_us: u64,
    pub entries: Vec<FlowReleaseSpec>,
}

#[derive(Debug, Default)]
struct PerKeyState {
    /// Backlog of queued flows for this key.
    queue: VecDeque<QueuedFlow>,
    /// Sorted, deduplicated release offsets in [0, period_us).
    offsets: Vec<u64>,
    /// Number of occurrences already consumed since last baseline.
    next_idx: u64,
}

/// Looping, system-driven release scheduler with rebase support.
#[derive(Debug, Default)]
pub struct ReleaseFlowScheduler {
    /// Baseline time where schedule time 0 begins.
    baseline_us: u64,
    /// Current schedule period.
    period_us: u64,
    /// Per-key state (ordered for determinism by FlowKey).
    keys: BTreeMap<FlowKey, PerKeyState>,
    /// Ready output buffer between polls.
    ready: VecDeque<QueuedFlow>,
}

impl ReleaseFlowScheduler {
    pub fn new() -> Self { Self::default() }

    /// Replace the current schedule and rebase to start at `now_us`.
    /// Keeps accumulated queues (frozen backlog) and maps them to the new schedule.
    pub fn rebase_to_new_schedule(&mut self, now_us: u64, schedule: FlowReleaseSchedule) {
        self.baseline_us = now_us;
        self.period_us = schedule.period_us;
        // Reset offsets and indices while keeping existing queues intact.
        let mut grouped: HashMap<FlowKey, Vec<u64>> = HashMap::new();
        for e in schedule.entries {
            let key = FlowKey { job_id: e.job_id, job_flow_idx: e.job_flow_idx };
            grouped.entry(key).or_default().push(e.offset_us % self.period_us);
        }
        for (key, offsets) in grouped.into_iter() {
            let mut offsets = offsets;
            offsets.sort_unstable();
            offsets.dedup();
            let st = self.keys.entry(key).or_default();
            st.offsets = offsets;
            st.next_idx = 0;
        }
        // Ensure keys that had backlog but are missing from new schedule still exist with empty offsets
        for (_key, st) in self.keys.iter_mut() {
            if st.offsets.is_empty() {
                st.next_idx = 0;
            }
        }
    }

    fn total_occurrences_until(&self, offsets: &[u64], delta_us: u64) -> u64 {
        if self.period_us == 0 || offsets.is_empty() { return 0; }
        let loops = delta_us / self.period_us;
        let within = delta_us % self.period_us;
        let in_loop = offsets.partition_point(|&o| o <= within) as u64;
        loops as u64 * offsets.len() as u64 + in_loop
    }

    /// Compute the next time (in ms) when the scheduler should be polled again,
    /// based on the earliest next scheduled release across all keys that still
    /// have backlog. Returns None if there is no upcoming release or no backlog.
    fn next_global_wake_after(&self, now_us: u64) -> Option<u64> {
        if self.period_us == 0 { return None; }
        let mut best: Option<u64> = None;
        for (_key, st) in self.keys.iter() {
            if st.queue.is_empty() || st.offsets.is_empty() { continue; }
            let next_idx = st.next_idx;
            let loop_idx = next_idx / st.offsets.len() as u64;
            let in_loop_idx = (next_idx % st.offsets.len() as u64) as usize;
            let candidate_offset = st.offsets[in_loop_idx] + loop_idx * self.period_us;
            let wake_time = self.baseline_us + candidate_offset;
            best = match best { Some(b) => Some(b.min(wake_time)), None => Some(wake_time) };
        }
        best.filter(|t| *t > now_us)
    }

    /// Compute the next wake time across all keys, optionally excluding one key.
    fn next_global_wake_after_excluding(&self, now_us: u64, exclude: Option<FlowKey>) -> Option<u64> {
        if self.period_us == 0 { return None; }
        let mut best: Option<u64> = None;
        for (key, st) in self.keys.iter() {
            if Some(*key) == exclude { continue; }
            if st.queue.is_empty() || st.offsets.is_empty() { continue; }
            let next_idx = st.next_idx;
            let loop_idx = next_idx / st.offsets.len() as u64;
            let in_loop_idx = (next_idx % st.offsets.len() as u64) as usize;
            let candidate_offset = st.offsets[in_loop_idx] + loop_idx * self.period_us;
            let wake_time = self.baseline_us + candidate_offset;
            best = match best { Some(b) => Some(b.min(wake_time)), None => Some(wake_time) };
        }
        best.filter(|t| *t > now_us)
    }
}

impl FlowScheduler for ReleaseFlowScheduler {
    fn enqueue_flow(&mut self, now_us: u64, flow: QueuedFlow) -> Option<u64> {
        let key = FlowKey { job_id: flow.job_id, job_flow_idx: flow.job_flow_idx };
        // Mutably access the per-key state once to check emptiness and push
        let (was_empty, offsets_clone, next_idx_after_push) = {
            let st = self.keys.entry(key).or_default();
            let was_empty = st.queue.is_empty();
            st.queue.push_back(flow);
            (was_empty, st.offsets.clone(), st.next_idx)
        };
        // If no schedule/offsets, no wake needed
        if self.period_us == 0 || offsets_clone.is_empty() { return None; }
        // Only consider scheduling if this key transitioned from empty -> non-empty.
        // Otherwise, there should already be a poll scheduled for this key's next release.
        if !was_empty { return None; }
        // Determine if an occurrence is owed now for this key
        let delta = now_us.saturating_sub(self.baseline_us);
        let total = self.total_occurrences_until(&offsets_clone, delta);
        let candidate_time = if total > next_idx_after_push {
            // Owed now
            now_us
        } else {
            // Next scheduled release time for this key
            let loop_idx = next_idx_after_push / offsets_clone.len() as u64;
            let in_loop_idx = (next_idx_after_push % offsets_clone.len() as u64) as usize;
            let candidate_offset = offsets_clone[in_loop_idx] + loop_idx * self.period_us;
            self.baseline_us + candidate_offset
        };
        // Compare against existing global next wake (excluding this key since it was empty before)
        match self.next_global_wake_after_excluding(now_us, Some(key)) {
            Some(existing) => if candidate_time < existing { Some(candidate_time) } else { None },
            None => Some(candidate_time),
        }
    }

    fn poll_ready(&mut self, now_us: u64) -> (Vec<QueuedFlow>, Option<u64>) {
        let delta = now_us.saturating_sub(self.baseline_us);
        // Avoid immutable borrow during mutable iteration by taking keys first.
        let key_list: Vec<FlowKey> = self.keys.keys().copied().collect();
        for key in key_list {
            // First, read-only data from an immutable borrow
            let (offsets_clone, next_idx_current, queue_len) = {
                let st_ro = self.keys.get(&key).unwrap();
                (st_ro.offsets.clone(), st_ro.next_idx, st_ro.queue.len())
            };
            if offsets_clone.is_empty() || self.period_us == 0 { continue; }
            let total = self.total_occurrences_until(&offsets_clone, delta);
            let owed = total.saturating_sub(next_idx_current);
            if owed == 0 { continue; }
            let to_release = std::cmp::min(owed as usize, queue_len);
            if to_release == 0 { continue; }
            // Now mutate: pop from queue and update next_idx
            let st = self.keys.get_mut(&key).unwrap();
            for _ in 0..to_release {
                if let Some(f) = st.queue.pop_front() { self.ready.push_back(f); }
            }
            st.next_idx = next_idx_current.saturating_add(to_release as u64);
        }
        let mut out = Vec::with_capacity(self.ready.len());
        while let Some(f) = self.ready.pop_front() { out.push(f); }
        (out, self.next_global_wake_after(now_us))
    }

    fn is_idle(&self) -> bool {
        self.keys.values().all(|st| st.queue.is_empty()) && self.ready.is_empty()
    }

    fn on_migration_begin(&mut self, _now_us: u64, ctx: &MLContext, affected_jobs: &[crate::simulator::ml_job::JobId]) {
        if affected_jobs.is_empty() { return; }
        let affected: std::collections::HashSet<_> = affected_jobs.iter().copied().collect();
        let placements = ctx.placements.borrow();
        // ready buffer
        for f in self.ready.iter_mut() {
            if !affected.contains(&f.job_id) { continue; }
            if let Some(w2h) = placements.get(&f.job_id) {
                if let (Some(&src_h), Some(&dst_h)) = (w2h.get(&f.src_worker), w2h.get(&f.dst_worker)) {
                    f.src_host = src_h;
                    f.dst_host = dst_h;
                }
            }
        }
        // per-key queues
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