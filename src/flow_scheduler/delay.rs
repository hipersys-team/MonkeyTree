use std::collections::{HashMap, VecDeque, BTreeMap, HashSet};

use crate::simulator::flow_scheduler::{FlowScheduler, QueuedFlow};
use crate::simulator::ml_simulator::MLContext;

/// A simple scheduler that accepts a per-job delay and releases flows from a
/// job only after its delay has elapsed since the first flow arrival for that
/// job. Once the delay passes for a job, all queued and future flows for that
/// job are eligible to be released immediately.
#[derive(Debug, Default)]
pub struct DelayScheduler {
    /// Configured per-job delays (in ms) for the current cycle.
    job_delay_us: HashMap<crate::simulator::ml_job::JobId, u64>,
    /// Backlog of flows per job waiting for their gate to open.
    per_job_queues: HashMap<crate::simulator::ml_job::JobId, VecDeque<QueuedFlow>>,
    /// Index of future gates to coalesce wakes: gate_time -> list of job ids with non-empty queues
    gate_index: BTreeMap<u64, Vec<crate::simulator::ml_job::JobId>>,
    /// Currently scheduled next wake (if any), used to coalesce duplicate wakes
    scheduled_next_wake: Option<u64>,
    /// Jobs whose gates are already open; any queued/future flows release immediately on poll
    open_jobs: HashSet<crate::simulator::ml_job::JobId>,
    // --- instrumentation ---
    /// Total number of times poll_ready was called
    instr_polls: u64,
    /// Total number of job-queues scanned during readiness checks
    instr_ready_scans: u64,
    /// Total number of queued flows released from poll_ready
    instr_flows_released: u64,
    ///// Total number of next-wake computations performed (approximate scans)
    //instr_nextwake_scans: u64,
    /// Total wakes requested from enqueue_flow
    instr_wakes_requested: u64,
    /// Wakes requested at exactly now_us
    instr_wakes_now: u64,
    /// Wakes requested in the future (gate time > now_us)
    instr_wakes_future: u64,
    /// First-flow arrival baseline per job for current cycle, if seen.
    first_arrival_us: HashMap<crate::simulator::ml_job::JobId, u64>,
}

impl DelayScheduler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Install the per-job delays (in milliseconds) for the current cycle.
    /// Delays are measured starting from when the FIRST flow for that job
    /// arrives to the scheduler.
    pub fn set_job_delays(
        &mut self,
        delays: Vec<(crate::simulator::ml_job::JobId, u64)>,
    ) {
        self.job_delay_us.clear();
        for (job_id, delay) in delays.into_iter() {
            self.job_delay_us.insert(job_id, delay);
        }
    }

    fn gate_time_for_job(&self, job_id: crate::simulator::ml_job::JobId) -> Option<u64> {
        let base = *self.first_arrival_us.get(&job_id)?;
        let delay = *self.job_delay_us.get(&job_id).unwrap_or(&0);
        Some(base.saturating_add(delay))
    }
}

impl FlowScheduler for DelayScheduler {
    fn enqueue_flow(&mut self, now_us: u64, flow: QueuedFlow) -> Option<u64> {
        let job_id = flow.job_id;

        let was_empty = {
            let queue = self.per_job_queues.entry(job_id).or_insert_with(VecDeque::new);
            let was_empty = queue.is_empty();
            queue.push_back(flow);
            was_empty
        };

        // If the queue was previously empty, this is the first arrival for the job.
        if was_empty {
            // Record first arrival baseline for this job
            self.first_arrival_us.entry(job_id).or_insert(now_us);
            if let Some(gate) = self.gate_time_for_job(job_id) {
                self.instr_wakes_requested += 1;
                if now_us >= gate {
                    // Gate already open; mark job as open for immediate release on next poll
                    self.instr_wakes_now += 1;
                    self.open_jobs.insert(job_id);
                    None
                } else {
                    // coalesce future wakes
                    let entry = self.gate_index.entry(gate).or_default();
                    if !entry.iter().any(|&j| j == job_id) { entry.push(job_id); }
                    let should_schedule = match self.scheduled_next_wake { None => true, Some(cur) => gate < cur };
                    if should_schedule { self.scheduled_next_wake = Some(gate); self.instr_wakes_future += 1; Some(gate) } else { None }
                }
            } else { None }
        } else { None }
    }

    fn poll_ready(&mut self, now_us: u64) -> (Vec<QueuedFlow>, Option<u64>) {
        let mut out: Vec<QueuedFlow> = Vec::new();

        // instrumentation: count a poll and how many job queues we scan for readiness
        self.instr_polls += 1;

        // Drain jobs whose gate time has passed per index; mark them open
        while let Some((&gate, _)) = self.gate_index.iter().next() {
            if gate > now_us { break; }
            if let Some(jobs) = self.gate_index.remove(&gate) {
                self.instr_ready_scans += jobs.len() as u64;
                for job_id in jobs {
                    if let Some(queue) = self.per_job_queues.get_mut(&job_id) {
                        while let Some(f) = queue.pop_front() { out.push(f); }
                    }
                    self.open_jobs.insert(job_id);
                }
            }
        }

        // Drain all currently open jobs
        let open_list: Vec<_> = self.open_jobs.iter().copied().collect();
        self.instr_ready_scans += open_list.len() as u64;
        for job_id in open_list {
            if let Some(queue) = self.per_job_queues.get_mut(&job_id) {
                while let Some(f) = queue.pop_front() { out.push(f); }
            }
        }

        // Next wake is earliest remaining gate time (if any)
        let next = self.gate_index.keys().next().copied();
        self.scheduled_next_wake = next;

        // instrumentation: released flow count this poll
        self.instr_flows_released += out.len() as u64;
        (out, next)
    }

    fn is_idle(&self) -> bool {
        self.per_job_queues.values().all(|q| q.is_empty())
    }

    fn on_migration_begin(&mut self, _now_us: u64, ctx: &MLContext, affected_jobs: &[crate::simulator::ml_job::JobId]) {
        if affected_jobs.is_empty() { return; }
        let placements = ctx.placements.borrow();
        for jid in affected_jobs.iter().copied() {
            if let Some(queue) = self.per_job_queues.get_mut(&jid) {
                if let Some(w2h) = placements.get(&jid) {
                    for f in queue.iter_mut() {
                        if let (Some(&src_h), Some(&dst_h)) = (w2h.get(&f.src_worker), w2h.get(&f.dst_worker)) {
                            f.src_host = src_h;
                            f.dst_host = dst_h;
                        }
                    }
                }
            }
        }
        // open_jobs and gate_index remain valid; host changes do not affect timing.
    }

    fn on_migration_end(&mut self, _now_us: u64) -> Option<u64> { self.scheduled_next_wake }
}

impl DelayScheduler {
    /// Print a concise summary of instrumentation counters.
    pub fn print_stats(&self) {
        //let unique_jobs = self.per_job_queues.len();
        // println!(
        //     "[DelayScheduler] summary: polls={} ready_scans_total={} nextwake_scans_total={} flows_released_total={} wakes_req_total={} wakes_now={} wakes_future={} job_queues={} pending_gates={} open_jobs={} scheduled_next_wake={:?}",
        //     self.instr_polls,
        //     self.instr_ready_scans,
        //     self.instr_nextwake_scans,
        //     self.instr_flows_released,
        //     self.instr_wakes_requested,
        //     self.instr_wakes_now,
        //     self.instr_wakes_future,
        //     unique_jobs,
        //     self.gate_index.len(),
        //     self.open_jobs.len(),
        //     self.scheduled_next_wake
        // );
    }
}

impl Drop for DelayScheduler {
    fn drop(&mut self) {
        // Print once at end of program scope
        // self.print_stats();
    }
}


