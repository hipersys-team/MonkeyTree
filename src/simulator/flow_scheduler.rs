use std::collections::VecDeque;
use crate::simulator::ml_simulator::MLContext;

/// Description of a flow awaiting scheduling by the ML flow scheduler.
#[derive(Debug, Clone)]
pub struct QueuedFlow {
    /// Job identifier this flow belongs to
    pub job_id: crate::simulator::ml_job::JobId,
    /// Job-local stable flow index (same across iterations)
    pub job_flow_idx: usize,
    /// Iteration index for this specific occurrence
    pub iter_idx: usize,
    /// Source worker id (for callback wiring)
    pub src_worker: crate::simulator::ml_worker::WorkerId,
    /// Destination worker id (for callback wiring)
    pub dst_worker: crate::simulator::ml_worker::WorkerId,
    /// Send event id (current iteration)
    pub send_event_id: usize,
    /// Receive event id (current iteration)
    pub receive_event_id: usize,
    /// Source host index
    pub src_host: usize,
    /// Destination host index
    pub dst_host: usize,
    /// Flow size in bytes
    pub size_bytes: u64,
}

/// Trait for ML-level flow scheduling strategies.
///
/// A flow scheduler is notified by the simulator when time advances and
/// may release any subset of queued flows to be generated in the network
/// simulator at chosen times (including immediately).
pub trait FlowScheduler {
    /// Notify the scheduler that a flow is ready; scheduler enqueues it.
    /// Returns an optional time (in ms) when the simulator should wake the
    /// flow scheduler to poll again. Implementations may return None to rely
    /// on the simulator's periodic polling only.
    fn enqueue_flow(&mut self, now_us: u64, flow: QueuedFlow) -> Option<u64>;

    /// Query which flows should run now. The simulator calls this before
    /// processing each event and immediately installs any returned flows into
    /// the network simulator.
    /// Returns any flows that should be released now, and an optional time
    /// (in ms) for when the simulator should wake the scheduler again.
    fn poll_ready(&mut self, now_us: u64) -> (Vec<QueuedFlow>, Option<u64>);

    /// Returns true if the scheduler has no queued flows.
    fn is_idle(&self) -> bool;

    /// Called at the start of a migration phase. Implementations should update
    /// any queued flows' src/dst host indices to reflect the current placements
    /// in the provided context for the specified affected jobs.
    fn on_migration_begin(&mut self, _now_us: u64, _ctx: &MLContext, _affected_jobs: &[crate::simulator::ml_job::JobId]) {}

    /// Called at the end of a migration phase. Implementations can return an
    /// optional next wake time to poll again.
    fn on_migration_end(&mut self, _now_us: u64) -> Option<u64> { None }
}

#[derive(Debug, Default)]
pub struct ImmediateFlowScheduler {
    /// Internal queue in case we want to batch within the same tick
    queue: VecDeque<QueuedFlow>,
}

impl ImmediateFlowScheduler {
    pub fn new() -> Self { Self { queue: VecDeque::new() } }
}

impl FlowScheduler for ImmediateFlowScheduler {
    fn enqueue_flow(&mut self, _now_us: u64, flow: QueuedFlow) -> Option<u64> {
        // Enqueue and let the simulator pick it up on the next poll.
        self.queue.push_back(flow);
        None
    }

    fn poll_ready(&mut self, _now_us: u64) -> (Vec<QueuedFlow>, Option<u64>) {
        let mut out = Vec::with_capacity(self.queue.len());
        while let Some(f) = self.queue.pop_front() {
            out.push(f);
        }
        (out, None)
    }

    fn is_idle(&self) -> bool {
        self.queue.is_empty()
    }

    fn on_migration_begin(&mut self, _now_us: u64, ctx: &MLContext, affected_jobs: &[crate::simulator::ml_job::JobId]) {
        if self.queue.is_empty() { return; }
        let affected: std::collections::HashSet<_> = affected_jobs.iter().copied().collect();
        let placements = ctx.placements.borrow();
        for f in self.queue.iter_mut() {
            if !affected.contains(&f.job_id) { continue; }
            if let Some(w2h) = placements.get(&f.job_id) {
                if let (Some(&src_h), Some(&dst_h)) = (w2h.get(&f.src_worker), w2h.get(&f.dst_worker)) {
                    f.src_host = src_h;
                    f.dst_host = dst_h;
                }
            }
        }
    }

    fn on_migration_end(&mut self, _now_us: u64) -> Option<u64> { None }
}