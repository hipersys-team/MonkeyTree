use crate::simulator::ml_worker::{MLWorker, WorkerEventKind};
use crate::network::topology::Topology;
use std::fmt;

/// Summary of a single flow-send template emitted by a worker.
#[derive(Debug, Clone)]
pub struct FlowSummary {
    /// Stable template identifier for this send event within the worker's DAG
    pub template_id: usize,
    /// Size of the flow emitted by this event
    pub size_bytes: u64,
}

/// A sequential step consisting of concrete computation followed by optional communication.
#[derive(Debug, Clone)]
pub struct Step {
    pub compute_us: u64,
    pub flow: Option<FlowSummary>,
}

impl fmt::Display for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.flow {
            Some(flow) => write!(
                f,
                "Compute({} ms) -> Flow(tid={}, size={} B)",
                self.compute_us, flow.template_id, flow.size_bytes
            ),
            None => write!(f, "Compute({} us)", self.compute_us),
        }
    }
}

/// Sequential description of a worker as compute segments and flows.
#[derive(Debug, Clone, Default)]
pub struct WorkerDescription {
    pub steps: Vec<Step>,
}

impl fmt::Display for WorkerDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, step) in self.steps.iter().enumerate() {
            writeln!(f, "  {}. {}", i + 1, step)?;
        }
        Ok(())
    }
}

impl WorkerDescription {
    /// Extracts a sequential description from an `MLWorker`'s template DAG.
    /// Uses declaration order; aggregates compute and emits a step when a FlowSend occurs.
    pub fn from_worker(worker: &MLWorker) -> Self {
        let mut steps: Vec<Step> = Vec::new();
        let mut compute_acc_us: u64 = 0;
        for ev in &worker.template_events {
            match ev.kind {
                WorkerEventKind::Compute => {
                    if let Some(comp) = &ev.compute {
                        compute_acc_us = compute_acc_us.saturating_add(comp.duration_us);
                    }
                }
                WorkerEventKind::FlowSend => {
                    if let Some(flow) = &ev.flow_send {
                        steps.push(Step { compute_us: compute_acc_us, flow: Some(FlowSummary { template_id: ev.template_id, size_bytes: flow.size_bytes }) });
                        compute_acc_us = 0;
                    }
                }
                WorkerEventKind::FlowReceive => {
                    // Receives are not part of the sender's emission profile
                }
            }
        }
        if compute_acc_us > 0 {
            steps.push(Step { compute_us: compute_acc_us, flow: None });
        }
        Self { steps }
    }
}

/// Computes the iteration "cost" for a worker description under the provided topology.
///
/// Note: This mirrors the current heuristic used by `compatibility`, summing compute time
/// in milliseconds with an estimate of network transmission time in seconds. While the
/// units are mixed, we preserve this behavior for relative comparisons.
fn iteration_time<T: Topology>(topo: &T, desc: &WorkerDescription) -> f64 {
    let mut total = 0.0f64;
    for step in desc.steps.iter() {
        total += step.compute_us as f64 / 1_000_000.0;
        if let Some(flow) = &step.flow {
            total += (flow.size_bytes as f64) * 8.0 / topo.link_bandwidth_bps();
        }
    }
    total
}

/// Computes a compatibility score between two worker descriptions.
/// Returns the ratio of the longer iteration to the shorter iteration (>= 1.0).
pub fn compatibility<T: Topology>(topo: &T, a: &WorkerDescription, b: &WorkerDescription) -> f64 {
    println!("Worker A description:\n{}", a);
    println!("Worker B description:\n{}", b);
    let iter1 = iteration_time(topo, a);
    let iter2 = iteration_time(topo, b);

    let score = if iter1 > 0.0 && iter2 > 0.0 {
        let (min_it, max_it) = if iter1 < iter2 { (iter1, iter2) } else { (iter2, iter1) };
        max_it / min_it
    } else {
        1.0
    };
    println!("Compatibility score: {}", score);
    score
}

/// Computes a compatibility score for an arbitrary number of worker descriptions.
/// Returns the largest ratio between any two jobs' iteration costs (>= 1.0).
pub fn compatibility_group<'a, T, I>(topo: &T, workers: I) -> f64
where
    T: Topology,
    I: IntoIterator<Item = &'a WorkerDescription>,
{
    let mut min_iteration = f64::INFINITY;
    let mut max_iteration = 0.0f64;
    let mut count = 0.0;

    for w in workers.into_iter() {
        count += 1.0;
        let cost = iteration_time(topo, w);
        min_iteration = min_iteration.min(cost);
        max_iteration = max_iteration.max(cost);
    }

    if count < 2.0 {
        1.0
    } else {
        max_iteration / min_iteration + (count * count * count)
    }
}


