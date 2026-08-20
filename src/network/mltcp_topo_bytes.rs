use crate::network::alloc::BandwidthAllocator;
use crate::utils::data::{DHashSet, DHashMap};
use crate::network::topology::Topology;
use crate::network::flow::FlowId;
use crate::network::flow::FlowDesc;
use crate::network::flow::FlowState;
use std::cmp::Ordering;
use indexmap::IndexMap;
use crate::simulator::ml_simulator::MLContext;
use xxhash_rust::xxh64::Xxh64;
use std::hash::Hasher;
use std::cell::{RefCell, Cell};

pub struct MLTCPTopoBytes {
    context: Option<MLContext>,
    tie_break_seed: u64,
    /// Per-job count of how many allocation rounds the job had active flows but was skipped
    skip_counts: RefCell<DHashMap<usize, u64>>, // job_id -> skips
    /// Monotonic per-allocation round index for deterministic reshuffle each call
    round_index: Cell<u64>,
}

impl MLTCPTopoBytes {
    pub fn new() -> Self {
        Self {
            context: None,
            tie_break_seed: 0x9E37_79B9_7F4A_7C15u64,
            skip_counts: RefCell::new(DHashMap::default()),
            round_index: Cell::new(0),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            context: None,
            tie_break_seed: seed,
            skip_counts: RefCell::new(DHashMap::default()),
            round_index: Cell::new(0),
        }
    }

    pub fn set_seed(&mut self, seed: u64) { self.tie_break_seed = seed; }

    #[inline]
    fn tie_key(&self, round: u64, job_id: usize) -> u64 {
        let mut hasher = Xxh64::new(self.tie_break_seed);
        hasher.write_u64(round);
        hasher.write_u64(job_id as u64);
        hasher.finish()
    }
}

impl BandwidthAllocator for MLTCPTopoBytes {
    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn allocate(&self, topo: &impl Topology, active_desc: &IndexMap<FlowId, FlowDesc>, active_state: &IndexMap<FlowId, FlowState>) -> Vec<f64> {
        let n_flows = active_desc.len();
        let mut rates = vec![0.0; n_flows];

        // Build flow_id to index mapping
        let fid_to_idx: DHashMap<FlowId, usize> = active_desc.keys().enumerate().map(|(i, fid)| (*fid, i)).collect();

        // --- Compute per-job current-iteration progress ratios and bytes sent ---
        let ctx = self.context.as_ref().unwrap();
        // Per-job per-iteration total bytes to send: sum workers' per-iteration totals from context
        let mut per_iter_total_by_job: DHashMap<usize, u64> = DHashMap::default();
        for ((jid, _wid), (_sent_in_iter, per_iter_total_w)) in ctx.worker_send_progress.borrow().iter() {
            let cur = per_iter_total_by_job.get(jid).copied().unwrap_or(0);
            per_iter_total_by_job.insert(*jid, cur.saturating_add(*per_iter_total_w));
        }

        // Base progress from completed sends within the current iteration (already capped and reset by simulator)
        let mut current_iter_progress_by_job: DHashMap<usize, u64> = DHashMap::default();
        for ((jid, _wid), (sent_in_iter, _per_iter_total_w)) in ctx.worker_send_progress.borrow().iter() {
            let cur = current_iter_progress_by_job.get(jid).copied().unwrap_or(0);
            current_iter_progress_by_job.insert(*jid, cur.saturating_add(*sent_in_iter));
        }

        // Add partial progress from currently active flows (not yet completed)
        let waiting = ctx.waiting_flows.borrow();
        for ((fid, desc), (_fid2, state)) in active_desc.iter().zip(active_state.iter()) {
            if let Some((jid, _job_flow_idx, _iter_idx, _src_worker, _dst_worker, _send_eid, _recv_eid)) = waiting.get(fid) {
                let partial = desc.size_bytes.saturating_sub(state.remaining_bytes);
                let cur = current_iter_progress_by_job.get(jid).copied().unwrap();
                current_iter_progress_by_job.insert(*jid, cur.saturating_add(partial));
            }
        }

        // Build list with deterministic per-call tie-break keys and per-job lifetime bytes-sent counts
        let mut job_bytes: Vec<(usize, u64, u64, u64)> = Vec::new();
        let round = self.round_index.get();
        let job_iters = ctx.job_iterations.borrow();
        let flow_completions = ctx.flow_completions_per_job.borrow();
        for (jid, per_iter_total_job) in per_iter_total_by_job.iter() {
            let prog = current_iter_progress_by_job.get(jid).copied().unwrap();
            let completed_iters: u64 = job_iters.get(jid).map(|(_total, completed)| *completed as u64).unwrap_or(0);
            let completed_bytes = completed_iters.saturating_mul(*per_iter_total_job);
            let lifetime_bytes = completed_bytes.saturating_add(prog);
            let completed_flows = flow_completions.get(jid).copied().unwrap_or(0);
            let tie_key = self.tie_key(round, *jid * 5 + 10);
            job_bytes.push((*jid, lifetime_bytes, completed_flows, tie_key));
        }

        // Sort by: lifetime bytes sent asc, then completed flows asc, then tie-key asc
        job_bytes.sort_by(|(_id_a, b_a, c_a, k_a), (_id_b, b_b, c_b, k_b)| {
            match b_a.cmp(b_b) {
                Ordering::Equal => match c_a.cmp(c_b) {
                    Ordering::Equal => k_a.cmp(k_b),
                    other => other,
                },
                other => other,
            }
        });

        let mut job_to_flows: DHashMap<usize, Vec<FlowId>> = DHashMap::default();
        for (jid, _bytes, _completed, _tie) in &job_bytes {
            job_to_flows.insert(*jid, Vec::new());
        }
        for (fid, _) in active_desc.iter() {
            let job_id = ctx.waiting_flows.borrow().get(fid).unwrap().0;
            job_to_flows.get_mut(&job_id).unwrap().push(*fid);
        }

        let mut acquired_links = DHashSet::default();
        let mut allocated_jobs: DHashSet<usize> = DHashSet::default();
        for (jid, _bytes, _completed, _tie) in &job_bytes {
            let mut free = true;
            for fid in job_to_flows.get(jid).unwrap() {
                if !free {
                    break;
                }
                let idx = fid_to_idx[fid];
                for &lid in active_state.get_index(idx).unwrap().1.path_cell.path.borrow().iter() {
                    if acquired_links.contains(&lid) {
                        free = false;
                        break;
                    }
                }
            }
            if free && !job_to_flows.get(jid).unwrap().is_empty() {
                for fid in job_to_flows.get(jid).unwrap() {
                    let idx = fid_to_idx[fid];
                    for &lid in active_state.get_index(idx).unwrap().1.path_cell.path.borrow().iter() {
                        acquired_links.insert(lid);
                    }
                    rates[idx] = topo.link_bandwidth_bps();
                }
                allocated_jobs.insert(*jid);
            }
        }

        // Increment skip counts for jobs that had active flows but were not allocated
        {
            let mut skips = self.skip_counts.borrow_mut();
            for (jid, _bytes, _completed, _tie) in &job_bytes {
                let had_active = job_to_flows.get(jid).map(|v| !v.is_empty()).unwrap_or(false);
                if had_active && !allocated_jobs.contains(jid) {
                    *skips.entry(*jid).or_insert(0) += 1;
                }
            }
        }

        // Advance round index for next allocation call
        self.round_index.set(round.wrapping_add(1));

        rates
    }
}


