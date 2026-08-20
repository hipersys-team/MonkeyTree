use crate::network::alloc::BandwidthAllocator;
use crate::utils::data::{DHashMap, DHashSet};
use crate::network::topology::Topology;
use crate::network::flow::{FlowDesc, FlowId, FlowState};
use crate::simulator::ml_simulator::MLContext;
use indexmap::IndexMap;
use std::cell::{Cell, RefCell};
use std::hash::Hasher;
use xxhash_rust::xxh64::Xxh64;

pub struct MLTCPTopoApprox {
    context: Option<MLContext>,
    tie_break_seed: u64,
    /// Per-job count of how many allocation rounds the job had active flows but was skipped
    skip_counts: RefCell<DHashMap<usize, u64>>, // job_id -> skips
    /// Monotonic per-allocation round index for deterministic reshuffle each call
    round_index: Cell<u64>,
}

impl MLTCPTopoApprox {
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

impl BandwidthAllocator for MLTCPTopoApprox {
    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn allocate(&self, topo: &impl Topology, active_desc: &IndexMap<FlowId, FlowDesc>, active_state: &IndexMap<FlowId, FlowState>) -> Vec<f64> {
        let n_flows = active_desc.len();
        let mut rates = vec![0.0; n_flows];

        // Build flow_id to index mapping
        let fid_to_idx: DHashMap<FlowId, usize> = active_desc.keys().enumerate().map(|(i, fid)| (*fid, i)).collect();

        let ctx = self.context.as_ref().unwrap();

        // Denominator: per-job per-iteration total bytes (sum workers' per-iter totals)
        let mut per_iter_total_by_job: DHashMap<usize, u64> = DHashMap::default();
        // Also collect per-worker sent_in_iter
        let mut sent_in_iter_by_worker: DHashMap<(usize, usize), u64> = DHashMap::default();
        for (&(jid, wid), (sent_in_iter, per_iter_total_w)) in ctx.worker_send_progress.borrow().iter() {
            let cur = per_iter_total_by_job.get(&jid).copied().unwrap_or(0);
            per_iter_total_by_job.insert(jid, cur.saturating_add(*per_iter_total_w));
            sent_in_iter_by_worker.insert((jid, wid), *sent_in_iter);
        }

        // Build per-job mapping of worker_id -> (pre_boundary_bytes, post_boundary_bytes)
        // Determine job-wide largest compute period from worker descriptions
        let active_jobs = ctx.active_jobs.borrow();
        let placements = ctx.placements.borrow(); // used to recover worker_id ordering

        let mut pre_post_by_worker: DHashMap<(usize, usize), (u64, u64)> = DHashMap::default();

        for (&jid, info) in active_jobs.iter() {
            // Find job-wide largest compute period (ms)
            let mut largest_compute_us: u64 = 0;
            for desc in &info.worker_descriptions {
                for step in &desc.steps {
                    if step.compute_us > largest_compute_us { largest_compute_us = step.compute_us; }
                }
            }
            // Recover worker ids in the same order as worker_descriptions: sorted by worker_id
            let mut worker_ids: Vec<usize> = placements.get(&jid)
                .map(|m| m.keys().copied().collect())
                .unwrap_or_else(Vec::new);
            worker_ids.sort_unstable();

            for (idx, desc) in info.worker_descriptions.iter().enumerate() {
                if idx >= worker_ids.len() { break; }
                let wid = worker_ids[idx];
                let mut cum_compute: u64 = 0;
                let mut pre_bytes: u64 = 0;
                let mut post_bytes: u64 = 0;
                for step in &desc.steps {
                    cum_compute = cum_compute.saturating_add(step.compute_us);
                    if let Some(flow) = &step.flow {
                        if cum_compute < largest_compute_us {
                            pre_bytes = pre_bytes.saturating_add(flow.size_bytes);
                        } else {
                            post_bytes = post_bytes.saturating_add(flow.size_bytes);
                        }
                    }
                }
                pre_post_by_worker.insert((jid, wid), (pre_bytes, post_bytes));
            }
        }

        // Compute numerator since boundary: per job, sum over workers of
        // completed_since_boundary + partials (for active flows) if worker passed the boundary
        let mut since_boundary_by_job: DHashMap<usize, u64> = DHashMap::default();

        // First, add completed since boundary from sent_in_iter deltas
        for ((jid, wid), sent) in sent_in_iter_by_worker.iter() {
            let (pre_b, post_b) = pre_post_by_worker.get(&(*jid, *wid)).copied().unwrap_or((0, 0));
            let completed_since = sent.saturating_sub(pre_b).min(post_b);
            let cur = since_boundary_by_job.get(jid).copied().unwrap_or(0);
            since_boundary_by_job.insert(*jid, cur.saturating_add(completed_since));
        }

        // Then add partials for active flows if source worker has crossed boundary
        // Track per-worker remaining budget in post-boundary to cap partials
        let mut remaining_post_budget: DHashMap<(usize, usize), u64> = DHashMap::default();
        for ((jid, wid), sent) in sent_in_iter_by_worker.iter() {
            let (pre_b, post_b) = pre_post_by_worker.get(&(*jid, *wid)).copied().unwrap_or((0, 0));
            let completed_since = sent.saturating_sub(pre_b).min(post_b);
            let rem = post_b.saturating_sub(completed_since);
            remaining_post_budget.insert((*jid, *wid), rem);
        }

        let waiting = ctx.waiting_flows.borrow();
        for ((fid, desc), (_fid2, state)) in active_desc.iter().zip(active_state.iter()) {
            if let Some((jid, _job_flow_idx, _iter_idx, src_worker, _dst_worker, _send_eid, _recv_eid)) = waiting.get(fid) {
                let sent = sent_in_iter_by_worker.get(&(*jid, *src_worker)).copied().unwrap_or(0);
                let (pre_b, post_b) = pre_post_by_worker.get(&(*jid, *src_worker)).copied().unwrap_or((0, 0));
                if sent >= pre_b && post_b > 0 {
                    let partial = desc.size_bytes.saturating_sub(state.remaining_bytes);
                    let key = (*jid, *src_worker);
                    let rem = remaining_post_budget.get(&key).copied().unwrap_or(0);
                    if rem > 0 {
                        let add = partial.min(rem);
                        let cur = since_boundary_by_job.get(jid).copied().unwrap_or(0);
                        since_boundary_by_job.insert(*jid, cur.saturating_add(add));
                        remaining_post_budget.insert(key, rem.saturating_sub(add));
                    }
                }
            }
        }

        // Build job order by ratio: since_boundary / per_iter_total (descending), then tie-key
        let mut job_ratios: Vec<(usize, f64, u64)> = Vec::new();
        let round = self.round_index.get();
        for (jid, denom) in per_iter_total_by_job.iter() {
            let num = since_boundary_by_job.get(jid).copied().unwrap_or(0);
            let ratio = if *denom == 0 { 0.0 } else { (num as f64) / (*denom as f64) };
            let tie_key = self.tie_key(round, *jid * 7 + 3);
            job_ratios.push((*jid, ratio, tie_key));
        }

        job_ratios.sort_by(|(_ida, ra, ka), (_idb, rb, kb)| {
            match rb.partial_cmp(ra).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Equal => ka.cmp(kb),
                other => other,
            }
        });

        // Group flows per job
        let mut job_to_flows: DHashMap<usize, Vec<FlowId>> = DHashMap::default();
        for (jid, _r, _k) in &job_ratios { job_to_flows.insert(*jid, Vec::new()); }
        for (fid, _) in active_desc.iter() {
            if let Some((jid, _job_flow_idx, _iter_idx, _src_worker, _dst_worker, _send_eid, _recv_eid)) = waiting.get(fid) {
                job_to_flows.get_mut(jid).unwrap().push(*fid);
            }
        }

        // Allocate whole link bandwidth to jobs in priority order subject to link disjointness
        let mut acquired_links = DHashSet::default();
        let mut allocated_jobs: DHashSet<usize> = DHashSet::default();
        for (jid, _r, _k) in &job_ratios {
            let mut free = true;
            for fid in job_to_flows.get(jid).unwrap() {
                if !free { break; }
                let idx = fid_to_idx[fid];
                for &lid in active_state.get_index(idx).unwrap().1.path_cell.path.borrow().iter() {
                    if acquired_links.contains(&lid) { free = false; break; }
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
            for (jid, _r, _k) in &job_ratios {
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


