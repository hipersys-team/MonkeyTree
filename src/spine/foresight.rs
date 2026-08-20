//! Foresight system module (skeleton)
//!
//! High-level idea:
//! - Observes job lifecycle and network state
//! - Computes repeating flow release schedules (periodic) for the cluster
//! - Reconfigures routing and rebases the ReleaseFlowScheduler on job changes
//!
//! Note: This is a skeleton for planning purposes and may not compile yet.

use crate::spine::{SpineTree, SpineSystemRouter, SpineTreeTopology};
use crate::network::topology::Topology;
use crate::simulator::{SystemModule, JobScheduler};
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::ml_worker::{WorkerId, WorkerEvent};
use crate::simulator::ml_simulator::MLContext;
use crate::flow_scheduler::release_scheduler::{ReleaseFlowScheduler, FlowReleaseSchedule, FlowReleaseSpec};
use std::collections::{BTreeMap, BTreeSet};

/// Foresight: a system module that plans periodic release schedules and
/// coordinates rebasing the ReleaseFlowScheduler on job changes.
#[derive(Debug, Default)]
pub struct Foresight {
    /// Active jobs tracked by the module: job_id -> (submit_time_us, sorted flow indices)
    active_jobs: BTreeMap<JobId, (u64, Vec<usize>)>,
    /// Monotonic schedule version counter
    next_version: u64,
    /// Slot size in ms for sequential offsets
    slot_us: u64,
    /// Cached models for advanced schedule builder
    job_models: BTreeMap<JobId, JobModel>,
}

impl Foresight {
    pub fn new() -> Self { Self { active_jobs: BTreeMap::new(), next_version: 1, slot_us: 1, job_models: BTreeMap::new() } }

    fn record_job(&mut self, job: &MLJob) {
        // Collect all job-local flow indices from the precomputed map
        let mut flow_idxs: Vec<usize> = job.send_template_to_flow_idx.values().copied().collect();
        flow_idxs.sort_unstable();
        self.active_jobs.insert(job.id, (job.submit_time_us, flow_idxs));
        self.job_models.insert(job.id, JobModel::from_job(job));
    }

    fn remove_job(&mut self, job_id: JobId) {
        self.active_jobs.remove(&job_id);
        self.job_models.remove(&job_id);
    }

    /// Build a schedule that allows one flow at a time across all active jobs,
    /// ordered by earliest job submit time. Flows within the same job are ordered
    /// by ascending job-local flow index.
    #[allow(unused)]
    fn build_serial_schedule(&mut self) -> FlowReleaseSchedule {
        // Sort jobs by (submit_time_us, job_id) using the BTreeMap iteration order on (job_id) and custom sort.
        let mut jobs: Vec<(JobId, u64, Vec<usize>)> = self
            .active_jobs
            .iter()
            .map(|(jid, (submit, flows))| (*jid, *submit, flows.clone()))
            .collect();
        jobs.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let mut entries: Vec<FlowReleaseSpec> = Vec::new();
        for (jid, _submit, mut flows) in jobs {
            flows.sort_unstable();
            for flow_idx in flows {
                entries.push(FlowReleaseSpec { job_id: jid, job_flow_idx: flow_idx, offset_us: 0 });
            }
        }

        // Assign offsets sequentially with fixed slot size
        for (i, e) in entries.iter_mut().enumerate() {
            e.offset_us = (i as u64) * self.slot_us;
        }
        let n = entries.len() as u64;
        let period_us = if n == 0 { 1 } else { n * self.slot_us };
        let schedule = FlowReleaseSchedule { version: self.next_version, period_us, entries };
        self.next_version += 1;
        schedule
    }

    /// Advanced schedule builder using virtual leaf↔spine links.
    /// Returns schedule and detailed placed flow instances for routing.
    fn build_flow_schedule(&mut self, topo: &SpineTree<SpineSystemRouter>) -> (FlowReleaseSchedule, Vec<PlacedFlow>) {
        let lane_bps = self.sample_leaf_spine_bandwidth(topo);
        // Homogeneous fabric: each leaf connects to all spines with identical bandwidth links.
        // Therefore lanes per leaf equals number of spines.
        let lanes_per_leaf: Vec<u32> = vec![topo.num_spines as u32; topo.num_leaves];

        // Initialize per-leaf lane timelines
        let mut timelines: Vec<LeafTimeline> = lanes_per_leaf
            .iter()
            .map(|&lanes| LeafTimeline::new(lanes))
            .collect();

        // Attained service and remaining iterations per job
        let mut service_us: BTreeMap<JobId, u64> = BTreeMap::new();
        let mut remaining: BTreeMap<JobId, usize> = BTreeMap::new();
        let mut iters_placed: BTreeMap<JobId, usize> = BTreeMap::new();
        for (jid, model) in &self.job_models {
            service_us.insert(*jid, 0);
            remaining.insert(*jid, model.total_iterations);
            iters_placed.insert(*jid, 0);
        }

        // Derive per-iteration flow templates for each job
        let mut job_flows: BTreeMap<JobId, Vec<FlowTemplate>> = BTreeMap::new();
        for (jid, model) in &self.job_models {
            let flows = self.derive_iteration_flows(model, topo, lane_bps);
            job_flows.insert(*jid, flows);
        }

        // Accumulate schedule entries and placed flows
        let mut entries: Vec<FlowReleaseSpec> = Vec::new();
        let mut placed: Vec<PlacedFlow> = Vec::new();
        // Track per-job earliest next start time (end of last iteration placed)
        let mut next_start: BTreeMap<JobId, u64> = BTreeMap::new();

        // Loop until all iterations are scheduled
        loop {
            let candidate = remaining
                .iter()
                .filter(|(_, &left)| left > 0)
                .min_by_key(|(jid, _)| service_us.get(jid).copied().unwrap_or(0));
            let Some((&jid, _)) = candidate else { break };

            let flows = job_flows.get(&jid).cloned().unwrap_or_default();
            // If no flows, count zero-length iteration
            if flows.is_empty() {
                *service_us.get_mut(&jid).unwrap() += 0;
                *remaining.get_mut(&jid).unwrap() -= 1;
                continue;
            }

            // earliest-fit start time (1ms discretization): end of job's last iteration
            let mut time = *next_start.get(&jid).unwrap_or(&0);
            'search: loop {
                // Capacity constraints apply only to flows that leave the leaf (src_leaf != dst_leaf)
                if flows.iter().filter(|f| f.src_leaf != f.dst_leaf).all(|f| {
                    let s = time + f.offset_us;
                    let e = s + f.duration_us;
                    timelines[f.src_leaf].can_place(s, e) && timelines[f.dst_leaf].can_place(s, e)
                }) {
                    // Place across all involved leaf links
                    let iter_idx = *iters_placed.get(&jid).unwrap_or(&0);
                    for f in &flows {
                        let s = time + f.offset_us;
                        let e = s + f.duration_us;
                        if f.src_leaf != f.dst_leaf {
                            timelines[f.src_leaf].place(s, e);
                            timelines[f.dst_leaf].place(s, e);
                        }
                        entries.push(FlowReleaseSpec { job_id: jid, job_flow_idx: f.job_flow_idx, offset_us: s });
                        placed.push(PlacedFlow { job_id: jid, job_flow_idx: f.job_flow_idx, iter_idx, start_us: s, end_us: e, src_host: f.src_host, dst_host: f.dst_host, src_leaf: f.src_leaf, dst_leaf: f.dst_leaf });
                    }
                    break 'search;
                }
                time += 1;
            }

            let iter_finish = flows.iter().map(|f| time + f.offset_us + f.duration_us).max().unwrap_or(time);
            *service_us.get_mut(&jid).unwrap() += iter_finish - time;
            next_start.insert(jid, iter_finish);
            *remaining.get_mut(&jid).unwrap() -= 1;
            *iters_placed.get_mut(&jid).unwrap() += 1;
        }

        entries.sort_by_key(|e| e.offset_us);
        let period_us = entries.last().map(|e| e.offset_us + 1).unwrap_or(1);
        let schedule = FlowReleaseSchedule { version: self.next_version, period_us, entries };
        self.next_version += 1;
        (schedule, placed)
    }

    /// Sample any leaf→spine link bandwidth as the unit lane bps (homogeneous)
    fn sample_leaf_spine_bandwidth(&self, topo: &SpineTree<SpineSystemRouter>) -> f64 {
        // Homogeneous: pick first leaf→spine edge
        let leaf = topo.leaf_switches[0];
        let spine = topo.spine_switches[0];
        let eidx = topo.graph.find_edge(leaf, spine).expect("leaf→spine edge present");
        topo.graph.edge_weight(eidx).unwrap().bandwidth
    }

    /// Build per-iteration flow templates (offset, duration, leaf endpoints) for a job.
    fn derive_iteration_flows(
        &self,
        model: &JobModel,
        topo: &SpineTree<SpineSystemRouter>,
        lane_bps: f64,
    ) -> Vec<FlowTemplate> {
        // Map worker→(host index, leaf index) per topology layout
        let mut w2host: BTreeMap<WorkerId, usize> = BTreeMap::new();
        let mut w2leaf: BTreeMap<WorkerId, usize> = BTreeMap::new();
        for (&wid, wm) in &model.workers {
            let host_idx = wm.host_index;
            let leaf_idx = host_idx / topo.hosts_per_leaf;
            w2host.insert(wid, host_idx);
            w2leaf.insert(wid, leaf_idx);
        }

        // Build a global DAG of events across all workers for this iteration
        #[derive(Clone)]
        struct NodeInfo { duration: u64 }

        // Assign a contiguous node index per (wid, event index)
        let mut node_index: BTreeMap<(WorkerId, usize), usize> = BTreeMap::new();
        let mut nodes: Vec<NodeInfo> = Vec::new();
        for (&wid, wm) in &model.workers {
            for (i, ev) in wm.template_events.iter().enumerate() {
                let idx = nodes.len();
                node_index.insert((wid, i), idx);
                let (duration, _is_send, _send_dst, _send_size) = match ev.kind {
                    crate::simulator::ml_worker::WorkerEventKind::Compute => (ev.compute.as_ref().map(|c| c.duration_us).unwrap_or(0), false, None, 0),
                    crate::simulator::ml_worker::WorkerEventKind::FlowSend => (bytes_to_us(ev.flow_send.as_ref().unwrap().size_bytes, lane_bps), true, Some(ev.flow_send.as_ref().unwrap().dst_worker), ev.flow_send.as_ref().unwrap().size_bytes),
                    _ => (0, false, None, 0), // FlowReceive has 0 duration
                };
                nodes.push(NodeInfo { duration });
            }
        }

        // Build edges with weights (predecessor duration), for both intra-worker deps and send→receive
        let mut edges: Vec<Vec<(usize, u64)>> = vec![Vec::new(); nodes.len()];
        let mut indeg: Vec<usize> = vec![0; nodes.len()];

        // Intra-worker dependencies
        for (&wid, wm) in &model.workers {
            // Map template_id -> event index in this worker
            let mut tidx: BTreeMap<usize, usize> = BTreeMap::new();
            for (i, ev) in wm.template_events.iter().enumerate() { tidx.insert(ev.template_id, i); }
            for (i, ev) in wm.template_events.iter().enumerate() {
                let v = node_index[&(wid, i)];
                for dep_tid in &ev.dependencies {
                    if let Some(&dep_i) = tidx.get(dep_tid) {
                        let u = node_index[&(wid, dep_i)];
                        edges[u].push((v, nodes[u].duration));
                        indeg[v] += 1;
                    }
                }
            }
        }

        // Build receive index by (dst_worker, src_worker) in order of appearance
        let mut recv_by_pair: BTreeMap<(WorkerId, WorkerId), Vec<usize>> = BTreeMap::new();
        for (&wid, wm) in &model.workers {
            for (i, ev) in wm.template_events.iter().enumerate() {
                if let crate::simulator::ml_worker::WorkerEventKind::FlowReceive = ev.kind {
                    let src = ev.flow_receive.as_ref().unwrap().src_worker;
                    recv_by_pair.entry((wid, src)).or_default().push(i);
                }
            }
        }

        // Send → matching receive edge (duration = send duration)
        for (&wid, wm) in &model.workers {
            for (i, ev) in wm.template_events.iter().enumerate() {
                if let crate::simulator::ml_worker::WorkerEventKind::FlowSend = ev.kind {
                    let dst = ev.flow_send.as_ref().unwrap().dst_worker;
                    if let Some(list) = recv_by_pair.get_mut(&(dst, wid)) {
                        if let Some(r_i) = list.first().copied() {
                            // consume this receive to preserve order
                            list.remove(0);
                            let u = node_index[&(wid, i)];
                            let v = node_index[&(dst, r_i)];
                            edges[u].push((v, nodes[u].duration));
                            indeg[v] += 1;
                        }
                    }
                }
            }
        }

        // Kahn's algorithm for earliest start times
        let mut q: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        let mut est: Vec<u64> = vec![0; nodes.len()];
        for i in 0..nodes.len() { if indeg[i] == 0 { q.push_back(i); } }
        while let Some(u) = q.pop_front() {
            let finish = est[u].saturating_add(nodes[u].duration);
            for &(v, w) in &edges[u] {
                if est[v] < finish.saturating_add(w.saturating_sub(nodes[u].duration)) {
                    // For our edge weights chosen as predecessor duration, this simplifies to est[v] = max(est[v], finish)
                    est[v] = finish.max(est[v]);
                }
                indeg[v] -= 1;
                if indeg[v] == 0 { q.push_back(v); }
            }
        }

        // Extract FlowSend nodes as flow templates with offsets and durations
        let mut out: Vec<FlowTemplate> = Vec::new();
        for (&wid, wm) in &model.workers {
            for (i, ev) in wm.template_events.iter().enumerate() {
                if let crate::simulator::ml_worker::WorkerEventKind::FlowSend = ev.kind {
                    let idx = node_index[&(wid, i)];
                    let send = ev.flow_send.as_ref().unwrap();
                    let src_host = w2host[&wid];
                    let dst_host = w2host[&send.dst_worker];
                    let src_leaf = w2leaf[&wid];
                    let dst_leaf = w2leaf[&send.dst_worker];
                    let duration_us = bytes_to_us(send.size_bytes, lane_bps);
                    let job_flow_idx = model.send_map.get(&(wid, ev.template_id)).copied().unwrap_or(ev.template_id);
                    out.push(FlowTemplate { src_host, dst_host, src_leaf, dst_leaf, offset_us: est[idx], duration_us, job_flow_idx });
                }
            }
        }
        // Neatly print the derived iteration flow template for this job
        println!(
            "ForesightDerivedTemplate job_id={} flows={} remaining_iters={}",
            model.job_id,
            out.len(),
            model.total_iterations
        );
        for f in &out {
            println!(
                "  src_leaf={} dst_leaf={} offset_us={} duration_us={} flow_idx={}",
                f.src_leaf,
                f.dst_leaf,
                f.offset_us,
                f.duration_us,
                f.job_flow_idx
            );
        }
        out
    }

    /// Compute routing assignments by bipartite edge coloring over groups of
    /// jobs whose placed flow intervals overlap in time (transitively).
    /// Reports the mapping job_id/flow_idx -> spine index.
    fn build_routing(&self, topo: &SpineTree<SpineSystemRouter>, placed: &[PlacedFlow]) {
        if placed.is_empty() { return; }

        let num_spines = topo.num_spines as usize;

        // Group flows by (job, iteration)
        let mut iter_to_indices: BTreeMap<(JobId, usize), Vec<usize>> = BTreeMap::new();
        for (i, pf) in placed.iter().enumerate() {
            iter_to_indices.entry((pf.job_id, pf.iter_idx)).or_default().push(i);
        }

        // Build per-(job,iter) merged intervals over placed flows
        let mut iter_intervals: BTreeMap<(JobId, usize), Vec<(u64, u64)>> = BTreeMap::new();
        for (ji, idxs) in &iter_to_indices {
            let mut ivals: Vec<(u64, u64)> = idxs.iter().map(|&i| (placed[i].start_us, placed[i].end_us)).collect();
            ivals.sort_unstable();
            let mut merged: Vec<(u64, u64)> = Vec::new();
            for (s, e) in ivals {
                if let Some(last) = merged.last_mut() {
                    if s <= last.1 { last.1 = last.1.max(e); } else { merged.push((s, e)); }
                } else { merged.push((s, e)); }
            }
            iter_intervals.insert(*ji, merged);
        }

        // Build overlap graph between (job,iter) nodes (transitive closure will form groups)
        let nodes: Vec<(JobId, usize)> = iter_to_indices.keys().copied().collect();
        let mut adj: BTreeMap<(JobId, usize), Vec<(JobId, usize)>> = BTreeMap::new();
        for (i, &a) in nodes.iter().enumerate() {
            for &b in nodes.iter().skip(i + 1) {
                if intervals_overlap(&iter_intervals[&a], &iter_intervals[&b]) {
                    adj.entry(a).or_default().push(b);
                    adj.entry(b).or_default().push(a);
                }
            }
        }

        // Connected components of (job,iter)
        let mut visited: BTreeSet<(JobId, usize)> = BTreeSet::new();
        let mut groups: Vec<Vec<(JobId, usize)>> = Vec::new();
        for &node in &nodes {
            if visited.contains(&node) { continue; }
            let mut q: std::collections::VecDeque<(JobId, usize)> = std::collections::VecDeque::new();
            let mut comp: Vec<(JobId, usize)> = Vec::new();
            visited.insert(node);
            q.push_back(node);
            while let Some(u) = q.pop_front() {
                comp.push(u);
                if let Some(neigh) = adj.get(&u) {
                    for &v in neigh {
                        if visited.insert(v) { q.push_back(v); }
                    }
                }
            }
            groups.push(comp);
        }

        println!("ForesightRouting groups={} spines={}", groups.len(), num_spines);

        // For each group, build bipartite graph (src_leaf ↔ dst_leaf) and color edges
        for (gidx, comp) in groups.iter().enumerate() {
            // Collect inter-leaf flow indices for this group
            let mut edge_indices: Vec<usize> = Vec::new();
            for &(jid, iter_idx) in comp {
                if let Some(list) = iter_to_indices.get(&(jid, iter_idx)) {
                    for &i in list { if placed[i].src_leaf != placed[i].dst_leaf { edge_indices.push(i); } }
                }
            }

            // Remap leaves to dense indices
            let mut left_map: BTreeMap<usize, usize> = BTreeMap::new();
            let mut right_map: BTreeMap<usize, usize> = BTreeMap::new();
            for &i in &edge_indices {
                let pf = &placed[i];
                let l = left_map.len();
                if !left_map.contains_key(&pf.src_leaf) { left_map.insert(pf.src_leaf, l); }
                let r = right_map.len();
                if !right_map.contains_key(&pf.dst_leaf) { right_map.insert(pf.dst_leaf, r); }
            }
            let left_len = left_map.len();
            let right_len = right_map.len();

            // Dense edge list
            let mut edges: Vec<(usize, usize, usize)> = Vec::new();
            for &i in &edge_indices {
                let pf = &placed[i];
                edges.push((left_map[&pf.src_leaf], right_map[&pf.dst_leaf], i));
            }

            // Compute Δ
            let mut deg_l = vec![0usize; left_len];
            let mut deg_r = vec![0usize; right_len];
            for &(u, v, _) in &edges { deg_l[u] += 1; deg_r[v] += 1; }
            let delta = deg_l.iter().chain(deg_r.iter()).copied().max().unwrap_or(0);
            if delta > num_spines { panic!("Routing requires {} colors but only {} spines available", delta, num_spines); }

            // Color arrays
            let colors_cap = num_spines;
            let mut edge_color: Vec<Option<usize>> = vec![None; edges.len()];
            let mut left_color_edge: Vec<Vec<Option<usize>>> = vec![vec![None; colors_cap]; left_len];
            let mut right_color_edge: Vec<Vec<Option<usize>>> = vec![vec![None; colors_cap]; right_len];

            for eidx in 0..edges.len() {
                let (u, v, _gi) = edges[eidx];
                // find smallest free color at left u
                let mut a = 0usize;
                while a < colors_cap && left_color_edge[u][a].is_some() { a += 1; }
                if a >= colors_cap { panic!("no free color at left; colors_cap insufficient"); }
                // If color a also free at v, assign directly
                if right_color_edge[v][a].is_none() {
                    left_color_edge[u][a] = Some(eidx);
                    right_color_edge[v][a] = Some(eidx);
                    edge_color[eidx] = Some(a);
                    continue;
                }
                // find smallest free color at right v
                let mut b = 0usize;
                while b < colors_cap && right_color_edge[v][b].is_some() { b += 1; }
                if b >= colors_cap { panic!("no free color at right; colors_cap insufficient"); }

                // Build alternating path from right node v alternating colors a (on right) and b (on left)
                // Store list of edges along the path in order: a, b, a, b, ... starting from right v
                let mut path: Vec<(usize, bool)> = Vec::new(); // (edge_idx, is_color_a)
                let mut cur_right = v;
                loop {
                    let e_a = right_color_edge[cur_right][a].expect("must exist a-colored edge at right to start path");
                    path.push((e_a, true));
                    let left_node = edges[e_a].0;
                    if let Some(e_b) = left_color_edge[left_node][b] {
                        path.push((e_b, false));
                        cur_right = edges[e_b].1; // advance to next right node
                    } else {
                        break; // terminated at left with no b-edge
                    }
                }
                // Flip colors along the path
                for &(pe, is_a) in &path {
                    let (uu, vv, _gii) = edges[pe];
                    let from = if is_a { a } else { b };
                    let to = if is_a { b } else { a };
                    // move pe from color 'from' to 'to'
                    debug_assert_eq!(edge_color[pe], Some(from));
                    left_color_edge[uu][from] = None;
                    right_color_edge[vv][from] = None;
                    debug_assert!(left_color_edge[uu][to].is_none());
                    debug_assert!(right_color_edge[vv][to].is_none());
                    left_color_edge[uu][to] = Some(pe);
                    right_color_edge[vv][to] = Some(pe);
                    edge_color[pe] = Some(to);
                }
                // Now color a is free at right v and was free at left u; assign a to (u,v)
                left_color_edge[u][a] = Some(eidx);
                right_color_edge[v][a] = Some(eidx);
                edge_color[eidx] = Some(a);
            }

            // Reporting and store assignments for later injection
            let mut group_set: BTreeSet<(JobId, usize)> = BTreeSet::new();
            for ji in comp { group_set.insert(*ji); }
            let colors_used: BTreeSet<usize> = edge_color.iter().filter_map(|&c| c).collect();
            println!("  Group {} job_iters={:?} colors_used={}", gidx, group_set, colors_used.len());

            // Build a map from placed index -> color
            let mut color_by_placed: BTreeMap<usize, usize> = BTreeMap::new();
            for (local_idx, &(_u, _v, gi)) in edges.iter().enumerate() {
                let c = edge_color[local_idx].expect("colored");
                color_by_placed.insert(gi, c);
            }

            // Construct and inject full link-path templates for all flows in this group (inter- and intra-leaf)
            let mut all_indices: Vec<usize> = Vec::new();
            for &(jid, iter_idx) in comp {
                if let Some(list) = iter_to_indices.get(&(jid, iter_idx)) { all_indices.extend(list.iter().copied()); }
            }
            for &i in &all_indices {
                let pf = &placed[i];
                // Build node path: if inter-leaf, use colored spine; if intra-leaf, only leaf hop
                let nodes = if pf.src_leaf == pf.dst_leaf {
                    let src = topo.get_host(pf.src_leaf, pf.src_host % topo.hosts_per_leaf).unwrap();
                    let dst = topo.get_host(pf.dst_leaf, pf.dst_host % topo.hosts_per_leaf).unwrap();
                    let leaf_node = topo.get_leaf(pf.src_leaf).unwrap();
                    vec![src, leaf_node, dst]
                } else {
                    let c = *color_by_placed.get(&i).expect("missing color for inter-leaf flow");
                    println!(
                        "    route job_id={} iter={} flow_idx={} src_leaf={} dst_leaf={} spine={} start_us={} end_us={}",
                        pf.job_id, pf.iter_idx, pf.job_flow_idx, pf.src_leaf, pf.dst_leaf, c, pf.start_us, pf.end_us
                    );
                    let src = topo.get_host(pf.src_leaf, pf.src_host % topo.hosts_per_leaf).unwrap();
                    let dst = topo.get_host(pf.dst_leaf, pf.dst_host % topo.hosts_per_leaf).unwrap();
                    let src_leaf_node = topo.get_leaf(pf.src_leaf).unwrap();
                    let dst_leaf_node = topo.get_leaf(pf.dst_leaf).unwrap();
                    let spine_node = topo.get_spine(c);
                    vec![src, src_leaf_node, spine_node, dst_leaf_node, dst]
                };
                // Convert to link path using graph
                let mut link_path: Vec<crate::network::topology::LinkId> = Vec::new();
                let graph = topo.topology();
                for w in nodes.windows(2) {
                    let eidx = graph.find_edge(w[0], w[1]).expect("edge exists");
                    let link = graph.edge_weight(eidx).unwrap();
                    link_path.push(link.id);
                }
                let mut router = topo.router.borrow_mut();
                router.inject_template(pf.job_id, pf.job_flow_idx, pf.iter_idx, link_path);
            }
        }
    }
}

fn intervals_overlap(a: &[(u64, u64)], b: &[(u64, u64)]) -> bool {
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        let (as_, ae) = a[i];
        let (bs, be) = b[j];
        if ae <= bs { i += 1; continue; }
        if be <= as_ { j += 1; continue; }
        // overlap if max(start) < min(end)
        if as_.max(bs) < ae.min(be) { return true; }
        if ae < be { i += 1; } else { j += 1; }
    }
    false
}

/// SystemModule binding for SpineTree with ECMP routing and ReleaseFlowScheduler.
///
/// This explicit impl ties Foresight to:
/// - Topology: SpineTree<SpineEcmpRouter>
/// - Flow scheduler: ReleaseFlowScheduler
impl<S> SystemModule<SpineTree<SpineSystemRouter>, S, ReleaseFlowScheduler> for Foresight
where
    S: JobScheduler,
{
    fn on_init(
        &mut self,
        _ctx: &MLContext,
        _topo: &SpineTree<SpineSystemRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut ReleaseFlowScheduler,
    ) {
        // Nothing to precompute; bandwidths computed locally when building schedule
    }

    fn on_job_scheduled(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        job: &MLJob,
        _topo: &SpineTree<SpineSystemRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut ReleaseFlowScheduler,
    ) {
        // Record job; reconfiguration will happen in on_reconfigure
        self.record_job(job);
    }

    fn on_job_completed(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        job_id: JobId,
        _job: &MLJob,
        _topo: &SpineTree<SpineSystemRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut ReleaseFlowScheduler,
    ) {
        // Remove job; reconfiguration will happen in on_reconfigure
        self.remove_job(job_id);
    }

    fn on_reconfigure(
        &mut self,
        now_us: u64,
        ctx: &MLContext,
        topo: &SpineTree<SpineSystemRouter>,
        _scheduler: &mut S,
        flow_scheduler: &mut ReleaseFlowScheduler,
    ) -> Option<crate::simulator::system::MigrationPlan> {
        // Capture remaining iterations from context and update our planning state
        if let Some(map) = ctx.job_iterations.try_borrow().ok() {
            // Reduce iterations for any completed jobs; remove fully completed from active set
            for (jid, (_total, completed)) in map.iter() {
                if let Some((_submit, _flows)) = self.active_jobs.get(jid) {
                    if *completed == 0 { /* nothing to skip */ } else {
                        // Reduce remaining iterations in job_models by marking total_iterations = total - completed
                        if let Some(model) = self.job_models.get_mut(jid) {
                            model.total_iterations = model.total_iterations.saturating_sub(*completed);
                        }
                    }
                }
            }
            // Drop any jobs with zero iterations remaining
            self.job_models.retain(|_, m| m.total_iterations > 0);
            self.active_jobs.retain(|jid, _| self.job_models.contains_key(jid));
        }

        // Build schedule (flow schedule + routing) and inject templates to router, then install.
        // Clear previous templates for efficiency and correctness.
        {
            let mut router = topo.router.borrow_mut();
            router.clear_templates();
        }
        let (schedule, placed) = self.build_flow_schedule(topo);

        // Pretty print the schedule being installed
        println!(
            "{} ForesightSchedule version={} period_us={} entries={}",
            now_us,
            schedule.version,
            schedule.period_us,
            schedule.entries.len()
        );
        for (i, e) in schedule.entries.iter().enumerate() {
            println!(
                "  slot={} job_id={} flow_idx={} offset_us={}",
                i,
                e.job_id,
                e.job_flow_idx,
                e.offset_us
            );
        }

        // Compute routing after schedule printout so logs reflect solve order
        self.build_routing(topo, &placed);

        flow_scheduler.rebase_to_new_schedule(now_us, schedule);
        None
    }
}

// ---------------- scaffolding for advanced planner ----------------

#[derive(Debug, Clone)]
struct WorkerModel {
    host_index: usize,
    template_events: Vec<WorkerEvent>,
}

#[derive(Debug, Clone)]
struct JobModel {
    #[allow(unused)]
    job_id: JobId,
    total_iterations: usize,
    workers: BTreeMap<WorkerId, WorkerModel>,
    /// Mapping from (src_worker, stable send template id) -> job-local flow index
    send_map: BTreeMap<(WorkerId, usize), usize>,
}

impl JobModel {
    fn from_job(job: &MLJob) -> Self {
        let mut workers = BTreeMap::new();
        for (&wid, w) in &job.workers {
            workers.insert(wid, WorkerModel { host_index: w.host_index, template_events: w.template_events.clone() });
        }
        let mut send_map = BTreeMap::new();
        for (k, v) in &job.send_template_to_flow_idx {
            send_map.insert(*k, *v);
        }
        Self { job_id: job.id, total_iterations: job.total_iterations, workers, send_map }
    }
}

#[derive(Debug, Clone)]
struct FlowTemplate {
    src_host: usize,
    dst_host: usize,
    src_leaf: usize,
    dst_leaf: usize,
    offset_us: u64,
    duration_us: u64,
    job_flow_idx: usize,
}

#[derive(Debug, Clone)]
struct PlacedFlow {
    job_id: JobId,
    job_flow_idx: usize,
    iter_idx: usize,
    start_us: u64,
    end_us: u64,
    src_host: usize,
    dst_host: usize,
    src_leaf: usize,
    dst_leaf: usize,
}

#[derive(Debug)]
struct LeafTimeline {
    lanes: u32,
    // difference map: time -> delta occupancy (lanes)
    diff: BTreeMap<u64, i32>,
}

impl LeafTimeline {
    fn new(lanes: u32) -> Self { Self { lanes, diff: BTreeMap::new() } }

    fn can_place(&self, start: u64, end: u64) -> bool {
        if start >= end { return true; }
        let mut cur: i32 = 0;
        // accumulate up to start
        for (_, &d) in self.diff.iter().take_while(|(&t, _)| t < start) { cur += d; }
        // scan change points within [start, end)
        for (_, &d) in self.diff.range(start..end) {
            if cur + 1 > self.lanes as i32 { return false; }
            cur += d;
        }
        // tail segment
        if cur + 1 > self.lanes as i32 { return false; }
        true
    }

    fn place(&mut self, start: u64, end: u64) {
        if start >= end { return; }
        *self.diff.entry(start).or_insert(0) += 1;
        *self.diff.entry(end).or_insert(0) -= 1;
    }
}

fn bytes_to_us(bytes: u64, bps: f64) -> u64 {
    if bps <= 0.0 { return u64::MAX / 4; }
    let bits = (bytes as f64) * 8.0;
    let secs = bits / bps;
    let ms = (secs * 1000.0).ceil() as u64;
    ms.max(1)
}