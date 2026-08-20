//! Fat-Tree SGLB: load-aware routing with local, per-hop top-K selection.
//!
//! Adapts SGLB's spine-leaf design to the 3-tier fat-tree topology.
//! Each switch layer independently selects the least-loaded next-hop:
//!
//! - **Intra-pod flows** (same pod, different ToR):
//!   ToR selects top-K aggregation switches by link load, hash-selects one.
//!
//! - **Cross-pod flows** (different pods), three independent local decisions:
//!   1. src_tor picks agg_src  (scored by src_tor→agg uplink load)
//!   2. agg_src picks core     (scored by agg→core uplink load)
//!   3. core   picks agg_dst   (scored by core→agg downlink load)
//!
//! Flow bindings are tracked for cleanup and optional periodic remapping.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use petgraph::graph::NodeIndex;
use twox_hash::XxHash64;

use crate::network::flow::FlowId;
use crate::network::routing::{PathCell, FatTreeRouter};
use crate::network::topology::{FatTree, FatTreeTopology, LinkId};
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{JobId, MLJob};
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::system::{SystemModule, TimerId};

use super::convert_node_path_to_links;

pub use crate::spine::SGLBConfig;

const SGLB_REMAP_TIMER_ID: TimerId = 1;

// =============================================================================
// Link cache
// =============================================================================

#[derive(Debug, Clone)]
struct FatTreeLinkCache {
    /// (pod, tor_local, agg_local) → (tor→agg link, agg→tor link)
    tor_agg: HashMap<(usize, usize, usize), (LinkId, LinkId)>,
    /// (pod, agg_local, core_idx) → (agg→core link, core→agg link)
    agg_core: HashMap<(usize, usize, usize), (LinkId, LinkId)>,
    degree_agg: usize,
    degree_core: usize,
}

// =============================================================================
// Flow binding
// =============================================================================

#[derive(Debug, Clone)]
enum FlowKind {
    IntraToR,
    IntraPod { agg_local: usize },
    CrossPod { agg_src_local: usize, core_idx: usize, agg_dst_local: usize },
}

#[derive(Debug, Clone)]
struct FlowBinding {
    kind: FlowKind,
    src_pod: usize,
    src_tor_local: usize,
    dst_pod: usize,
    dst_tor_local: usize,
    path_cell: PathCell,
    tracked_links: Vec<LinkId>,
}

// =============================================================================
// FatTreeSGLBRouter
// =============================================================================

#[derive(Debug)]
pub struct FatTreeSGLBRouter {
    context: Option<MLContext>,
    config: SGLBConfig,
    link_flow_count: HashMap<LinkId, usize>,
    flow_bindings: HashMap<FlowId, FlowBinding>,
    link_cache: RefCell<Option<FatTreeLinkCache>>,
}

impl Clone for FatTreeSGLBRouter {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            config: self.config.clone(),
            link_flow_count: self.link_flow_count.clone(),
            flow_bindings: self.flow_bindings.clone(),
            link_cache: RefCell::new(self.link_cache.borrow().clone()),
        }
    }
}

impl FatTreeSGLBRouter {
    pub fn new(config: SGLBConfig) -> Self {
        Self {
            context: None,
            config,
            link_flow_count: HashMap::new(),
            flow_bindings: HashMap::new(),
            link_cache: RefCell::new(None),
        }
    }

    fn get_link_count(&self, link_id: LinkId) -> usize {
        self.link_flow_count.get(&link_id).copied().unwrap_or(0)
    }

    fn increment_link(&mut self, link_id: LinkId) {
        *self.link_flow_count.entry(link_id).or_insert(0) += 1;
    }

    fn decrement_link(&mut self, link_id: LinkId) {
        if let Some(count) = self.link_flow_count.get_mut(&link_id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.link_flow_count.remove(&link_id);
            }
        }
    }

    fn ensure_link_cache(&self, topo: &impl FatTreeTopology) {
        let mut cache_ref = self.link_cache.borrow_mut();
        if cache_ref.is_some() {
            return;
        }

        let graph = topo.topology();
        let num_pods = topo.num_pods();
        let degree_tor = topo.degree_tor();
        let degree_agg = topo.degree_agg();
        let degree_core = topo.degree_core();

        let mut tor_agg = HashMap::new();
        let mut agg_core = HashMap::new();

        for pod in 0..num_pods {
            for tor_local in 0..degree_tor {
                let tor_node = topo.get_tor(pod, tor_local).unwrap();
                for agg_local in 0..degree_agg {
                    let agg_node = topo.get_agg(pod, agg_local);
                    let up = graph.find_edge(tor_node, agg_node)
                        .expect("tor→agg edge must exist");
                    let down = graph.find_edge(agg_node, tor_node)
                        .expect("agg→tor edge must exist");
                    tor_agg.insert(
                        (pod, tor_local, agg_local),
                        (graph.edge_weight(up).unwrap().id, graph.edge_weight(down).unwrap().id),
                    );
                }
            }
            for agg_local in 0..degree_agg {
                let agg_node = topo.get_agg(pod, agg_local);
                for core_idx in 0..degree_core {
                    let core_node = topo.get_core(core_idx);
                    let up = graph.find_edge(agg_node, core_node)
                        .expect("agg→core edge must exist");
                    let down = graph.find_edge(core_node, agg_node)
                        .expect("core→agg edge must exist");
                    agg_core.insert(
                        (pod, agg_local, core_idx),
                        (graph.edge_weight(up).unwrap().id, graph.edge_weight(down).unwrap().id),
                    );
                }
            }
        }

        *cache_ref = Some(FatTreeLinkCache { tor_agg, agg_core, degree_agg, degree_core });
    }

    fn cache_dims(&self) -> (usize, usize) {
        let c = self.link_cache.borrow();
        let c = c.as_ref().expect("link cache not built");
        (c.degree_agg, c.degree_core)
    }

    fn get_tor_agg_links(&self, pod: usize, tor_local: usize, agg_local: usize) -> (LinkId, LinkId) {
        self.link_cache.borrow().as_ref().unwrap()
            .tor_agg[&(pod, tor_local, agg_local)]
    }

    fn get_agg_core_links(&self, pod: usize, agg_local: usize, core_idx: usize) -> (LinkId, LinkId) {
        self.link_cache.borrow().as_ref().unwrap()
            .agg_core[&(pod, agg_local, core_idx)]
    }

    // ---- Scoring helpers ----

    /// Intra-pod: score agg by upstream + downstream link load through that agg.
    fn score_agg_intrapod(
        &self, pod: usize, src_tor_local: usize, dst_tor_local: usize, agg_local: usize,
    ) -> f64 {
        let (up, _) = self.get_tor_agg_links(pod, src_tor_local, agg_local);
        let (_, down) = self.get_tor_agg_links(pod, dst_tor_local, agg_local);
        self.config.w1 * self.get_link_count(up) as f64
            + self.config.w2 * self.get_link_count(down) as f64
    }

    /// Cross-pod hop 1: score agg_src by src_tor→agg uplink load.
    fn score_agg_src(&self, src_pod: usize, src_tor_local: usize, agg_local: usize) -> f64 {
        let (up, _) = self.get_tor_agg_links(src_pod, src_tor_local, agg_local);
        self.get_link_count(up) as f64
    }

    /// Cross-pod hop 2: score core by agg_src→core uplink load.
    fn score_core(&self, src_pod: usize, agg_src_local: usize, core_idx: usize) -> f64 {
        let (up, _) = self.get_agg_core_links(src_pod, agg_src_local, core_idx);
        self.get_link_count(up) as f64
    }

    /// Cross-pod hop 3: score agg_dst by core→agg_dst downlink load.
    fn score_agg_dst(&self, dst_pod: usize, agg_local: usize, core_idx: usize) -> f64 {
        let (_, down) = self.get_agg_core_links(dst_pod, agg_local, core_idx);
        self.get_link_count(down) as f64
    }

    // ---- Selection helpers ----

    fn top_k(&self, scores: impl Iterator<Item = (f64, usize)>) -> Vec<usize> {
        let mut v: Vec<(f64, usize)> = scores.collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.config.k.min(v.len());
        v.into_iter().take(k).map(|(_, idx)| idx).collect()
    }

    fn hash_select(seed: u64, eligible: &[usize], salt: &[u64]) -> usize {
        if eligible.len() == 1 {
            return eligible[0];
        }
        let mut h = XxHash64::with_seed(seed);
        for v in salt { v.hash(&mut h); }
        eligible[h.finish() as usize % eligible.len()]
    }

    fn remap_hash_select(&self, flow_id: FlowId, eligible: &[usize], hop: u64) -> usize {
        let mut h = XxHash64::with_seed(self.config.seed.wrapping_add(flow_id as u64));
        flow_id.hash(&mut h);
        hop.hash(&mut h);
        eligible[h.finish() as usize % eligible.len()]
    }

    /// Three independent local decisions for cross-pod routing.
    fn select_cross_pod(
        &self,
        src_pod: usize, src_tor_local: usize,
        dst_pod: usize,
        flow_id: FlowId, job_id: JobId,
    ) -> (usize, usize, usize) {
        let (degree_agg, degree_core) = self.cache_dims();

        let eligible_agg_src = self.top_k(
            (0..degree_agg).map(|a| (self.score_agg_src(src_pod, src_tor_local, a), a)),
        );
        let agg_src = Self::hash_select(
            self.config.seed, &eligible_agg_src,
            &[flow_id as u64, job_id as u64, 0],
        );

        let eligible_core = self.top_k(
            (0..degree_core).map(|c| (self.score_core(src_pod, agg_src, c), c)),
        );
        let core = Self::hash_select(
            self.config.seed, &eligible_core,
            &[flow_id as u64, job_id as u64, 1],
        );

        let eligible_agg_dst = self.top_k(
            (0..degree_agg).map(|a| (self.score_agg_dst(dst_pod, a, core), a)),
        );
        let agg_dst = Self::hash_select(
            self.config.seed, &eligible_agg_dst,
            &[flow_id as u64, job_id as u64, 2],
        );

        (agg_src, core, agg_dst)
    }

    fn intrapod_tracked_links(
        &self, pod: usize, src_tor_local: usize, dst_tor_local: usize, agg_local: usize,
    ) -> Vec<LinkId> {
        let (up, _) = self.get_tor_agg_links(pod, src_tor_local, agg_local);
        let (_, down) = self.get_tor_agg_links(pod, dst_tor_local, agg_local);
        vec![up, down]
    }

    fn crosspod_tracked_links(
        &self,
        src_pod: usize, src_tor_local: usize,
        dst_pod: usize, dst_tor_local: usize,
        agg_src: usize, core_idx: usize, agg_dst: usize,
    ) -> Vec<LinkId> {
        let (l1, _) = self.get_tor_agg_links(src_pod, src_tor_local, agg_src);
        let (l2, _) = self.get_agg_core_links(src_pod, agg_src, core_idx);
        let (_, l3) = self.get_agg_core_links(dst_pod, agg_dst, core_idx);
        let (_, l4) = self.get_tor_agg_links(dst_pod, dst_tor_local, agg_dst);
        vec![l1, l2, l3, l4]
    }

    fn host_tor_local(topo: &impl FatTreeTopology, host: NodeIndex) -> (usize, usize) {
        let idx = host.index();
        let hpt = topo.hosts_per_tor();
        let hpp = topo.degree_tor() * hpt;
        let pod = idx / hpp;
        let tor_local = (idx % hpp) / hpt;
        (pod, tor_local)
    }

    // ---- Remapping ----

    pub fn remap_ineligible_flows(&mut self, topo: &FatTree<FatTreeSGLBRouter>) {
        if !self.config.enable_remapping {
            return;
        }
        self.ensure_link_cache(topo);
        let (degree_agg, degree_core) = self.cache_dims();

        let mut to_remap: Vec<FlowId> = Vec::new();

        for (&fid, b) in &self.flow_bindings {
            match &b.kind {
                FlowKind::IntraToR => {}
                FlowKind::IntraPod { agg_local } => {
                    let eligible = self.top_k(
                        (0..degree_agg).map(|a| (
                            self.score_agg_intrapod(b.src_pod, b.src_tor_local, b.dst_tor_local, a), a,
                        )),
                    );
                    if !eligible.contains(agg_local) {
                        to_remap.push(fid);
                    }
                }
                FlowKind::CrossPod { agg_src_local, core_idx, agg_dst_local } => {
                    let e1 = self.top_k(
                        (0..degree_agg).map(|a| (self.score_agg_src(b.src_pod, b.src_tor_local, a), a)),
                    );
                    let e2 = self.top_k(
                        (0..degree_core).map(|c| (self.score_core(b.src_pod, *agg_src_local, c), c)),
                    );
                    let e3 = self.top_k(
                        (0..degree_agg).map(|a| (self.score_agg_dst(b.dst_pod, a, *core_idx), a)),
                    );
                    if !e1.contains(agg_src_local)
                        || !e2.contains(core_idx)
                        || !e3.contains(agg_dst_local)
                    {
                        to_remap.push(fid);
                    }
                }
            }
        }

        for fid in to_remap {
            let b = self.flow_bindings.get(&fid).unwrap();
            let src_pod = b.src_pod;
            let src_tor_local = b.src_tor_local;
            let dst_pod = b.dst_pod;
            let dst_tor_local = b.dst_tor_local;
            let old_links = b.tracked_links.clone();
            let path_cell = b.path_cell.clone();
            let old_kind = b.kind.clone();

            for lid in &old_links { self.decrement_link(*lid); }

            match old_kind {
                FlowKind::IntraPod { agg_local: old_agg } => {
                    let eligible = self.top_k(
                        (0..degree_agg).map(|a| (
                            self.score_agg_intrapod(src_pod, src_tor_local, dst_tor_local, a), a,
                        )),
                    );
                    let new_agg = self.remap_hash_select(fid, &eligible, 0);

                    if new_agg == old_agg {
                        for lid in &old_links { self.increment_link(*lid); }
                        continue;
                    }

                    let new_links = self.intrapod_tracked_links(
                        src_pod, src_tor_local, dst_tor_local, new_agg,
                    );
                    {
                        let mut path = path_cell.path.borrow_mut();
                        // path: [host→tor, tor→agg, agg→tor, tor→host]
                        path[1] = new_links[0];
                        path[2] = new_links[1];
                    }
                    for lid in &new_links { self.increment_link(*lid); }

                    let b = self.flow_bindings.get_mut(&fid).unwrap();
                    b.kind = FlowKind::IntraPod { agg_local: new_agg };
                    b.tracked_links = new_links;
                }
                FlowKind::CrossPod { .. } => {
                    let job_id = self.context.as_ref()
                        .and_then(|ctx| ctx.waiting_flows.borrow().get(&fid).map(|t| t.0))
                        .unwrap_or(0);

                    let (new_as, new_c, new_ad) = self.select_cross_pod(
                        src_pod, src_tor_local, dst_pod, fid, job_id,
                    );
                    let new_links = self.crosspod_tracked_links(
                        src_pod, src_tor_local, dst_pod, dst_tor_local,
                        new_as, new_c, new_ad,
                    );
                    {
                        let mut path = path_cell.path.borrow_mut();
                        // path: [host→tor, tor→agg_src, agg_src→core, core→agg_dst, agg_dst→tor, tor→host]
                        path[1] = new_links[0];
                        path[2] = new_links[1];
                        path[3] = new_links[2];
                        path[4] = new_links[3];
                    }
                    for lid in &new_links { self.increment_link(*lid); }

                    let b = self.flow_bindings.get_mut(&fid).unwrap();
                    b.kind = FlowKind::CrossPod {
                        agg_src_local: new_as, core_idx: new_c, agg_dst_local: new_ad,
                    };
                    b.tracked_links = new_links;
                }
                FlowKind::IntraToR => unreachable!(),
            }
        }
    }

    pub fn get_stats(&self) -> FatTreeSGLBStats {
        FatTreeSGLBStats {
            active_flows: self.flow_bindings.len(),
            links_with_flows: self.link_flow_count.len(),
            max_link_count: self.link_flow_count.values().copied().max().unwrap_or(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FatTreeSGLBStats {
    pub active_flows: usize,
    pub links_with_flows: usize,
    pub max_link_count: usize,
}

// =============================================================================
// FatTreeRouter impl
// =============================================================================

impl FatTreeRouter for FatTreeSGLBRouter {
    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn route(
        &mut self, topo: &impl FatTreeTopology,
        src: NodeIndex, dst: NodeIndex, flow_id: FlowId,
    ) -> PathCell {
        self.ensure_link_cache(topo);

        let src_tor = topo.get_host_tor(src);
        let dst_tor = topo.get_host_tor(dst);
        let (src_pod, src_tor_local) = Self::host_tor_local(topo, src);
        let (dst_pod, dst_tor_local) = Self::host_tor_local(topo, dst);

        // Intra-ToR: direct path, no switch choice
        if src_tor == dst_tor {
            let path = convert_node_path_to_links(topo, &[src, src_tor, dst]);
            let pc = PathCell { path: Rc::new(RefCell::new(path)) };
            self.flow_bindings.insert(flow_id, FlowBinding {
                kind: FlowKind::IntraToR,
                src_pod, src_tor_local, dst_pod, dst_tor_local,
                path_cell: pc.clone(), tracked_links: vec![],
            });
            return pc;
        }

        let job_id = self.context.as_ref()
            .and_then(|ctx| ctx.waiting_flows.borrow().get(&flow_id).map(|t| t.0))
            .unwrap_or(0);

        if src_pod == dst_pod {
            // Intra-pod: pick agg by upstream + downstream load
            let (degree_agg, _) = self.cache_dims();
            let eligible = self.top_k(
                (0..degree_agg).map(|a| (
                    self.score_agg_intrapod(src_pod, src_tor_local, dst_tor_local, a), a,
                )),
            );
            let agg_local = Self::hash_select(
                self.config.seed, &eligible,
                &[src.index() as u64, dst.index() as u64, job_id as u64],
            );

            let agg_node = topo.get_agg(src_pod, agg_local);
            let path = convert_node_path_to_links(topo, &[src, src_tor, agg_node, dst_tor, dst]);
            let tracked = self.intrapod_tracked_links(src_pod, src_tor_local, dst_tor_local, agg_local);
            for lid in &tracked { self.increment_link(*lid); }

            let pc = PathCell { path: Rc::new(RefCell::new(path)) };
            self.flow_bindings.insert(flow_id, FlowBinding {
                kind: FlowKind::IntraPod { agg_local },
                src_pod, src_tor_local, dst_pod, dst_tor_local,
                path_cell: pc.clone(), tracked_links: tracked,
            });
            pc
        } else {
            // Cross-pod: three independent local decisions
            let (agg_src, core_idx, agg_dst) = self.select_cross_pod(
                src_pod, src_tor_local, dst_pod, flow_id, job_id,
            );

            let agg_src_node = topo.get_agg(src_pod, agg_src);
            let core_node = topo.get_core(core_idx);
            let agg_dst_node = topo.get_agg(dst_pod, agg_dst);
            let path = convert_node_path_to_links(
                topo, &[src, src_tor, agg_src_node, core_node, agg_dst_node, dst_tor, dst],
            );
            let tracked = self.crosspod_tracked_links(
                src_pod, src_tor_local, dst_pod, dst_tor_local,
                agg_src, core_idx, agg_dst,
            );
            for lid in &tracked { self.increment_link(*lid); }

            let pc = PathCell { path: Rc::new(RefCell::new(path)) };
            self.flow_bindings.insert(flow_id, FlowBinding {
                kind: FlowKind::CrossPod { agg_src_local: agg_src, core_idx, agg_dst_local: agg_dst },
                src_pod, src_tor_local, dst_pod, dst_tor_local,
                path_cell: pc.clone(), tracked_links: tracked,
            });
            pc
        }
    }

    fn complete_flow(&mut self, flow_id: FlowId) {
        if let Some(binding) = self.flow_bindings.remove(&flow_id) {
            for lid in binding.tracked_links {
                self.decrement_link(lid);
            }
        }
    }
}

// =============================================================================
// FatTreeSGLBSystem (standalone, no migrations)
// =============================================================================

#[derive(Debug)]
pub struct FatTreeSGLBSystem {
    config: SGLBConfig,
    timer_started: bool,
}

impl FatTreeSGLBSystem {
    pub fn new(config: SGLBConfig) -> Self {
        Self { config, timer_started: false }
    }

    fn schedule_remap_timer(&self, ctx: &MLContext) {
        if self.config.remap_interval_us > 0 {
            ctx.schedule_timer(self.config.remap_interval_us, SGLB_REMAP_TIMER_ID);
        }
    }
}

impl<S, FS> SystemModule<FatTree<FatTreeSGLBRouter>, S, FS> for FatTreeSGLBSystem
where
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_init(
        &mut self, ctx: &MLContext,
        _topo: &FatTree<FatTreeSGLBRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {
        println!(
            "[FatTreeSGLB] Initialized: K={}, w1={}, w2={}, remapping={}, remap_interval={}us",
            self.config.k, self.config.w1, self.config.w2,
            self.config.enable_remapping, self.config.remap_interval_us,
        );
        if self.config.enable_remapping && self.config.remap_interval_us > 0 {
            self.schedule_remap_timer(ctx);
            self.timer_started = true;
        }
    }

    fn on_job_scheduled(
        &mut self, _now_us: u64, _ctx: &MLContext, _job_id: JobId, _job: &MLJob,
        _topo: &FatTree<FatTreeSGLBRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {}

    fn on_job_completed(
        &mut self, _now_us: u64, _ctx: &MLContext, _job_id: JobId, _job: &MLJob,
        _topo: &FatTree<FatTreeSGLBRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {}

    fn on_reconfigure(
        &mut self, _now_us: u64, _ctx: &MLContext,
        topo: &FatTree<FatTreeSGLBRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) -> Option<crate::simulator::system::MigrationPlan> {
        if self.config.enable_remapping && self.config.remap_interval_us == 0 {
            topo.router.borrow_mut().remap_ineligible_flows(topo);
        }
        None
    }

    fn on_timer(
        &mut self, _now_us: u64, ctx: &MLContext, timer_id: TimerId,
        topo: &FatTree<FatTreeSGLBRouter>, _scheduler: &mut S, _flow_scheduler: &mut FS,
    ) {
        if timer_id != SGLB_REMAP_TIMER_ID {
            return;
        }
        if self.config.enable_remapping {
            topo.router.borrow_mut().remap_ineligible_flows(topo);
        }
        self.schedule_remap_timer(ctx);
    }
}
