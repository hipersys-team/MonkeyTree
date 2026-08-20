//! SGLB (Spine Group Load Balancing) for rail-optimized topologies.
//!
//! Adapts the spine-leaf SGLB design to the three-layer rail topology
//! (hosts → rail switches → spine switches).
//!
//! The only routing decision in a rail topology is **spine selection for
//! cross-pod flows**: a cross-pod flow always traverses
//! `src → src_rail → spine → dst_rail → dst`, and SGLB picks the spine.
//! Intra-pod flows (same host, same rail, or different rail within a pod)
//! never touch the spine layer and are routed on their fixed path.
//!
//! For a cross-pod flow, the "rail" plays the role that the "leaf" plays in
//! spine-leaf SGLB. The source rail is determined by `(src_pod, src_gpu_offset)`
//! and the destination rail by `(dst_pod, dst_gpu_offset)`. SGLB:
//! - Scores each spine by the flow load on `src_rail→spine` and `spine→dst_rail`.
//! - Keeps the top-K least-loaded spines eligible.
//! - Uses flow-consistent hashing to pick among eligible spines.
//! - Optionally remaps in-flight cross-pod flows when their spine leaves the top-K.
//!
//! ## Components
//!
//! - [`RailSGLBRouter`]: per-flow load-aware spine selection for cross-pod flows.
//! - [`RailSGLBSystem`]: system module that schedules periodic remap timers.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use petgraph::graph::NodeIndex;
use twox_hash::XxHash64;

use crate::network::flow::FlowId;
use crate::network::routing::{Path, PathCell};
use crate::network::topology::LinkId;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::ml_job::{JobId, MLJob};
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::system::{SystemModule, TimerId};

use super::routing::build_rail_node_path;
use super::topology::{RailTopology, RailTree, RailTreeRouter};

/// Reuse the spine-leaf SGLB configuration verbatim.
pub use crate::spine::SGLBConfig;

/// Timer ID for periodic eligibility recomputation.
const SGLB_REMAP_TIMER_ID: TimerId = 1;

// =============================================================================
// Flow binding
// =============================================================================

#[derive(Debug, Clone)]
enum FlowKind {
    /// Intra-pod flow (same host, same rail, or different rail in same pod).
    /// Never uses a spine; nothing to score or remap.
    IntraPod,
    /// Cross-pod flow routed through the selected spine.
    CrossPod { spine_idx: usize },
}

#[derive(Debug, Clone)]
struct FlowBinding {
    kind: FlowKind,
    /// Global rail index of the source rail (src_pod * rails_per_pod + src_gpu_offset).
    src_rail_global: usize,
    /// Global rail index of the destination rail.
    dst_rail_global: usize,
    /// The path cell (for in-place remapping).
    path_cell: PathCell,
    /// Spine links used by this flow, for load counting: [src_rail→spine, spine→dst_rail].
    /// Empty for intra-pod flows.
    links: Vec<LinkId>,
}

// =============================================================================
// RailSGLBRouter
// =============================================================================

/// Load-aware spine selection with top-K eligibility for rail topologies.
#[derive(Debug)]
pub struct RailSGLBRouter {
    context: Option<MLContext>,
    config: SGLBConfig,
    /// Active flow count per link.
    link_flow_count: HashMap<LinkId, usize>,
    /// Flow bindings for cleanup and remapping.
    flow_bindings: HashMap<FlowId, FlowBinding>,
    /// Cached link IDs for each (rail_global, spine_idx) pair:
    /// (rail_global, spine_idx) -> (rail→spine link, spine→rail link).
    /// Built lazily on first use.
    link_cache: RefCell<Option<HashMap<(usize, usize), (LinkId, LinkId)>>>,
}

impl RailSGLBRouter {
    pub fn new(config: SGLBConfig) -> Self {
        Self {
            context: None,
            config,
            link_flow_count: HashMap::new(),
            flow_bindings: HashMap::new(),
            link_cache: RefCell::new(None),
        }
    }

    pub fn with_k(k: usize) -> Self {
        Self::new(SGLBConfig::with_k(k))
    }

    fn get_link_count(&self, link_id: LinkId) -> usize {
        self.link_flow_count.get(&link_id).copied().unwrap_or(0)
    }

    /// Link count with a set of links treated as not-present (each occurrence subtracts one).
    /// Used to evaluate a flow's spine eligibility while excluding the flow's own load.
    fn get_link_count_excluding(&self, link_id: LinkId, exclude: &[LinkId]) -> usize {
        let dec = exclude.iter().filter(|&&l| l == link_id).count();
        self.get_link_count(link_id).saturating_sub(dec)
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

    /// Build the link cache: (rail_global, spine_idx) -> (rail→spine, spine→rail).
    fn ensure_link_cache(&self, topo: &impl RailTopology) {
        let mut cache = self.link_cache.borrow_mut();
        if cache.is_some() {
            return;
        }

        let graph = topo.topology();
        let num_pods = topo.num_pods();
        let rails_per_pod = topo.num_rails_per_pod();
        let num_spines = topo.num_spines();

        let mut map = HashMap::new();
        for pod in 0..num_pods {
            for rail_offset in 0..rails_per_pod {
                let rail_global = pod * rails_per_pod + rail_offset;
                let rail_node = topo.get_rail(pod, rail_offset).unwrap();
                for spine_idx in 0..num_spines {
                    let spine_node = topo.get_spine(spine_idx);

                    let up_edge = graph.find_edge(rail_node, spine_node)
                        .expect("rail→spine edge must exist");
                    let up_link = graph.edge_weight(up_edge).unwrap().id;

                    let down_edge = graph.find_edge(spine_node, rail_node)
                        .expect("spine→rail edge must exist");
                    let down_link = graph.edge_weight(down_edge).unwrap().id;

                    map.insert((rail_global, spine_idx), (up_link, down_link));
                }
            }
        }

        *cache = Some(map);
    }

    /// Get (rail→spine, spine→rail) link IDs for a (rail_global, spine_idx) pair.
    fn get_links(&self, rail_global: usize, spine_idx: usize) -> (LinkId, LinkId) {
        self.link_cache.borrow().as_ref().unwrap()
            .get(&(rail_global, spine_idx)).copied()
            .expect("link cache should be populated")
    }

    /// Compute top-K eligible spines for a (src_rail, dst_rail) pair.
    fn compute_eligible_spines(&self, src_rail_global: usize, dst_rail_global: usize, num_spines: usize) -> Vec<usize> {
        self.compute_eligible_spines_excluding(src_rail_global, dst_rail_global, num_spines, &[])
    }

    /// Like `compute_eligible_spines`, but treats the links in `exclude` as not-present.
    /// Used to judge a flow's own spine eligibility without counting the flow's own load
    /// against itself (otherwise a lone, uncontended flow would flag itself for remapping).
    fn compute_eligible_spines_excluding(
        &self,
        src_rail_global: usize,
        dst_rail_global: usize,
        num_spines: usize,
        exclude: &[LinkId],
    ) -> Vec<usize> {
        let mut scored: Vec<(f64, usize)> = (0..num_spines)
            .map(|s| {
                let up = self.get_links(src_rail_global, s).0;
                let down = self.get_links(dst_rail_global, s).1;
                let score = self.config.w1 * self.get_link_count_excluding(up, exclude) as f64
                    + self.config.w2 * self.get_link_count_excluding(down, exclude) as f64;
                (score, s)
            })
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.config.k.min(num_spines).max(1);
        scored.into_iter().take(k).map(|(_, s)| s).collect()
    }

    /// Flow-consistent hash selection among eligible spines.
    fn hash_select_spine(&self, src: NodeIndex, dst: NodeIndex, job_id: JobId, eligible: &[usize]) -> usize {
        if eligible.len() == 1 {
            return eligible[0];
        }

        let mut hasher = XxHash64::with_seed(self.config.seed);
        src.hash(&mut hasher);
        dst.hash(&mut hasher);
        job_id.hash(&mut hasher);
        let hash = hasher.finish() as usize;

        eligible[hash % eligible.len()]
    }

    /// Convert a node path into a link-id path.
    fn convert_path(&self, topo: &impl RailTopology, node_path: &[NodeIndex]) -> Path {
        let graph = topo.topology();
        let mut link_path = Vec::with_capacity(node_path.len().saturating_sub(1));
        for window in node_path.windows(2) {
            let edge = graph.find_edge(window[0], window[1])
                .unwrap_or_else(|| panic!("RailSGLB: nonexistent edge {:?} -> {:?}", window[0], window[1]));
            link_path.push(graph.edge_weight(edge).unwrap().id);
        }
        link_path
    }

    /// Global rail index for a host: host_pod * rails_per_pod + host_gpu_offset.
    fn host_rail_global(topo: &impl RailTopology, host: NodeIndex) -> usize {
        topo.host_pod(host) * topo.num_rails_per_pod() + topo.host_gpu_offset(host)
    }

    /// Remap in-flight cross-pod flows whose spine is no longer in top-K.
    pub fn remap_ineligible_flows(&mut self, topo: &impl RailTopology) {
        if !self.config.enable_remapping {
            return;
        }

        self.ensure_link_cache(topo);
        let num_spines = topo.num_spines();

        // Collect cross-pod flows that need remapping. A flow is only remapped when its
        // spine is no longer in the top-K *due to other traffic* — we exclude the flow's
        // own load from the judgment so an uncontended flow never bounces off its spine.
        let mut to_remap: Vec<FlowId> = Vec::new();
        for (&flow_id, binding) in &self.flow_bindings {
            if let FlowKind::CrossPod { spine_idx } = binding.kind {
                let eligible = self.compute_eligible_spines_excluding(
                    binding.src_rail_global, binding.dst_rail_global, num_spines, &binding.links,
                );
                if !eligible.contains(&spine_idx) {
                    to_remap.push(flow_id);
                }
            }
        }

        for flow_id in to_remap {
            let binding = self.flow_bindings.get(&flow_id).unwrap();
            let old_spine = match binding.kind {
                FlowKind::CrossPod { spine_idx } => spine_idx,
                FlowKind::IntraPod => continue,
            };
            let src_rail_global = binding.src_rail_global;
            let dst_rail_global = binding.dst_rail_global;
            let old_links = binding.links.clone();
            let path_cell = binding.path_cell.clone();

            // Temporarily remove old counts so the eligible set reflects this flow moving.
            for link_id in &old_links {
                self.decrement_link(*link_id);
            }

            let eligible = self.compute_eligible_spines(src_rail_global, dst_rail_global, num_spines);

            // Remap-specific hash (keeps remaps deterministic but decorrelated from initial placement).
            let mut hasher = XxHash64::with_seed(self.config.seed.wrapping_add(flow_id as u64));
            flow_id.hash(&mut hasher);
            src_rail_global.hash(&mut hasher);
            dst_rail_global.hash(&mut hasher);
            let hash = hasher.finish() as usize;
            let new_spine = eligible[hash % eligible.len()];

            if new_spine == old_spine {
                // No change: restore old counts.
                for link_id in &old_links {
                    self.increment_link(*link_id);
                }
                continue;
            }

            // Compute new spine links and rewrite the path in place.
            let (new_up, _) = self.get_links(src_rail_global, new_spine);
            let (_, new_down) = self.get_links(dst_rail_global, new_spine);
            let new_links = vec![new_up, new_down];

            {
                let mut path = path_cell.path.borrow_mut();
                // Cross-pod path links: [host→src_rail, src_rail→spine, spine→dst_rail, dst_rail→host]
                if path.len() >= 4 {
                    path[1] = new_up;
                    path[2] = new_down;
                }
            }

            for link_id in &new_links {
                self.increment_link(*link_id);
            }

            let binding = self.flow_bindings.get_mut(&flow_id).unwrap();
            binding.kind = FlowKind::CrossPod { spine_idx: new_spine };
            binding.links = new_links;
        }
    }

    pub fn get_stats(&self) -> RailSGLBStats {
        RailSGLBStats {
            active_flows: self.flow_bindings.len(),
            links_with_flows: self.link_flow_count.len(),
            max_link_count: self.link_flow_count.values().copied().max().unwrap_or(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RailSGLBStats {
    pub active_flows: usize,
    pub links_with_flows: usize,
    pub max_link_count: usize,
}

impl Default for RailSGLBRouter {
    fn default() -> Self {
        Self::new(SGLBConfig::default())
    }
}

impl Clone for RailSGLBRouter {
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

impl RailTreeRouter for RailSGLBRouter {
    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn route(&mut self, topo: &impl RailTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        self.ensure_link_cache(topo);

        let src_pod = topo.host_pod(src);
        let dst_pod = topo.host_pod(dst);

        let src_rail_global = Self::host_rail_global(topo, src);
        let dst_rail_global = Self::host_rail_global(topo, dst);

        // Intra-pod flows never touch the spine: build the fixed path, no spine choice.
        if src_pod == dst_pod {
            let nodes = build_rail_node_path(topo, src, dst, None);
            let path = self.convert_path(topo, &nodes);
            let path_cell = PathCell { path: Rc::new(RefCell::new(path)) };

            self.flow_bindings.insert(flow_id, FlowBinding {
                kind: FlowKind::IntraPod,
                src_rail_global,
                dst_rail_global,
                path_cell: path_cell.clone(),
                links: vec![],
            });

            return path_cell;
        }

        // Cross-pod: compute eligible spines and select one.
        let num_spines = topo.num_spines();
        let eligible = self.compute_eligible_spines(src_rail_global, dst_rail_global, num_spines);

        let job_id = self.context.as_ref()
            .and_then(|ctx| ctx.waiting_flows.borrow().get(&flow_id).map(|t| t.0))
            .unwrap_or(0);

        let spine_idx = self.hash_select_spine(src, dst, job_id, &eligible);

        let spine_node = topo.get_spine(spine_idx);
        let nodes = build_rail_node_path(topo, src, dst, Some(spine_node));
        let path = self.convert_path(topo, &nodes);
        let path_cell = PathCell { path: Rc::new(RefCell::new(path)) };

        // Track the two spine links for load counting.
        let (up_link, _) = self.get_links(src_rail_global, spine_idx);
        let (_, down_link) = self.get_links(dst_rail_global, spine_idx);
        let tracked = vec![up_link, down_link];
        for link_id in &tracked {
            self.increment_link(*link_id);
        }

        self.flow_bindings.insert(flow_id, FlowBinding {
            kind: FlowKind::CrossPod { spine_idx },
            src_rail_global,
            dst_rail_global,
            path_cell: path_cell.clone(),
            links: tracked,
        });

        path_cell
    }

    fn complete_flow(&mut self, flow_id: FlowId) {
        if let Some(binding) = self.flow_bindings.remove(&flow_id) {
            for link_id in binding.links {
                self.decrement_link(link_id);
            }
        }
    }
}

// =============================================================================
// RailSGLBSystem
// =============================================================================

/// System module that coordinates rail SGLB routing across the job lifecycle.
///
/// Schedules a periodic timer to remap in-flight cross-pod flows whose spine
/// has left the top-K eligible set. Does not trigger job migrations.
#[derive(Debug)]
pub struct RailSGLBSystem {
    config: SGLBConfig,
    log_stats: bool,
    timer_started: bool,
}

impl RailSGLBSystem {
    pub fn new(config: SGLBConfig) -> Self {
        Self { config, log_stats: true, timer_started: false }
    }

    pub fn with_k(k: usize) -> Self {
        Self::new(SGLBConfig::with_k(k))
    }

    pub fn without_logging(mut self) -> Self {
        self.log_stats = false;
        self
    }

    pub fn config(&self) -> &SGLBConfig {
        &self.config
    }

    fn schedule_remap_timer(&self, ctx: &MLContext) {
        if self.config.remap_interval_us > 0 {
            ctx.schedule_timer(self.config.remap_interval_us, SGLB_REMAP_TIMER_ID);
        }
    }
}

impl Default for RailSGLBSystem {
    fn default() -> Self {
        Self::new(SGLBConfig::default())
    }
}

impl<S, FS> SystemModule<RailTree<RailSGLBRouter>, S, FS> for RailSGLBSystem
where
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_init(
        &mut self,
        ctx: &MLContext,
        _topo: &RailTree<RailSGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        println!("[RailSGLB] Initialized with K={}, w1={}, w2={}, remapping={}, remap_interval={}us",
            self.config.k, self.config.w1, self.config.w2,
            self.config.enable_remapping, self.config.remap_interval_us);

        if self.config.enable_remapping && self.config.remap_interval_us > 0 {
            self.schedule_remap_timer(ctx);
            self.timer_started = true;
        }
    }

    fn on_job_scheduled(
        &mut self,
        now_us: u64,
        _ctx: &MLContext,
        job_id: JobId,
        _job: &MLJob,
        topo: &RailTree<RailSGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        if self.log_stats {
            let stats = topo.router.borrow().get_stats();
            println!("[RailSGLB] t={} Job {} scheduled. Active flows: {}, max link count: {}",
                now_us, job_id, stats.active_flows, stats.max_link_count);
        }
    }

    fn on_job_completed(
        &mut self,
        now_us: u64,
        _ctx: &MLContext,
        job_id: JobId,
        _job: &MLJob,
        topo: &RailTree<RailSGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        if self.log_stats {
            let stats = topo.router.borrow().get_stats();
            println!("[RailSGLB] t={} Job {} completed. Active flows: {}, max link count: {}",
                now_us, job_id, stats.active_flows, stats.max_link_count);
        }
    }

    fn on_reconfigure(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        topo: &RailTree<RailSGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) -> Option<crate::simulator::system::MigrationPlan> {
        // Only remap immediately on reconfigure if the periodic timer is disabled;
        // otherwise the timer handles it.
        if self.config.enable_remapping && self.config.remap_interval_us == 0 {
            topo.router.borrow_mut().remap_ineligible_flows(topo);
        }
        None // SGLB doesn't trigger migrations.
    }

    fn on_timer(
        &mut self,
        _now_us: u64,
        ctx: &MLContext,
        timer_id: TimerId,
        topo: &RailTree<RailSGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::topology::Topology;

    // 2 pods, 2 blocks/pod, block_size 4 => 16 GPUs.
    // Pod 0 = hosts 0..8 (block0: 0..4, block1: 4..8). Pod 1 = hosts 8..16.
    fn make_topo(num_spines: usize, config: SGLBConfig) -> RailTree<RailSGLBRouter> {
        RailTree::new(4, 2, 2, num_spines, 400.0e9, 400.0e9, 400.0e9, RailSGLBRouter::new(config))
    }

    #[test]
    fn cross_pod_selects_spine_and_tracks_two_links() {
        let topo = make_topo(4, SGLBConfig::default());
        let src = topo.get_host_by_index(0).unwrap();  // pod 0, gpu offset 0 => rail_global 0
        let dst = topo.get_host_by_index(8).unwrap();  // pod 1, gpu offset 0 => rail_global 4

        let cell = topo.route(src, dst, 1);
        let path = cell.path.borrow();
        // host->src_rail, src_rail->spine, spine->dst_rail, dst_rail->host
        assert_eq!(path.len(), 4, "cross-pod path must have 4 links");

        let r = topo.router.borrow();
        let b = r.flow_bindings.get(&1).expect("binding must exist");
        assert_eq!(b.src_rail_global, 0);
        assert_eq!(b.dst_rail_global, 4);
        let spine_idx = match b.kind {
            FlowKind::CrossPod { spine_idx } => spine_idx,
            FlowKind::IntraPod => panic!("expected cross-pod flow"),
        };
        assert!(spine_idx < 4);

        let up = r.get_links(0, spine_idx).0;   // src_rail -> spine
        let down = r.get_links(4, spine_idx).1; // spine -> dst_rail
        assert_eq!(b.links, vec![up, down], "must track exactly the two spine links");
        assert_eq!(path[1], up, "path[1] must be src_rail->spine");
        assert_eq!(path[2], down, "path[2] must be spine->dst_rail");
        assert_eq!(r.get_link_count(up), 1);
        assert_eq!(r.get_link_count(down), 1);
    }

    #[test]
    fn intra_pod_uses_no_spine_and_tracks_nothing() {
        let topo = make_topo(4, SGLBConfig::default());
        // host 0 (pod0, block0, gpu0) -> host 5 (pod0, block1, gpu1): same pod, different rail.
        let src = topo.get_host_by_index(0).unwrap();
        let dst = topo.get_host_by_index(5).unwrap();

        let cell = topo.route(src, dst, 7);
        // src -> bridge(intra-host) -> rail -> dst  => 3 links, none crossing a spine.
        assert!(cell.path.borrow().len() >= 2);

        let r = topo.router.borrow();
        let b = r.flow_bindings.get(&7).unwrap();
        assert!(matches!(b.kind, FlowKind::IntraPod));
        assert!(b.links.is_empty(), "intra-pod flow must track no spine links");
        assert!(r.link_flow_count.is_empty(), "no spine link load for intra-pod traffic");
    }

    #[test]
    fn same_rail_intra_pod_is_intra() {
        let topo = make_topo(4, SGLBConfig::default());
        // host 0 (pod0, block0, gpu0) -> host 4 (pod0, block1, gpu0): same pod, same rail.
        let src = topo.get_host_by_index(0).unwrap();
        let dst = topo.get_host_by_index(4).unwrap();
        topo.route(src, dst, 3);
        let r = topo.router.borrow();
        assert!(matches!(r.flow_bindings.get(&3).unwrap().kind, FlowKind::IntraPod));
        assert!(r.link_flow_count.is_empty());
    }

    #[test]
    fn complete_flow_releases_link_load() {
        let topo = make_topo(4, SGLBConfig::default());
        let src = topo.get_host_by_index(0).unwrap();
        let dst = topo.get_host_by_index(8).unwrap();
        topo.route(src, dst, 1);
        assert_eq!(topo.router.borrow().link_flow_count.values().sum::<usize>(), 2);

        topo.complete_flow(1);
        let r = topo.router.borrow();
        assert!(r.flow_bindings.is_empty(), "binding removed on completion");
        assert!(r.link_flow_count.is_empty(), "link load released on completion");
    }

    #[test]
    fn load_spreads_across_multiple_spines() {
        // K=2 with 4 spines: flows from the same rail pair must not all pile on one spine.
        let topo = make_topo(4, SGLBConfig::with_k(2));
        let src = topo.get_host_by_index(0).unwrap();  // rail_global 0
        let dst = topo.get_host_by_index(8).unwrap();  // rail_global 4

        let n = 8;
        for fid in 0..n {
            topo.route(src, dst, fid as FlowId);
        }

        let r = topo.router.borrow();
        // Count distinct spines used by inspecting the src_rail(0)->spine up-links.
        let mut used = 0usize;
        let mut max_on_one = 0usize;
        for s in 0..4 {
            let up = r.get_links(0, s).0;
            let c = r.get_link_count(up);
            if c > 0 { used += 1; }
            max_on_one = max_on_one.max(c);
        }
        assert!(used >= 2, "expected load spread over >=2 spines, got {}", used);
        assert!(max_on_one < n, "no single spine should carry all {} flows (max={})", n, max_on_one);
        // Total tracked up-link load must equal number of cross-pod flows.
        let total: usize = (0..4).map(|s| r.get_link_count(r.get_links(0, s).0)).sum();
        assert_eq!(total, n);
    }

    #[test]
    fn remap_moves_flow_whose_spine_left_top_k() {
        // K=1: only the single least-loaded spine is eligible.
        let topo = make_topo(4, SGLBConfig::with_k(1));
        let src = topo.get_host_by_index(0).unwrap();  // rail_global 0
        let dst = topo.get_host_by_index(8).unwrap();  // rail_global 4

        topo.route(src, dst, 1);
        let old_spine = match topo.router.borrow().flow_bindings.get(&1).unwrap().kind {
            FlowKind::CrossPod { spine_idx } => spine_idx,
            FlowKind::IntraPod => panic!(),
        };

        // Inject heavy *external* load onto the chosen spine's up-link so it drops out of top-1.
        {
            let mut r = topo.router.borrow_mut();
            let up_old = r.get_links(0, old_spine).0;
            for _ in 0..5 {
                r.increment_link(up_old);
            }
        }

        topo.router.borrow_mut().remap_ineligible_flows(&topo);

        let r = topo.router.borrow();
        let b = r.flow_bindings.get(&1).unwrap();
        let new_spine = match b.kind {
            FlowKind::CrossPod { spine_idx } => spine_idx,
            FlowKind::IntraPod => panic!(),
        };
        assert_ne!(new_spine, old_spine, "flow should have been remapped off the overloaded spine");

        // The path and tracked links must reflect the new spine.
        let up_new = r.get_links(0, new_spine).0;
        let down_new = r.get_links(4, new_spine).1;
        assert_eq!(b.links, vec![up_new, down_new]);
        let path = b.path_cell.path.borrow();
        assert_eq!(path[1], up_new);
        assert_eq!(path[2], down_new);
    }

    #[test]
    fn new_arrival_can_remap_an_existing_flow() {
        // Established flow A is stable on its own, but a newly-arriving flow B that loads
        // A's spine (via a shared rail link) pushes A out of the top-K, so the next remap
        // moves A. This is the intended behavior; the self-exclusion fix does NOT prevent it.
        //
        // rail_global 0 = pod0 gpu-offset 0 (hosts 0, 4);  rail_global 4 = pod1 gpu-offset 0 (hosts 8, 12)
        // rail_global 7 = pod1 gpu-offset 3 (host 11)
        let spine_of = |t: &RailTree<RailSGLBRouter>, fid: FlowId| match t.router.borrow().flow_bindings.get(&fid).unwrap().kind {
            FlowKind::CrossPod { spine_idx } => Some(spine_idx),
            FlowKind::IntraPod => None,
        };

        // --- Control: no new arrival -> A stays put ---
        let control = make_topo(2, SGLBConfig::with_k(1));
        let h0 = control.get_host_by_index(0).unwrap();
        let h8 = control.get_host_by_index(8).unwrap();
        let h11 = control.get_host_by_index(11).unwrap();
        control.route(h0, h11, 10);          // pre-flow occupies spine0 on rail0->rail7
        control.route(h0, h8, 1);            // flow A: rail0->rail4, forced onto spine1
        let a_spine_control = spine_of(&control, 1).unwrap();
        control.router.borrow_mut().remap_ineligible_flows(&control);
        assert_eq!(spine_of(&control, 1), Some(a_spine_control),
            "without a new arrival, A must stay on its spine");

        // --- With a new arrival B sharing rail0 -> it tips A's spine out of the top-K ---
        let topo = make_topo(2, SGLBConfig::with_k(1));
        let h0 = topo.get_host_by_index(0).unwrap();
        let h4 = topo.get_host_by_index(4).unwrap();   // also rail_global 0 (pod0 gpu-offset 0, block1)
        let h8 = topo.get_host_by_index(8).unwrap();
        let h11 = topo.get_host_by_index(11).unwrap();
        topo.route(h0, h11, 10);             // pre-flow on spine0
        topo.route(h0, h8, 1);               // flow A onto spine1
        let a_spine_before = spine_of(&topo, 1).unwrap();
        topo.route(h4, h11, 2);              // NEW flow B arrives on rail0->rail7, loads rail0->spine1
        topo.router.borrow_mut().remap_ineligible_flows(&topo);
        let a_spine_after = spine_of(&topo, 1).unwrap();
        assert_ne!(a_spine_after, a_spine_before,
            "B's arrival should have caused A to be remapped off its spine");
    }

    #[test]
    fn lone_flow_is_never_remapped() {
        // A flow with no competing traffic must never be remapped at all:
        // its own load must not count against its spine's eligibility.
        let topo = make_topo(4, SGLBConfig::with_k(2));
        let src = topo.get_host_by_index(0).unwrap();
        let dst = topo.get_host_by_index(8).unwrap();
        topo.route(src, dst, 1);

        let spine_of = |t: &RailTree<RailSGLBRouter>| match t.router.borrow().flow_bindings.get(&1).unwrap().kind {
            FlowKind::CrossPod { spine_idx } => spine_idx,
            FlowKind::IntraPod => panic!(),
        };
        let initial = spine_of(&topo);

        for _ in 0..10 {
            topo.router.borrow_mut().remap_ineligible_flows(&topo);
            assert_eq!(spine_of(&topo), initial,
                "uncontended lone flow must stay on its initial spine");
        }
    }
}
