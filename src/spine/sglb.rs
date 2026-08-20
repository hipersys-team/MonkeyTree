//! SGLB (Spine Group Load Balancing) for spine-leaf topologies.
//!
//! SGLB is a load-aware routing system that:
//! - Scores spines based on current flow counts on their links
//! - Maintains top-K eligible spines per destination ToR
//! - Uses flow-consistent hashing to select among eligible spines
//! - Optionally remaps in-flight flows when eligibility changes
//!
//! ## Components
//!
//! - [`SGLBRouter`]: Router that makes per-flow spine selection based on load
//! - [`SGLBSystemModule`]: System module that configures and coordinates SGLB

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
use crate::simulator::ml_job::JobId;
use crate::simulator::ml_simulator::MLContext;
use crate::simulator::system::{SystemModule, TimerId};

use super::topology::{SpineTree, SpineTreeRouter, SpineTreeTopology};

// =============================================================================
// Configuration
// =============================================================================

/// Timer ID for periodic eligibility recomputation
const SGLB_REMAP_TIMER_ID: TimerId = 1;

/// Configuration for SGLB routing.
#[derive(Debug, Clone)]
pub struct SGLBConfig {
    /// Number of top spines to consider as eligible (top-K).
    /// If K >= num_spines, all spines are always eligible.
    pub k: usize,
    /// Weight for upstream link (src_tor → spine) in scoring.
    pub w1: f64,
    /// Weight for downstream link (spine → dst_tor) in scoring.
    pub w2: f64,
    /// Hash seed for flow-consistent spine selection.
    pub seed: u64,
    /// Whether to remap in-flight flows when their spine becomes ineligible.
    pub enable_remapping: bool,
    /// Interval (in microseconds) for periodic eligibility recomputation.
    /// If 0, periodic remapping is disabled (only remap on job events).
    pub remap_interval_us: u64,
}

impl Default for SGLBConfig {
    fn default() -> Self {
        Self {
            k: 4,
            w1: 1.0,
            w2: 1.0,
            seed: 0x5A3B_1C2D_4E5F_6A7Bu64,
            enable_remapping: true,
            remap_interval_us: 10_000, // 10ms default interval
        }
    }
}

impl SGLBConfig {
    /// Create config with specified K value.
    pub fn with_k(k: usize) -> Self {
        Self { k, ..Default::default() }
    }

    /// Create config with custom weights.
    pub fn with_weights(k: usize, w1: f64, w2: f64) -> Self {
        Self { k, w1, w2, ..Default::default() }
    }

    /// Create config with a specific remap interval.
    pub fn with_remap_interval(k: usize, interval_us: u64) -> Self {
        Self { 
            k, 
            remap_interval_us: interval_us,
            ..Default::default() 
        }
    }

    /// Disable periodic remapping (only remap on job events).
    pub fn without_periodic_remap(mut self) -> Self {
        self.remap_interval_us = 0;
        self
    }
}

// =============================================================================
// SGLBRouter
// =============================================================================

/// Flow binding information for cleanup and remapping.
#[derive(Debug, Clone)]
struct FlowBinding {
    /// The spine index this flow is using
    spine_idx: usize,
    /// Source leaf index
    src_leaf: usize,
    /// Destination leaf index  
    dst_leaf: usize,
    /// The path cell (for potential remapping)
    path_cell: PathCell,
    /// Link IDs used by this flow (for count tracking)
    links: Vec<LinkId>,
}

/// SGLB Router: load-aware spine selection with top-K eligibility.
///
/// On each routing decision:
/// 1. Computes spine scores based on current flow counts
/// 2. Selects top-K spines as eligible
/// 3. Hashes flow onto one of the eligible spines
/// 4. Tracks flow bindings for cleanup and optional remapping
#[derive(Debug)]
pub struct SGLBRouter {
    context: Option<MLContext>,
    config: SGLBConfig,
    /// Active flow count per link
    link_flow_count: HashMap<LinkId, usize>,
    /// Flow bindings for cleanup and remapping
    flow_bindings: HashMap<FlowId, FlowBinding>,
    /// Cached link IDs for each (leaf, spine) pair: (leaf_idx, spine_idx) -> (up_link, down_link)
    /// Built lazily on first use
    link_cache: RefCell<Option<HashMap<(usize, usize), (LinkId, LinkId)>>>,
}

impl SGLBRouter {
    /// Create a new SGLB router with the given configuration.
    pub fn new(config: SGLBConfig) -> Self {
        Self {
            context: None,
            config,
            link_flow_count: HashMap::new(),
            flow_bindings: HashMap::new(),
            link_cache: RefCell::new(None),
        }
    }

    /// Create a new SGLB router with default configuration.
    pub fn with_k(k: usize) -> Self {
        Self::new(SGLBConfig::with_k(k))
    }

    /// Get current flow count on a link.
    fn get_link_count(&self, link_id: LinkId) -> usize {
        self.link_flow_count.get(&link_id).copied().unwrap_or(0)
    }

    /// Increment flow count on a link.
    fn increment_link(&mut self, link_id: LinkId) {
        *self.link_flow_count.entry(link_id).or_insert(0) += 1;
    }

    /// Decrement flow count on a link.
    fn decrement_link(&mut self, link_id: LinkId) {
        if let Some(count) = self.link_flow_count.get_mut(&link_id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.link_flow_count.remove(&link_id);
            }
        }
    }

    /// Build link cache for fast lookup of (leaf, spine) -> link IDs.
    fn ensure_link_cache<T: SpineTreeTopology>(&self, topo: &T) {
        let mut cache = self.link_cache.borrow_mut();
        if cache.is_some() {
            return;
        }

        let graph = topo.topology();
        let mut map = HashMap::new();

        for leaf_idx in 0..topo.num_leaves() {
            let leaf_node = topo.get_leaf(leaf_idx).unwrap();
            for spine_idx in 0..topo.num_spines() {
                let spine_node = topo.get_spine(spine_idx);

                // Leaf → Spine (upstream)
                let up_edge = graph.find_edge(leaf_node, spine_node)
                    .expect("leaf→spine edge must exist");
                let up_link = graph.edge_weight(up_edge).unwrap().id;

                // Spine → Leaf (downstream)
                let down_edge = graph.find_edge(spine_node, leaf_node)
                    .expect("spine→leaf edge must exist");
                let down_link = graph.edge_weight(down_edge).unwrap().id;

                map.insert((leaf_idx, spine_idx), (up_link, down_link));
            }
        }

        *cache = Some(map);
    }

    /// Get link IDs for a (leaf, spine) pair.
    fn get_links(&self, leaf_idx: usize, spine_idx: usize) -> (LinkId, LinkId) {
        let cache = self.link_cache.borrow();
        cache.as_ref().unwrap().get(&(leaf_idx, spine_idx)).copied()
            .expect("link cache should be populated")
    }

    /// Compute spine score (lower is better).
    /// Score = w1 * count(src_leaf→spine) + w2 * count(spine→dst_leaf)
    fn compute_spine_score(&self, src_leaf_idx: usize, dst_leaf_idx: usize, spine_idx: usize) -> f64 {
        let (up_link, _) = self.get_links(src_leaf_idx, spine_idx);
        let (_, down_link) = self.get_links(dst_leaf_idx, spine_idx);

        let up_count = self.get_link_count(up_link) as f64;
        let down_count = self.get_link_count(down_link) as f64;

        self.config.w1 * up_count + self.config.w2 * down_count
    }

    /// Compute top-K eligible spines for a (src_leaf, dst_leaf) pair.
    fn compute_eligible_spines(&self, src_leaf_idx: usize, dst_leaf_idx: usize, num_spines: usize) -> Vec<usize> {
        let mut scored: Vec<(f64, usize)> = (0..num_spines)
            .map(|s| (self.compute_spine_score(src_leaf_idx, dst_leaf_idx, s), s))
            .collect();

        // Sort by score ascending (lower is better)
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-K
        let k = self.config.k.min(num_spines);
        scored.into_iter().take(k).map(|(_, s)| s).collect()
    }

    /// Hash-based selection among eligible spines.
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

    /// Build the full link path for a flow.
    fn build_path<T: SpineTreeTopology>(&self, topo: &T, src: NodeIndex, dst: NodeIndex, spine_idx: usize) -> (Path, Vec<LinkId>) {
        let src_leaf = topo.get_host_leaf(src);
        let dst_leaf = topo.get_host_leaf(dst);
        let spine = topo.get_spine(spine_idx);

        let nodes = vec![src, src_leaf, spine, dst_leaf, dst];
        let graph = topo.topology();

        let mut link_path = Vec::with_capacity(4);
        for window in nodes.windows(2) {
            let edge = graph.find_edge(window[0], window[1])
                .expect("path edge must exist");
            let link_id = graph.edge_weight(edge).unwrap().id;
            link_path.push(link_id);
        }

        // The inter-leaf links are indices 1 and 2 (src_leaf→spine, spine→dst_leaf)
        let tracked_links = if link_path.len() >= 3 {
            vec![link_path[1], link_path[2]]
        } else {
            vec![]
        };

        (link_path, tracked_links)
    }

    /// Build path for intra-leaf flow (no spine needed).
    fn build_intra_leaf_path<T: SpineTreeTopology>(&self, topo: &T, src: NodeIndex, dst: NodeIndex) -> Path {
        let leaf = topo.get_host_leaf(src);
        let nodes = vec![src, leaf, dst];
        let graph = topo.topology();

        let mut link_path = Vec::with_capacity(2);
        for window in nodes.windows(2) {
            let edge = graph.find_edge(window[0], window[1])
                .expect("path edge must exist");
            link_path.push(graph.edge_weight(edge).unwrap().id);
        }
        link_path
    }

    /// Remap in-flight flows whose spines are no longer in top-K.
    /// Called by the system module on reconfiguration.
    pub fn remap_ineligible_flows<T: SpineTreeTopology>(&mut self, topo: &T) {
        if !self.config.enable_remapping {
            return;
        }

        self.ensure_link_cache(topo);
        let num_spines = topo.num_spines();

        // Collect flows that need remapping
        let mut to_remap: Vec<(FlowId, usize, usize)> = Vec::new(); // (flow_id, src_leaf, dst_leaf)

        for (&flow_id, binding) in &self.flow_bindings {
            // Skip intra-leaf flows
            if binding.src_leaf == binding.dst_leaf {
                continue;
            }

            let eligible = self.compute_eligible_spines(binding.src_leaf, binding.dst_leaf, num_spines);
            if !eligible.contains(&binding.spine_idx) {
                to_remap.push((flow_id, binding.src_leaf, binding.dst_leaf));
            }
        }

        // Remap each flow
        for (flow_id, src_leaf, dst_leaf) in to_remap {
            let binding = self.flow_bindings.get(&flow_id).unwrap();
            let old_spine = binding.spine_idx;
            let old_links = binding.links.clone();
            let path_cell = binding.path_cell.clone();

            // Temporarily decrement old link counts to get accurate eligible set
            for link_id in &old_links {
                self.decrement_link(*link_id);
            }

            // Recompute eligible set and pick new spine
            let eligible = self.compute_eligible_spines(src_leaf, dst_leaf, num_spines);
            
            // Use a remap-specific hash
            let mut hasher = XxHash64::with_seed(self.config.seed.wrapping_add(flow_id as u64));
            flow_id.hash(&mut hasher);
            src_leaf.hash(&mut hasher);
            dst_leaf.hash(&mut hasher);
            let hash = hasher.finish() as usize;
            let new_spine = eligible[hash % eligible.len()];

            // If spine didn't change, just restore old counts and continue
            if new_spine == old_spine {
                for link_id in &old_links {
                    self.increment_link(*link_id);
                }
                continue;
            }

            // Actually remap to new spine
            let src_leaf_node = topo.get_leaf(src_leaf).unwrap();
            let dst_leaf_node = topo.get_leaf(dst_leaf).unwrap();
            let new_spine_node = topo.get_spine(new_spine);
            
            // Get link IDs for new path
            let graph = topo.topology();
            let up_edge = graph.find_edge(src_leaf_node, new_spine_node).unwrap();
            let down_edge = graph.find_edge(new_spine_node, dst_leaf_node).unwrap();
            let new_up_link = graph.edge_weight(up_edge).unwrap().id;
            let new_down_link = graph.edge_weight(down_edge).unwrap().id;
            let new_links = vec![new_up_link, new_down_link];

            // Update path in-place (keeping host links the same)
            {
                let mut path = path_cell.path.borrow_mut();
                if path.len() >= 3 {
                    path[1] = new_up_link;
                    path[2] = new_down_link;
                }
            }

            // Increment new link counts
            for link_id in &new_links {
                self.increment_link(*link_id);
            }

            // Update binding
            let binding = self.flow_bindings.get_mut(&flow_id).unwrap();
            binding.spine_idx = new_spine;
            binding.links = new_links;
        }
    }

    /// Get current statistics for debugging/logging.
    pub fn get_stats(&self) -> SGLBStats {
        SGLBStats {
            active_flows: self.flow_bindings.len(),
            links_with_flows: self.link_flow_count.len(),
            max_link_count: self.link_flow_count.values().copied().max().unwrap_or(0),
        }
    }
}

/// Statistics from the SGLB router.
#[derive(Debug, Clone)]
pub struct SGLBStats {
    pub active_flows: usize,
    pub links_with_flows: usize,
    pub max_link_count: usize,
}

impl Default for SGLBRouter {
    fn default() -> Self {
        Self::new(SGLBConfig::default())
    }
}

impl Clone for SGLBRouter {
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

impl SpineTreeRouter for SGLBRouter {
    fn set_context(&mut self, context: &MLContext) {
        self.context = Some(context.clone());
    }

    fn route(&mut self, topo: &impl SpineTreeTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        // Ensure link cache is built
        self.ensure_link_cache(topo);

        let src_leaf = topo.get_host_leaf(src);
        let dst_leaf = topo.get_host_leaf(dst);
        let src_leaf_idx = topo.leaf_index_from_node(src_leaf);
        let dst_leaf_idx = topo.leaf_index_from_node(dst_leaf);

        // Intra-leaf: no spine needed
        if src_leaf == dst_leaf {
            let path = self.build_intra_leaf_path(topo, src, dst);
            let path_cell = PathCell { path: Rc::new(RefCell::new(path)) };
            
            // Track binding (no inter-leaf links to count)
            self.flow_bindings.insert(flow_id, FlowBinding {
                spine_idx: 0, // unused for intra-leaf
                src_leaf: src_leaf_idx,
                dst_leaf: dst_leaf_idx,
                path_cell: path_cell.clone(),
                links: vec![],
            });
            
            return path_cell;
        }

        // Inter-leaf: compute eligible spines and select one
        let num_spines = topo.num_spines();
        let eligible = self.compute_eligible_spines(src_leaf_idx, dst_leaf_idx, num_spines);

        // Get job_id for hashing
        let job_id = self.context.as_ref()
            .and_then(|ctx| {
                ctx.waiting_flows.borrow().get(&flow_id).map(|t| t.0)
            })
            .unwrap_or(0);

        let spine_idx = self.hash_select_spine(src, dst, job_id, &eligible);

        // Build path and track links
        let (path, tracked_links) = self.build_path(topo, src, dst, spine_idx);
        let path_cell = PathCell { path: Rc::new(RefCell::new(path)) };

        // Increment link counts
        for link_id in &tracked_links {
            self.increment_link(*link_id);
        }

        // Track binding
        self.flow_bindings.insert(flow_id, FlowBinding {
            spine_idx,
            src_leaf: src_leaf_idx,
            dst_leaf: dst_leaf_idx,
            path_cell: path_cell.clone(),
            links: tracked_links,
        });

        path_cell
    }

    fn complete_flow(&mut self, flow_id: FlowId) {
        if let Some(binding) = self.flow_bindings.remove(&flow_id) {
            // Decrement link counts
            for link_id in binding.links {
                self.decrement_link(link_id);
            }
        }
    }
}

// =============================================================================
// SGLBSystemModule
// =============================================================================

/// SGLB System Module: coordinates SGLB routing across job lifecycle.
///
/// The system module:
/// - Holds configuration for the router
/// - Schedules periodic timers for flow remapping
/// - Triggers flow remapping when timers fire
/// - Provides logging and statistics
#[derive(Debug)]
pub struct SGLBSystemModule {
    config: SGLBConfig,
    /// Whether to log stats on reconfigure
    log_stats: bool,
    /// Whether the periodic timer has been started
    timer_started: bool,
}

impl SGLBSystemModule {
    /// Create a new SGLB system module with the given configuration.
    pub fn new(config: SGLBConfig) -> Self {
        Self { 
            config, 
            log_stats: true,
            timer_started: false,
        }
    }

    /// Create with a specific K value.
    pub fn with_k(k: usize) -> Self {
        Self::new(SGLBConfig::with_k(k))
    }

    /// Disable stats logging.
    pub fn without_logging(mut self) -> Self {
        self.log_stats = false;
        self
    }

    /// Get the configuration (for creating matching router).
    pub fn config(&self) -> &SGLBConfig {
        &self.config
    }

    /// Schedule the next periodic remap timer.
    fn schedule_remap_timer(&self, ctx: &MLContext) {
        if self.config.remap_interval_us > 0 {
            ctx.schedule_timer(self.config.remap_interval_us, SGLB_REMAP_TIMER_ID);
        }
    }
}

impl Default for SGLBSystemModule {
    fn default() -> Self {
        Self::new(SGLBConfig::default())
    }
}

impl<S, FS> SystemModule<SpineTree<SGLBRouter>, S, FS> for SGLBSystemModule
where
    S: JobScheduler,
    FS: FlowScheduler,
{
    fn on_init(
        &mut self,
        ctx: &MLContext,
        _topo: &SpineTree<SGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        println!("[SGLB] Initialized with K={}, w1={}, w2={}, remapping={}, remap_interval={}us",
            self.config.k, self.config.w1, self.config.w2, 
            self.config.enable_remapping, self.config.remap_interval_us);
        
        // Start the periodic remap timer
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
        _job: &crate::simulator::ml_job::MLJob,
        topo: &SpineTree<SGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        if self.log_stats {
            let stats = topo.router.borrow().get_stats();
            println!("[SGLB] t={} Job {} scheduled. Active flows: {}, max link count: {}",
                now_us, job_id, stats.active_flows, stats.max_link_count);
        }
    }

    fn on_job_completed(
        &mut self,
        now_us: u64,
        _ctx: &MLContext,
        job_id: JobId,
        _job: &crate::simulator::ml_job::MLJob,
        topo: &SpineTree<SGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        if self.log_stats {
            let stats = topo.router.borrow().get_stats();
            println!("[SGLB] t={} Job {} completed. Active flows: {}, max link count: {}",
                now_us, job_id, stats.active_flows, stats.max_link_count);
        }
    }

    fn on_reconfigure(
        &mut self,
        now_us: u64,
        _ctx: &MLContext,
        topo: &SpineTree<SGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) -> Option<crate::simulator::system::MigrationPlan> {
        // Only do immediate remapping on reconfigure if periodic timer is disabled
        // Otherwise, the periodic timer handles remapping
        if self.config.enable_remapping && self.config.remap_interval_us == 0 {
            topo.router.borrow_mut().remap_ineligible_flows(topo);
        }

        if self.log_stats {
            let stats = topo.router.borrow().get_stats();
            println!("[SGLB] t={} Reconfigure. Active flows: {}, max link count: {}",
                now_us, stats.active_flows, stats.max_link_count);
        }

        None // SGLB doesn't trigger migrations
    }

    fn on_timer(
        &mut self,
        _now_us: u64,
        ctx: &MLContext,
        timer_id: TimerId,
        topo: &SpineTree<SGLBRouter>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        if timer_id != SGLB_REMAP_TIMER_ID {
            return;
        }

        // Perform periodic flow remapping
        if self.config.enable_remapping {
            topo.router.borrow_mut().remap_ineligible_flows(topo);
        }

        // Reschedule the timer for the next interval
        self.schedule_remap_timer(ctx);
    }
}
