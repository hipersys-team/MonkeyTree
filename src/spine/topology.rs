use petgraph::graph::{DiGraph, NodeIndex};
use std::cell::RefCell;

use crate::network::flow::FlowId;
use crate::network::routing::{PathCell};
use crate::network::topology::{Link, Topology};
use crate::simulator::ml_simulator::MLContext;

/// Trait for two-layer spine-leaf topologies.
///
/// Provides convenient accessors that routing algorithms can use to
/// understand the physical structure of the topology without needing to know
/// implementation details.
pub trait SpineTreeTopology: Topology {
    /// Number of hosts connected to each leaf (ToR) switch.
    fn hosts_per_leaf(&self) -> usize;
    /// Total number of hosts.
    fn num_hosts(&self) -> usize;
    /// Total number of leaf switches.
    fn num_leaves(&self) -> usize;
    /// Total number of spine switches.
    fn num_spines(&self) -> usize;

    // ---------------- Convenience helpers ----------------
    fn get_host(&self, leaf: usize, host: usize) -> Option<NodeIndex>;
    fn get_leaf(&self, leaf: usize) -> Option<NodeIndex>;
    fn get_spine(&self, spine: usize) -> NodeIndex;
    fn get_host_leaf(&self, host: NodeIndex) -> NodeIndex;

    // ---------------- NodeIndex -> usize conversions ----------------
    /// Returns the raw petgraph index for a node.
    #[inline]
    fn node_raw_index(&self, node: NodeIndex) -> usize {
        node.index()
    }

    /// Convert a host NodeIndex to a global host index [0, num_hosts).
    /// Panics if the node is not in the host range.
    #[inline]
    fn host_index_from_node(&self, host: NodeIndex) -> usize {
        let idx = host.index();
        let nh = self.num_hosts();
        assert!(idx < nh, "host_index_from_node: NodeIndex {} is not a host (num_hosts={})", idx, nh);
        idx
    }

    /// Convert a leaf NodeIndex to a leaf index [0, num_leaves).
    /// Panics if the node is not in the leaf range.
    #[inline]
    fn leaf_index_from_node(&self, leaf: NodeIndex) -> usize {
        let idx = leaf.index();
        let nh = self.num_hosts();
        let nl = self.num_leaves();
        assert!(idx >= nh && idx < nh + nl, "leaf_index_from_node: NodeIndex {} is not a leaf (host_range=[0, {}), leaf_range=[{}, {}))", idx, nh, nh, nh + nl);
        idx - nh
    }

    /// Convert a spine NodeIndex to a spine index [0, num_spines).
    /// Panics if the node is not in the spine range.
    #[inline]
    fn spine_index_from_node(&self, spine: NodeIndex) -> usize {
        let idx = spine.index();
        let nh = self.num_hosts();
        let nl = self.num_leaves();
        let ns = self.num_spines();
        let spine_start = nh + nl;
        assert!(idx >= spine_start && idx < spine_start + ns, "spine_index_from_node: NodeIndex {} is not a spine (spine_range=[{}, {}))", idx, spine_start, spine_start + ns);
        idx - spine_start
    }

    /// Convert a host NodeIndex to (leaf_index, host_offset_within_leaf).
    /// Panics if the node is not a host.
    #[inline]
    fn leaf_and_host_from_node(&self, host: NodeIndex) -> (usize, usize) {
        let hidx = self.host_index_from_node(host);
        let hpl = self.hosts_per_leaf();
        (hidx / hpl, hidx % hpl)
    }
}

/// Routing trait specifically for Spine-Leaf topologies.
/// 
/// This trait mirrors `FatTreeRouter` but for two-layer fabrics.
pub trait SpineTreeRouter {
    fn route(&mut self, topo: &impl SpineTreeTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell;
    fn set_context(&mut self, context: &MLContext);
    fn complete_flow(&mut self, flow_id: FlowId);
}

/// Two-layer spine-leaf (a.k.a. leaf-spine) topology implementation.
///
/// All leaves are fully-meshed with all spines. Every link has identical
/// bandwidth capacity.
#[derive(Debug, Clone)]
pub struct SpineTree<R: SpineTreeRouter> {
    /// Underlying directed graph.
    pub graph: DiGraph<(), Link>,
    pub hosts_per_leaf: usize,
    pub num_leaves: usize,
    pub num_spines: usize,
    /// Hosts indexed as [leaf][host].
    pub hosts: Vec<Vec<NodeIndex>>,
    pub leaf_switches: Vec<NodeIndex>,
    pub spine_switches: Vec<NodeIndex>,
    pub router: RefCell<R>,
    /// Homogeneous link bandwidth (bps)
    pub link_bandwidth_bps: f64,
}

impl<R: SpineTreeRouter> SpineTree<R> {
    /// Construct a new spine-leaf fabric.
    pub fn new(
        hosts_per_leaf: usize,
        num_leaves: usize,
        num_spines: usize,
        bandwidth: f64,
        router: R,
    ) -> Self {
        let mut graph = DiGraph::new();

        // 1. Hosts ---------------------------------------------------------
        let mut hosts: Vec<Vec<NodeIndex>> = Vec::with_capacity(num_leaves);
        for _ in 0..num_leaves {
            let mut leaf_hosts = Vec::with_capacity(hosts_per_leaf);
            for _ in 0..hosts_per_leaf {
                leaf_hosts.push(graph.add_node(()));
            }
            hosts.push(leaf_hosts);
        }

        // 2. Leaf switches -------------------------------------------------
        let mut leaf_switches = Vec::with_capacity(num_leaves);
        for _ in 0..num_leaves {
            leaf_switches.push(graph.add_node(()));
        }

        // 3. Spine switches ------------------------------------------------
        let mut spine_switches = Vec::with_capacity(num_spines);
        for _ in 0..num_spines {
            spine_switches.push(graph.add_node(()));
        }

        // 4. Links ---------------------------------------------------------
        let mut link_counter: usize = 0;

        // Hosts ↔ leaf switches.
        for leaf_idx in 0..num_leaves {
            let leaf_switch = leaf_switches[leaf_idx];
            for &host in &hosts[leaf_idx] {
                // Host → Leaf
                graph.add_edge(host, leaf_switch, Link { id: link_counter, bandwidth });
                link_counter += 1;
                // Leaf → Host
                graph.add_edge(leaf_switch, host, Link { id: link_counter, bandwidth });
                link_counter += 1;
            }
        }

        // Full mesh leaves ↔ spines.
        for &leaf_switch in &leaf_switches {
            for &spine_switch in &spine_switches {
                // Leaf → Spine
                graph.add_edge(leaf_switch, spine_switch, Link { id: link_counter, bandwidth });
                link_counter += 1;
                // Spine → Leaf
                graph.add_edge(spine_switch, leaf_switch, Link { id: link_counter, bandwidth });
                link_counter += 1;
            }
        }

        Self {
            graph,
            hosts_per_leaf,
            num_leaves,
            num_spines,
            hosts,
            leaf_switches,
            spine_switches,
            router: RefCell::new(router),
            link_bandwidth_bps: bandwidth,
        }
    }
}

// -------------------------------------------------------------------------
// Trait Implementations
// -------------------------------------------------------------------------
impl<R: SpineTreeRouter> Topology for SpineTree<R> {
    fn set_context(&self, context: &MLContext) {
        self.router.borrow_mut().set_context(context);
    }

    fn complete_flow(&self, flow_id: FlowId) {
        self.router.borrow_mut().complete_flow(flow_id);
    }

    fn total_hosts(&self) -> usize {
        self.num_leaves * self.hosts_per_leaf
    }

    fn get_host_by_index(&self, index: usize) -> Option<NodeIndex> {
        if index < self.total_hosts() {
            Some(NodeIndex::new(index))
        } else {
            None
        }
    }

    fn route(&self, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        self.router.borrow_mut().route(self, src, dst, flow_id)
    }

    fn topology(&self) -> &DiGraph<(), Link> {
        &self.graph
    }

    fn link_bandwidth_bps(&self) -> f64 { self.link_bandwidth_bps }
}

impl<R: SpineTreeRouter> SpineTreeTopology for SpineTree<R> {
    fn hosts_per_leaf(&self) -> usize {
        self.hosts_per_leaf
    }

    fn num_hosts(&self) -> usize {
        self.num_leaves * self.hosts_per_leaf
    }

    fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    fn num_spines(&self) -> usize {
        self.num_spines
    }

    fn get_host(&self, leaf: usize, host: usize) -> Option<NodeIndex> {
        self.hosts.get(leaf)?.get(host).copied()
    }

    fn get_leaf(&self, leaf: usize) -> Option<NodeIndex> {
        self.leaf_switches.get(leaf).copied()
    }

    fn get_spine(&self, spine: usize) -> NodeIndex {
        self.spine_switches[spine]
    }

    fn get_host_leaf(&self, host: NodeIndex) -> NodeIndex {
        let host_idx = host.index();
        let leaf_idx = host_idx / self.hosts_per_leaf;
        self.leaf_switches[leaf_idx]
    }
}
