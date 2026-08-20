use petgraph::graph::{NodeIndex, DiGraph};
use crate::network::flow::FlowId;
use crate::network::routing::{PathCell, SingleLinkRouter, FatTreeRouter};
use std::cell::RefCell;
use crate::simulator::ml_simulator::MLContext;

/// Unique identifier for a link (edge) in the topology graph.
pub type LinkId = usize;

/// Basic network link with a fixed bandwidth capacity in **bps**.
#[derive(Debug, Clone)]
pub struct Link {
    pub id: LinkId,
    pub bandwidth: f64,
}

/// Common interface for all network topology types.
/// 
/// This trait provides a unified way to interact with different topology implementations
/// while still allowing each type to have its own specialized methods.
pub trait Topology {
    /// Returns the total number of hosts in this topology.
    fn total_hosts(&self) -> usize;
    
    /// Gets a host by linear index.
    /// 
    /// # Arguments
    /// * `index` - Linear index of the host (0-based)
    /// 
    /// # Returns
    /// The `NodeIndex` of the host, or `None` if the index is out of bounds.
    fn get_host_by_index(&self, index: usize) -> Option<NodeIndex>;

    fn route(&self, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell;

    fn topology(&self) -> &petgraph::graph::DiGraph<(), Link>;

    fn set_context(&self, context: &MLContext);

    fn complete_flow(&self, flow_id: FlowId);

    /// Returns the homogeneous link bandwidth (bps) for this topology.
    /// All current topologies are homogeneous, so this value is constant.
    fn link_bandwidth_bps(&self) -> f64;
}

// CONTRACT: Topology implementations must assign host indices such that
// sorting by host_index gives a left-to-right ordering through the topology.
// For spine-leaf topologies, this means: host_index = leaf * hosts_per_leaf + offset
// This enables rank reassignment to order workers optimally for ring collectives.


pub trait SingleLinkTopology: Topology {
    fn source(&self) -> NodeIndex;
    fn destination(&self) -> NodeIndex;
    fn link_id(&self) -> LinkId;
    fn bandwidth(&self) -> f64;
}


/// A simple topology with exactly two hosts connected by a single link.
/// 
/// This is a convenience wrapper around `Topology` that creates a basic
/// point-to-point network for testing and simple simulations. The topology
/// is parameterized by a router type that implements the basic Router trait.
#[derive(Debug, Clone)]
pub struct SingleLink<R: SingleLinkRouter> {
    /// Index of the source host
    pub source: NodeIndex,
    /// Index of the destination host
    pub destination: NodeIndex,
    /// ID of the connecting link
    pub link_id: LinkId,
    /// Bandwidth of the link in bps
    pub bandwidth: f64,
    /// Router for computing paths
    pub router: RefCell<R>,
    /// Graph of the topology
    pub graph: DiGraph<(), Link>,
}

impl<R: SingleLinkRouter> SingleLink<R> {
    pub fn new(bandwidth: f64, router: R) -> Self {
        let mut graph = DiGraph::new();
        let source: NodeIndex = graph.add_node(());
        let destination: NodeIndex = graph.add_node(());
        // Create bidirectional link: source -> destination and destination -> source
        let link_id_fwd: LinkId = graph.add_edge(source, destination, Link { id: 0, bandwidth }).index();
        let _link_id_rev: LinkId = graph.add_edge(destination, source, Link { id: 1, bandwidth }).index();
        Self {
            source,
            destination,
            link_id: link_id_fwd, // Keep the forward direction as the primary link_id for compatibility
            bandwidth,
            router: RefCell::new(router),
            graph,
        }
    }
}

impl<R: SingleLinkRouter> Topology for SingleLink<R> {

    fn set_context(&self, context: &MLContext) {
        self.router.borrow_mut().set_context(context);
    }

    fn complete_flow(&self, flow_id: FlowId) {
        self.router.borrow_mut().complete_flow(flow_id);
    }

    fn total_hosts(&self) -> usize {
        2
    }

    fn get_host_by_index(&self, index: usize) -> Option<NodeIndex> {
        match index {
            0 => Some(self.source),
            1 => Some(self.destination),
            _ => None,
        }
    }

    // TODO: why does this function exist?
    fn route(&self, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        self.router.borrow_mut().route(self, src, dst, flow_id)
    }

    fn topology(&self) -> &petgraph::graph::DiGraph<(), Link> {
        &self.graph
    }

    fn link_bandwidth_bps(&self) -> f64 { self.bandwidth }
}

impl<R: SingleLinkRouter> SingleLinkTopology for SingleLink<R> {
    /// Returns the capacity of the single link.
    /// 
    /// # Returns
    /// The bandwidth capacity in bits per second.
    fn bandwidth(&self) -> f64 {
        self.bandwidth
    }
    
    /// Returns the source host index.
    fn source(&self) -> NodeIndex {
        self.source
    }
    
    /// Returns the destination host index.
    fn destination(&self) -> NodeIndex {
        self.destination
    }
    
    /// Returns the link ID.
    fn link_id(&self) -> LinkId {
        self.link_id
    }
}

/// Trait specifically for fat tree topologies to provide access to fat tree structure.
/// 
/// This trait allows fat tree routers to access pod/ToR/aggregation switch information
/// for implementing advanced routing algorithms.
pub trait FatTreeTopology: Topology {
    /// Returns the number of hosts per ToR switch.
    fn hosts_per_tor(&self) -> usize;
    
    /// Returns the number of ToR switches per pod.
    fn degree_tor(&self) -> usize;
    
    /// Returns the number of aggregation switches per pod.
    fn degree_agg(&self) -> usize;
    
    /// Returns the number of core switches.
    fn degree_core(&self) -> usize;
    
    /// Returns the number of pods.
    fn num_pods(&self) -> usize;

    fn total_tor_switches(&self) -> usize;

    fn total_agg_switches(&self) -> usize;

    fn total_core_switches(&self) -> usize;
    
    /// Gets a host by coordinates (pod, tor, host).
    fn get_host(&self, pod: usize, tor: usize, host: usize) -> Option<NodeIndex>;
    
    /// Gets a ToR switch by coordinates (pod, tor).
    fn get_tor(&self, pod: usize, tor: usize) -> Option<NodeIndex>;
    
    /// Gets an aggregation switch by coordinates (pod, agg).
    fn get_agg(&self, pod: usize, agg: usize) -> NodeIndex;
    
    /// Gets a core switch by index.
    fn get_core(&self, core: usize) -> NodeIndex;

    fn get_host_pod(&self, host: NodeIndex) -> usize;

    fn get_host_tor(&self, host: NodeIndex) -> NodeIndex;

    /// Returns the global core indices connected to a given agg switch.
    fn cores_connected_to_agg(&self, _agg_idx: usize) -> Vec<usize> {
        (0..self.total_core_switches()).collect()
    }

    /// Returns the agg switch indices (within a pod) connected to a given core.
    fn aggs_connected_to_core(&self, _core_idx: usize) -> Vec<usize> {
        (0..self.degree_agg()).collect()
    }
}

/// A Fat Tree topology implementation for data center networks.
/// 
/// Fat Tree is a hierarchical topology commonly used in data centers that provides
/// high bisection bandwidth and multiple paths between any pair of hosts. The topology
/// consists of three layers: core, aggregation (agg), and top-of-rack (ToR) switches,
/// with hosts connected to ToR switches.
/// 
/// The topology is organized into pods, where each pod contains:
/// - ToR switches with hosts connected to them
/// - Aggregation switches that connect ToR switches within the pod
/// - Core switches that connect different pods
/// 
/// The topology is parameterized by a router type that implements the FatTreeRouter trait.
#[derive(Debug, Clone)]
pub struct FatTree<R: FatTreeRouter> {
    // The underlying topology graph
    pub graph: DiGraph<(), Link>,
    /// Number of hosts connected to each ToR switch
    pub hosts_per_tor: usize,
    /// Number of ToR switches per pod
    pub degree_tor: usize,
    /// Number of aggregation switches per pod
    pub degree_agg: usize,
    /// Number of core switches
    pub degree_core: usize,
    /// Number of pods (calculated as degree_tor)
    pub num_pods: usize,
    /// Node indices for hosts (organized as [pod][tor][host])
    pub hosts: Vec<Vec<Vec<NodeIndex>>>,
    /// Node indices for ToR switches (organized as [pod][tor])
    pub tor_switches: Vec<Vec<NodeIndex>>,
    /// Node indices for aggregation switches (organized as [pod][agg])
    pub agg_switches: Vec<Vec<NodeIndex>>,
    /// Node indices for core switches
    pub core_switches: Vec<NodeIndex>,
    /// Router for computing paths
    pub router: RefCell<R>,
    /// Homogeneous link bandwidth (bps)
    pub link_bandwidth_bps: f64,
}

impl<R: FatTreeRouter> FatTree<R> {
    /// Creates a new Fat Tree topology with the specified parameters and router.
    /// 
    /// # Arguments
    /// * `hosts_per_tor` - Number of hosts connected to each ToR switch
    /// * `degree_tor` - Number of ToR switches per pod
    /// * `degree_agg` - Number of aggregation switches per pod
    /// * `degree_core` - Number of core switches
    /// * `num_pods` - Number of pods in the fat tree topology
    /// * `link_capacity_bps` - Bandwidth capacity for all links in bits per second
    /// * `router` - The routing algorithm to use for path computation
    /// 
    /// # Returns
    /// A new `FatTreeTopology` with the specified structure and uniform link capacities.
    /// 
    /// # Fat Tree Structure
    /// - **Pods**: The topology contains `num_pods` pods, each with `degree_tor` ToR switches and `degree_agg` aggregation switches
    /// - **Hosts**: Each ToR switch connects to `hosts_per_tor` hosts
    /// - **ToR-Agg Links**: Each ToR switch connects to all aggregation switches in its pod
    /// - **Agg-Core Links**: Each aggregation switch connects to `degree_core/degree_agg` core switches
    /// - **Total hosts**: `num_pods * degree_tor * hosts_per_tor`
    /// 
    /// # Example
    /// ```
    /// use network_sim::network::topology::FatTreeTopology;
    /// 
    /// // Create a fat tree with 2 pods: 4 hosts per ToR, 2 ToRs per pod, 2 agg per pod, 2 cores
    /// // This creates 2 pods with 16 total hosts
    /// ```
    pub fn new(
        hosts_per_tor: usize,
        degree_tor: usize,
        degree_agg: usize,
        degree_core: usize,
        num_pods: usize,
        bandwidth: f64,
        router: R,
    ) -> Self {
        let mut graph = DiGraph::new();

        // Initialize storage for node indices
        let mut hosts = Vec::with_capacity(num_pods);
        let mut tor_switches = Vec::with_capacity(num_pods);
        let mut agg_switches = Vec::with_capacity(num_pods);
        let mut core_switches = Vec::with_capacity(degree_core);

        // =========================================
        // PHASE 1: CREATE ALL NODES IN ORDER
        // Order: hosts → ToR switches → agg switches → core switches
        // =========================================

        // Create all hosts first (NodeIndex 0, 1, 2, ...)
        for _pod in 0..num_pods {
            let mut pod_hosts = Vec::with_capacity(degree_tor);
            for _tor in 0..degree_tor {
                let mut tor_hosts = Vec::with_capacity(hosts_per_tor);
                for _ in 0..hosts_per_tor {
                    let host = graph.add_node(());
                    tor_hosts.push(host);
                }
                pod_hosts.push(tor_hosts);
            }
            hosts.push(pod_hosts);
        }

        // Create all ToR switches next
        for _pod in 0..num_pods {
            let mut pod_tors = Vec::with_capacity(degree_tor);
            for _ in 0..degree_tor {
                let tor_switch = graph.add_node(());
                pod_tors.push(tor_switch);
            }
            tor_switches.push(pod_tors);
        }

        // Create all aggregation switches next
        for _pod in 0..num_pods {
            let mut pod_aggs = Vec::with_capacity(degree_agg);
            for _ in 0..degree_agg {
                let agg_switch = graph.add_node(());
                pod_aggs.push(agg_switch);
            }
            agg_switches.push(pod_aggs);
        }

        // Create all core switches last
        for _ in 0..degree_core {
            core_switches.push(graph.add_node(()));
        }

        // =========================================
        // PHASE 2: CREATE ALL EDGES
        // =========================================

        let mut link_counter: usize = 0;

        // Connect hosts to their ToR switches (bidirectional)
        for pod in 0..num_pods {
            for tor in 0..degree_tor {
                let tor_switch = tor_switches[pod][tor];
                for &host in &hosts[pod][tor] {
                    // Host to ToR
                    graph.add_edge(host, tor_switch, Link { id: link_counter, bandwidth });
                    link_counter += 1;
                    // ToR to Host
                    graph.add_edge(tor_switch, host, Link { id: link_counter, bandwidth });
                    link_counter += 1;
                }
            }
        }

        // Connect ToR switches to aggregation switches within each pod (bidirectional)
        for pod in 0..num_pods {
            for &tor_switch in &tor_switches[pod] {
                for &agg_switch in &agg_switches[pod] {
                    // ToR to Agg
                    graph.add_edge(tor_switch, agg_switch, Link { id: link_counter, bandwidth });
                    link_counter += 1;
                    // Agg to ToR
                    graph.add_edge(agg_switch, tor_switch, Link { id: link_counter, bandwidth });
                    link_counter += 1;
                }
            }
        }

        // Connect aggregation switches to core switches (fully connected, bidirectional)
        for pod in 0..num_pods {
            for &agg_switch in &agg_switches[pod] {
                for &core_switch in &core_switches {
                    graph.add_edge(agg_switch, core_switch, Link { id: link_counter, bandwidth });
                    link_counter += 1;
                    graph.add_edge(core_switch, agg_switch, Link { id: link_counter, bandwidth });
                    link_counter += 1;
                }
            }
        }

        Self {
            graph,
            hosts_per_tor,
            degree_tor,
            degree_agg,
            degree_core,
            num_pods,
            hosts,
            tor_switches,
            agg_switches,
            core_switches,
            router: RefCell::new(router),
            link_bandwidth_bps: bandwidth,
        }
    }
}

impl<R: FatTreeRouter> Topology for FatTree<R> {
    fn set_context(&self, context: &MLContext) {
        self.router.borrow_mut().set_context(context);
    }

    fn complete_flow(&self, flow_id: FlowId) {
        self.router.borrow_mut().complete_flow(flow_id);
    }

    /// Returns the total number of hosts in the fat tree
    fn total_hosts(&self) -> usize {
        self.num_pods * self.degree_tor * self.hosts_per_tor
    }

    fn get_host_by_index(&self, index: usize) -> Option<NodeIndex> {
        // With new ordering: hosts come first, so linear index maps directly to NodeIndex
        if index < self.total_hosts() {
            Some(NodeIndex::new(index))
        } else {
            None
        }
    }

    fn route(&self, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell {
        self.router.borrow_mut().route(self, src, dst, flow_id)
    }

    fn topology(&self) -> &petgraph::graph::DiGraph<(), Link> {
        &self.graph
    }

    fn link_bandwidth_bps(&self) -> f64 { self.link_bandwidth_bps }
}

impl<R: FatTreeRouter> FatTreeTopology for FatTree<R> {
    /// Returns the node index of a specific host by coordinates.
    /// 
    /// # Arguments
    /// * `pod` - Pod index (0-based)
    /// * `tor` - ToR switch index within the pod (0-based)
    /// * `host` - Host index within the ToR (0-based)
    /// 
    /// # Returns
    /// The `NodeIndex` of the specified host, or `None` if indices are out of bounds.
    fn get_host(&self, pod: usize, tor: usize, host: usize) -> Option<NodeIndex> {
        self.hosts.get(pod)?.get(tor)?.get(host).copied()
    }

    /// Returns the node index of a specific ToR switch.
    /// 
    /// # Arguments
    /// * `pod` - Pod index (0-based)
    /// * `tor` - ToR switch index within the pod (0-based)
    fn get_tor(&self, pod: usize, tor: usize) -> Option<NodeIndex> {
        self.tor_switches.get(pod)?.get(tor).copied()
    }

    /// Returns the node index of a specific aggregation switch.
    /// 
    /// # Arguments
    /// * `pod` - Pod index (0-based)
    /// * `agg` - Aggregation switch index within the pod (0-based)
    fn get_agg(&self, pod: usize, agg: usize) -> NodeIndex {
        let agg = self.agg_switches.get(pod).unwrap().get(agg);
        agg.unwrap().clone()
    }

    /// Returns the node index of a specific core switch.
    /// 
    /// # Arguments
    /// * `core` - Core switch index (0-based)
    fn get_core(&self, core: usize) -> NodeIndex {
        self.core_switches.get(core).unwrap().clone()
    }

    fn hosts_per_tor(&self) -> usize {
        self.hosts_per_tor
    }

    fn degree_tor(&self) -> usize {
        self.degree_tor
    }

    fn degree_agg(&self) -> usize {
        self.degree_agg
    }

    fn degree_core(&self) -> usize {
        self.degree_core
    }

    fn num_pods(&self) -> usize {
        self.num_pods
    }



    /// Returns the total number of ToR switches in the fat tree.
    fn total_tor_switches(&self) -> usize {
        self.num_pods * self.degree_tor
    }

    /// Returns the total number of aggregation switches in the fat tree.
    fn total_agg_switches(&self) -> usize {
        self.num_pods * self.degree_agg
    }

    /// Returns the total number of core switches in the fat tree.
    fn total_core_switches(&self) -> usize {
        self.degree_core
    }

    fn get_host_pod(&self, host: NodeIndex) -> usize {
        // With new ordering: hosts come first, so calculation is straightforward
        let host_idx = host.index();
        let pod = host_idx / (self.degree_tor * self.hosts_per_tor);
        pod 
    }

    fn get_host_tor(&self, host: NodeIndex) -> NodeIndex {
        // With new ordering: calculate which ToR this host belongs to
        let host_idx = host.index();
        let hosts_per_pod = self.degree_tor * self.hosts_per_tor;
        let pod = host_idx / hosts_per_pod;
        let host_within_pod = host_idx % hosts_per_pod;
        let tor_within_pod = host_within_pod / self.hosts_per_tor;
        
        // Get the actual ToR NodeIndex from our data structure
        self.tor_switches[pod][tor_within_pod]
    }
}