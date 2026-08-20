use petgraph::graph::{DiGraph, NodeIndex};
use std::cell::RefCell;

use crate::network::flow::FlowId;
use crate::network::routing::PathCell;
use crate::network::topology::{Link, Topology};
use crate::simulator::ml_simulator::MLContext;

/// Trait for rail-optimized topologies.
///
/// A rail-optimized topology has three layers:
///   1. Hosts (blocks of GPUs with all-to-all intra-block connectivity)
///   2. Rail switches (one per GPU-offset per pod; rail i connects to GPU i in every block)
///   3. Spine switches (full mesh with all rail switches)
pub trait RailTopology: Topology {
    /// Number of GPUs per host (block).
    fn block_size(&self) -> usize;
    /// Number of hosts (blocks) per pod.
    fn blocks_per_pod(&self) -> usize;
    /// Total number of pods.
    fn num_pods(&self) -> usize;
    /// Number of rail switches per pod (== block_size).
    fn num_rails_per_pod(&self) -> usize;
    /// Total number of spine switches.
    fn num_spines(&self) -> usize;
    /// Total number of hosts (GPUs).
    fn total_gpus(&self) -> usize;

    fn get_host(&self, pod: usize, block: usize, gpu: usize) -> Option<NodeIndex>;
    fn get_rail(&self, pod: usize, rail: usize) -> Option<NodeIndex>;
    fn get_spine(&self, spine: usize) -> NodeIndex;

    /// Pod index for a host NodeIndex.
    fn host_pod(&self, host: NodeIndex) -> usize;
    /// Block (server) index within its pod for a host NodeIndex.
    fn host_block_in_pod(&self, host: NodeIndex) -> usize;
    /// GPU offset within its block for a host NodeIndex (0..block_size).
    fn host_gpu_offset(&self, host: NodeIndex) -> usize;
    /// Global block index (across all pods).
    fn host_block_global(&self, host: NodeIndex) -> usize;

    /// Convert a host NodeIndex to a flat host index [0, total_gpus).
    #[inline]
    fn host_index_from_node(&self, host: NodeIndex) -> usize {
        let idx = host.index();
        assert!(idx < self.total_gpus(),
            "host_index_from_node: NodeIndex {} is not a host (total_gpus={})", idx, self.total_gpus());
        idx
    }

    /// Convert a rail NodeIndex to a (pod, rail_offset) pair.
    fn rail_coords_from_node(&self, rail: NodeIndex) -> (usize, usize) {
        let idx = rail.index();
        let ng = self.total_gpus();
        let total_rails = self.num_pods() * self.num_rails_per_pod();
        assert!(idx >= ng && idx < ng + total_rails,
            "rail_coords_from_node: NodeIndex {} is not a rail", idx);
        let rail_flat = idx - ng;
        (rail_flat / self.num_rails_per_pod(), rail_flat % self.num_rails_per_pod())
    }

    /// Global rail index for a rail NodeIndex.
    fn rail_global_index(&self, rail: NodeIndex) -> usize {
        let (pod, offset) = self.rail_coords_from_node(rail);
        pod * self.num_rails_per_pod() + offset
    }

    /// GPUs per pod (= block_size * blocks_per_pod).
    fn gpus_per_pod(&self) -> usize {
        self.block_size() * self.blocks_per_pod()
    }
}

/// Routing trait for rail-optimized topologies.
pub trait RailTreeRouter {
    fn route(&mut self, topo: &impl RailTopology, src: NodeIndex, dst: NodeIndex, flow_id: FlowId) -> PathCell;
    fn set_context(&mut self, context: &MLContext);
    fn complete_flow(&mut self, flow_id: FlowId);
}

/// Rail-optimized topology implementation.
///
/// Node order in the graph:
///   [0 .. total_gpus)                     = host (GPU) nodes
///   [total_gpus .. total_gpus + total_rails) = rail switch nodes
///   [total_gpus + total_rails .. )         = spine switch nodes
///
/// Within hosts, ordering is: pod-major, then block, then gpu offset.
///   host_flat_index = pod * (blocks_per_pod * block_size) + block * block_size + gpu
#[derive(Debug, Clone)]
pub struct RailTree<R: RailTreeRouter> {
    pub graph: DiGraph<(), Link>,
    pub block_size: usize,
    pub blocks_per_pod: usize,
    pub num_pods: usize,
    pub num_spines: usize,
    /// hosts[pod][block][gpu]
    pub hosts: Vec<Vec<Vec<NodeIndex>>>,
    /// rails[pod][rail_offset]
    pub rail_switches: Vec<Vec<NodeIndex>>,
    pub spine_switches: Vec<NodeIndex>,
    pub router: RefCell<R>,
    pub host_bandwidth_bps: f64,
    pub rail_bandwidth_bps: f64,
    pub spine_bandwidth_bps: f64,
}

impl<R: RailTreeRouter> RailTree<R> {
    pub fn new(
        block_size: usize,
        blocks_per_pod: usize,
        num_pods: usize,
        num_spines: usize,
        host_bandwidth: f64,
        rail_bandwidth: f64,
        spine_bandwidth: f64,
        router: R,
    ) -> Self {
        let mut graph = DiGraph::new();

        // 1. Host (GPU) nodes
        let mut hosts: Vec<Vec<Vec<NodeIndex>>> = Vec::with_capacity(num_pods);
        for _ in 0..num_pods {
            let mut pod_blocks = Vec::with_capacity(blocks_per_pod);
            for _ in 0..blocks_per_pod {
                let mut block_gpus = Vec::with_capacity(block_size);
                for _ in 0..block_size {
                    block_gpus.push(graph.add_node(()));
                }
                pod_blocks.push(block_gpus);
            }
            hosts.push(pod_blocks);
        }

        // 2. Rail switches (block_size per pod)
        let mut rail_switches: Vec<Vec<NodeIndex>> = Vec::with_capacity(num_pods);
        for _ in 0..num_pods {
            let mut pod_rails = Vec::with_capacity(block_size);
            for _ in 0..block_size {
                pod_rails.push(graph.add_node(()));
            }
            rail_switches.push(pod_rails);
        }

        // 3. Spine switches
        let mut spine_switches = Vec::with_capacity(num_spines);
        for _ in 0..num_spines {
            spine_switches.push(graph.add_node(()));
        }

        // 4. Links
        let mut link_counter: usize = 0;

        // 4a. Intra-host all-to-all links within each block
        for pod in 0..num_pods {
            for block in 0..blocks_per_pod {
                for i in 0..block_size {
                    for j in 0..block_size {
                        if i != j {
                            let src = hosts[pod][block][i];
                            let dst = hosts[pod][block][j];
                            graph.add_edge(src, dst, Link { id: link_counter, bandwidth: host_bandwidth });
                            link_counter += 1;
                        }
                    }
                }
            }
        }

        // 4b. Host-to-rail: GPU i in each block connects to rail i in its pod
        for pod in 0..num_pods {
            for block in 0..blocks_per_pod {
                for gpu in 0..block_size {
                    let host_node = hosts[pod][block][gpu];
                    let rail_node = rail_switches[pod][gpu];
                    // Host -> Rail
                    graph.add_edge(host_node, rail_node, Link { id: link_counter, bandwidth: rail_bandwidth });
                    link_counter += 1;
                    // Rail -> Host
                    graph.add_edge(rail_node, host_node, Link { id: link_counter, bandwidth: rail_bandwidth });
                    link_counter += 1;
                }
            }
        }

        // 4c. Rail-to-spine full mesh
        for pod in 0..num_pods {
            for rail in 0..block_size {
                let rail_node = rail_switches[pod][rail];
                for &spine_node in &spine_switches {
                    // Rail -> Spine
                    graph.add_edge(rail_node, spine_node, Link { id: link_counter, bandwidth: spine_bandwidth });
                    link_counter += 1;
                    // Spine -> Rail
                    graph.add_edge(spine_node, rail_node, Link { id: link_counter, bandwidth: spine_bandwidth });
                    link_counter += 1;
                }
            }
        }

        Self {
            graph,
            block_size,
            blocks_per_pod,
            num_pods,
            num_spines,
            hosts,
            rail_switches,
            spine_switches,
            router: RefCell::new(router),
            host_bandwidth_bps: host_bandwidth,
            rail_bandwidth_bps: rail_bandwidth,
            spine_bandwidth_bps: spine_bandwidth,
        }
    }
}

impl<R: RailTreeRouter> Topology for RailTree<R> {
    fn set_context(&self, context: &MLContext) {
        self.router.borrow_mut().set_context(context);
    }

    fn complete_flow(&self, flow_id: FlowId) {
        self.router.borrow_mut().complete_flow(flow_id);
    }

    fn total_hosts(&self) -> usize {
        self.num_pods * self.blocks_per_pod * self.block_size
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

    fn link_bandwidth_bps(&self) -> f64 {
        self.rail_bandwidth_bps
    }
}

impl<R: RailTreeRouter> RailTopology for RailTree<R> {
    fn block_size(&self) -> usize { self.block_size }
    fn blocks_per_pod(&self) -> usize { self.blocks_per_pod }
    fn num_pods(&self) -> usize { self.num_pods }
    fn num_rails_per_pod(&self) -> usize { self.block_size }
    fn num_spines(&self) -> usize { self.num_spines }

    fn total_gpus(&self) -> usize {
        self.num_pods * self.blocks_per_pod * self.block_size
    }

    fn get_host(&self, pod: usize, block: usize, gpu: usize) -> Option<NodeIndex> {
        self.hosts.get(pod)?.get(block)?.get(gpu).copied()
    }

    fn get_rail(&self, pod: usize, rail: usize) -> Option<NodeIndex> {
        self.rail_switches.get(pod)?.get(rail).copied()
    }

    fn get_spine(&self, spine: usize) -> NodeIndex {
        self.spine_switches[spine]
    }

    fn host_pod(&self, host: NodeIndex) -> usize {
        let idx = host.index();
        let gpus_per_pod = self.blocks_per_pod * self.block_size;
        idx / gpus_per_pod
    }

    fn host_block_in_pod(&self, host: NodeIndex) -> usize {
        let idx = host.index();
        let gpus_per_pod = self.blocks_per_pod * self.block_size;
        (idx % gpus_per_pod) / self.block_size
    }

    fn host_gpu_offset(&self, host: NodeIndex) -> usize {
        host.index() % self.block_size
    }

    fn host_block_global(&self, host: NodeIndex) -> usize {
        let idx = host.index();
        idx / self.block_size
    }
}
