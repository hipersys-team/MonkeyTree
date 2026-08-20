pub mod perfect;
pub mod crux;
pub mod monkeytree;
pub mod sglb;

use petgraph::graph::NodeIndex;
use crate::network::routing::Path;
use crate::network::topology::Topology;

pub(crate) fn convert_node_path_to_links(topo: &impl Topology, nodes: &[NodeIndex]) -> Path {
    let mut link_path = Vec::with_capacity(nodes.len().saturating_sub(1));
    let graph = topo.topology();
    for window in nodes.windows(2) {
        let edge = graph
            .find_edge(window[0], window[1])
            .unwrap_or_else(|| panic!("No edge between {:?} and {:?}", window[0], window[1]));
        link_path.push(graph.edge_weight(edge).unwrap().id);
    }
    link_path
}

pub use perfect::{FatTreePerfectRouter, FatTreePerfectSystem};
pub use crux::{FatTreeCruxRouter, FatTreeCruxSystem};
pub use monkeytree::{FatTreeMonkeyTreePerfect, FatTreeMonkeyTree3};
pub use sglb::{FatTreeSGLBRouter, FatTreeSGLBSystem};
