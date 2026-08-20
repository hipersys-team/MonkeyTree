use crate::network::alloc::BandwidthAllocator;
use crate::utils::data::DHashSet;
use crate::network::topology::Topology;
use crate::network::flow::FlowId;
use crate::network::flow::FlowDesc;
use crate::network::flow::FlowState;
use indexmap::IndexMap;
use crate::network::routing::PathCell;
use crate::simulator::ml_simulator::MLContext;

pub struct MLTCP;

impl BandwidthAllocator for MLTCP {
    fn set_context(&mut self, _context: &MLContext) {}

    fn allocate(&self, topo: &impl Topology, active_desc: &IndexMap<FlowId, FlowDesc>, active_state: &IndexMap<FlowId, FlowState>) -> Vec<f64> {
        let n_flows = active_desc.len();
        let mut rates = vec![0.0; n_flows];

        struct AllocState {
            flow_idx: usize,
            remaining_bytes_ratio: f64,
            path_cell: PathCell,
        }

        let mut alloc_states = Vec::new();

        for (i, ((_fid, desc), (_fid2, state))) in active_desc.iter().zip(active_state.iter()).enumerate() {
            let remaining_bytes_ratio = state.remaining_bytes as f64 / desc.size_bytes as f64;
            let alloc_state = AllocState {
                flow_idx: i,
                remaining_bytes_ratio,
                path_cell: state.path_cell.clone(),
            };
            alloc_states.push(alloc_state);
        }

        alloc_states.sort_by(|a, b| b.remaining_bytes_ratio.partial_cmp(&a.remaining_bytes_ratio).unwrap());

        let mut acquired_links = DHashSet::default();
        for alloc_state in alloc_states {
            let mut free = true;
            for &lid in alloc_state.path_cell.path.borrow().iter() {
                if acquired_links.contains(&lid) {
                    free = false;
                    break;
                }
            }
            if free {
                for &lid in alloc_state.path_cell.path.borrow().iter() {
                    acquired_links.insert(lid);
                }
                rates[alloc_state.flow_idx] = topo.link_bandwidth_bps();
            }
        }

        rates
    }
}