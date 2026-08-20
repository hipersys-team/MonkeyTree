//! Deterministic **max‑min fair‑share** allocator.
use crate::network::flow::{FlowDesc, FlowId, FlowState};
use crate::network::topology::Topology;
use indexmap::IndexMap;
use std::cell::RefCell;
use crate::simulator::ml_simulator::MLContext;

pub trait BandwidthAllocator {
    fn set_context(&mut self, context: &MLContext);
    /// Allocate bandwidth to flows. Returns rates in the same order as active_desc iteration.
    fn allocate(&self, topo: &impl Topology, active_desc: &IndexMap<FlowId, FlowDesc>, active_state: &IndexMap<FlowId, FlowState>) -> Vec<f64>;
}

/// Computes max-min fair bandwidth allocation for active flows.
/// 
/// This function implements the max-min fairness algorithm, which ensures that
/// no flow can increase its rate without decreasing the rate of a flow that
/// already has a lower or equal rate. The algorithm works by iteratively
/// finding bottleneck links and allocating the maximum possible fair share
/// to flows constrained by those bottlenecks.
/// 
/// # Arguments
/// * `topo` - The network topology containing link capacities
/// * `active_desc` - Map of currently active flows with their descriptions
/// * `active_state` - Map of currently active flows with their states (containing paths)
/// 
/// # Returns
/// A HashMap mapping each FlowId to its allocated transmission rate in bits per second.
/// 
/// # Algorithm
/// 1. Start with all flows unallocated and full link capacities available
/// 2. For each iteration:
///    - Count how many flows use each link
///    - Calculate the fair share each flow could get on each of its links
///    - Find the minimum fair share (bottleneck constraint)
///    - Allocate this rate to all flows with this bottleneck constraint
///    - Remove these flows from consideration and subtract their usage from link capacities
/// 3. Repeat until all flows are allocated

/// Scratch buffers for MaxMin allocator to avoid per-call allocations.
#[derive(Default)]
struct MaxMinScratch {
    /// Number of links in topology (for sizing dense arrays)
    num_links: usize,
    /// Cached base link capacities from topology, indexed by link_id (dense)
    base_cap: Vec<f64>,
    /// Working capacity per link, indexed by link_id (dense)
    cap: Vec<f64>,
    /// Flow count per link, indexed by link_id (dense)
    cnt: Vec<usize>,
    /// Output rates in input order (reused each call)
    rates: Vec<f64>,
    /// Flow indices sorted by initial bottleneck share
    sorted_flows: Vec<usize>,
    /// Initial shares for sorting (flow_idx -> initial_share)
    initial_shares: Vec<f64>,
}

pub struct MaxMin {
    scratch: RefCell<MaxMinScratch>,
}

impl MaxMin {
    pub fn new() -> Self {
        Self {
            scratch: RefCell::new(MaxMinScratch::default()),
        }
    }
}

impl Default for MaxMin {
    fn default() -> Self {
        Self::new()
    }
}

/// Find minimum link share for a flow's path.
/// Computes min(cap[lid] / cnt[lid]) for all links in the path.
#[inline(always)]
fn find_min_share(path: &[usize], cap: &[f64], cnt: &[usize]) -> f64 {
    let mut min_share = f64::INFINITY;
    for &lid in path {
        let share = cap[lid] / cnt[lid] as f64;
        if share < min_share {
            min_share = share;
        }
    }
    min_share
}

impl BandwidthAllocator for MaxMin {
    fn set_context(&mut self, _context: &MLContext) {}

    /// O(F × L) max-min fair allocation using sorted processing.
    /// 
    /// Algorithm:
    /// 1. Count flows per link: O(F × L)
    /// 2. Compute initial bottleneck share per flow: O(F × L)
    /// 3. Sort flows by initial share: O(F log F)
    /// 4. Process in sorted order, computing actual rate from current state: O(F × L)
    /// 
    /// Key insight: processing flows in order of initial share is valid because
    /// removing a flow can only increase (never decrease) other flows' shares.
    fn allocate(&self, topo: &impl Topology, active_desc: &IndexMap<FlowId, FlowDesc>, active_state: &IndexMap<FlowId, FlowState>) -> Vec<f64> {
        let mut scratch = self.scratch.borrow_mut();
        
        // Destructure to allow independent borrowing of fields
        let MaxMinScratch {
            num_links,
            base_cap,
            cap,
            cnt,
            rates,
            sorted_flows,
            initial_shares,
        } = &mut *scratch;
        
        let n_flows = active_desc.len();
        
        // Early return for empty input
        if n_flows == 0 {
            return Vec::new();
        }
        
        // Rebuild base capacity cache if empty (first call or topology changed)
        if base_cap.is_empty() {
            let graph = topo.topology();
            // Find max link ID to size our dense arrays
            let max_lid = graph.edge_references()
                .map(|e| e.weight().id)
                .max()
                .unwrap_or(0);
            *num_links = max_lid + 1;
            
            // Initialize base_cap as dense array indexed by link_id
            base_cap.resize(*num_links, 0.0);
            for e in graph.edge_references() {
                base_cap[e.weight().id] = e.weight().bandwidth;
            }
            
            // Pre-allocate working arrays
            cap.resize(*num_links, 0.0);
            cnt.resize(*num_links, 0);
        }
        
        // Reset working capacity from cached base capacities (fast memcpy)
        cap.copy_from_slice(base_cap);
        
        // Reset count array to zeros
        cnt.fill(0);
        
        // Prepare output rates array (indexed by flow position in input order)
        rates.clear();
        rates.resize(n_flows, 0.0);
        
        // Step 1: Count flows per link - O(F × L)
        // Access paths directly from active_state to avoid cloning
        for (_fid, state) in active_state.iter() {
            let path = state.path_cell.path.borrow();
            for &lid in path.iter() {
                cnt[lid] += 1;
            }
        }
        
        // Step 2: Compute initial bottleneck share per flow - O(F × L)
        initial_shares.clear();
        initial_shares.resize(n_flows, 0.0);
        for (i, (_fid, state)) in active_state.iter().enumerate() {
            let path = state.path_cell.path.borrow();
            initial_shares[i] = find_min_share(&path, cap, cnt);
        }
        
        // Step 3: Sort flow indices by initial share - O(F log F)
        sorted_flows.clear();
        sorted_flows.extend(0..n_flows);
        sorted_flows.sort_unstable_by(|&a, &b| {
            initial_shares[a].partial_cmp(&initial_shares[b]).unwrap()
        });
        
        // Step 4: Process flows in sorted order - O(F × L)
        // Each flow's rate = current bottleneck share (after lower-share flows removed)
        for &flow_idx in sorted_flows.iter() {
            // Access path by index position in IndexMap
            let (_fid, state) = active_state.get_index(flow_idx).unwrap();
            let path = state.path_cell.path.borrow();
            
            // Compute current share from current capacities/counts
            let rate = find_min_share(&path, cap, cnt);
            rates[flow_idx] = rate;
            
            // Update capacities and counts for this flow
            for &lid in path.iter() {
                cap[lid] -= rate;
                cnt[lid] -= 1;
            }
        }
        
        // Return rates - use std::mem::take to avoid clone, scratch will reallocate next call
        std::mem::take(rates)
    }
}