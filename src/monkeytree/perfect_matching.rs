//! Bipartite Edge Coloring via Perfect-Matching Decomposition
//!
//! Uses Hopcroft-Karp algorithm to find perfect matchings and decomposes
//! a bipartite graph into edge-disjoint matchings (colors).
//!
//! For our use case:
//! - Left vertices = source ToRs
//! - Right vertices = destination ToRs  
//! - Edges = flows between ToRs
//! - Colors = spine switches to use

use std::collections::HashMap;

/// An edge in the bipartite graph
#[derive(Debug, Clone)]
pub struct BipartiteEdge {
    pub id: usize,
    pub left: usize,   // source ToR
    pub right: usize,  // destination ToR
    pub is_dummy: bool,
}

/// Result of edge coloring
#[derive(Debug, Clone)]
pub struct EdgeColoring {
    /// Maps edge_id -> color (0-indexed)
    pub edge_to_color: HashMap<usize, usize>,
    /// Number of colors used
    pub num_colors: usize,
}

/// Compute edge coloring for a bipartite graph.
/// 
/// Returns a coloring where edges incident to the same vertex have different colors.
/// Uses at most Δ colors where Δ is the maximum degree.
pub fn compute_edge_coloring(
    num_vertices: usize,  // number of ToRs (same for left and right)
    edges: Vec<(usize, usize, usize)>,  // (left, right, edge_id)
) -> EdgeColoring {
    if edges.is_empty() {
        return EdgeColoring {
            edge_to_color: HashMap::new(),
            num_colors: 0,
        };
    }

    let n = num_vertices;
    
    // Step 1: Compute degrees and Δ
    let mut deg_l = vec![0usize; n];
    let mut deg_r = vec![0usize; n];
    
    let mut all_edges: Vec<BipartiteEdge> = Vec::new();
    
    for (left, right, id) in edges {
        deg_l[left] += 1;
        deg_r[right] += 1;
        all_edges.push(BipartiteEdge {
            id,
            left,
            right,
            is_dummy: false,
        });
    }
    
    let delta = deg_l.iter().chain(deg_r.iter()).copied().max().unwrap_or(0);
    
    if delta == 0 {
        return EdgeColoring {
            edge_to_color: HashMap::new(),
            num_colors: 0,
        };
    }
    
    // Step 2: Make the graph Δ-regular by adding dummy edges
    let mut left_slots: Vec<usize> = Vec::new();
    let mut right_slots: Vec<usize> = Vec::new();
    
    for u in 0..n {
        let deficit = delta - deg_l[u];
        for _ in 0..deficit {
            left_slots.push(u);
        }
    }
    
    for v in 0..n {
        let deficit = delta - deg_r[v];
        for _ in 0..deficit {
            right_slots.push(v);
        }
    }
    
    // Add dummy edges
    let mut dummy_id = usize::MAX / 2; // Start from a high number to avoid conflicts
    for i in 0..left_slots.len() {
        all_edges.push(BipartiteEdge {
            id: dummy_id,
            left: left_slots[i],
            right: right_slots[i],
            is_dummy: true,
        });
        dummy_id += 1;
    }
    
    // Step 3: Repeatedly extract perfect matchings and color them
    let mut edge_to_color: HashMap<usize, usize> = HashMap::new();
    let mut active: Vec<bool> = vec![true; all_edges.len()];
    
    for color in 0..delta {
        // Build adjacency list for active edges
        let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n]; // adj[left] = [(right, edge_idx), ...]
        
        for (edge_idx, edge) in all_edges.iter().enumerate() {
            if active[edge_idx] {
                adj[edge.left].push((edge.right, edge_idx));
            }
        }
        
        // Find perfect matching using Hopcroft-Karp
        let matching = hopcroft_karp(n, &adj);
        
        // Assign color to real edges in matching, deactivate all matched edges
        for edge_idx in matching {
            active[edge_idx] = false;
            let edge = &all_edges[edge_idx];
            if !edge.is_dummy {
                edge_to_color.insert(edge.id, color);
            }
        }
    }
    
    EdgeColoring {
        edge_to_color,
        num_colors: delta,
    }
}

/// Hopcroft-Karp algorithm for maximum bipartite matching.
/// Returns indices of edges in the matching.
fn hopcroft_karp(n: usize, adj: &[Vec<(usize, usize)>]) -> Vec<usize> {
    const NIL: usize = usize::MAX;
    
    let mut pair_u: Vec<usize> = vec![NIL; n]; // pair_u[left] = right vertex (or NIL)
    let mut pair_v: Vec<usize> = vec![NIL; n]; // pair_v[right] = left vertex (or NIL)
    let mut matched_edge: Vec<usize> = vec![NIL; n]; // matched_edge[left] = edge_idx
    let mut dist: Vec<usize> = vec![0; n + 1];
    
    fn bfs(
        n: usize,
        adj: &[Vec<(usize, usize)>],
        pair_u: &[usize],
        pair_v: &[usize],
        dist: &mut [usize],
    ) -> bool {
        const NIL: usize = usize::MAX;
        const INF: usize = usize::MAX;
        
        let mut queue = std::collections::VecDeque::new();
        
        for u in 0..n {
            if pair_u[u] == NIL {
                dist[u] = 0;
                queue.push_back(u);
            } else {
                dist[u] = INF;
            }
        }
        
        dist[n] = INF; // dist[NIL] represented as dist[n]
        
        while let Some(u) = queue.pop_front() {
            if dist[u] < dist[n] {
                for &(v, _edge_idx) in &adj[u] {
                    let partner = pair_v[v];
                    let partner_idx = if partner == NIL { n } else { partner };
                    if dist[partner_idx] == INF {
                        dist[partner_idx] = dist[u] + 1;
                        if partner != NIL {
                            queue.push_back(partner);
                        }
                    }
                }
            }
        }
        
        dist[n] != INF
    }
    
    fn dfs(
        u: usize,
        n: usize,
        adj: &[Vec<(usize, usize)>],
        pair_u: &mut [usize],
        pair_v: &mut [usize],
        matched_edge: &mut [usize],
        dist: &mut [usize],
    ) -> bool {
        const NIL: usize = usize::MAX;
        const INF: usize = usize::MAX;
        
        if u == n {
            // Represents NIL
            return true;
        }
        
        for &(v, edge_idx) in &adj[u] {
            let partner = pair_v[v];
            let partner_idx = if partner == NIL { n } else { partner };
            
            if dist[partner_idx] == dist[u] + 1 {
                if dfs(partner_idx, n, adj, pair_u, pair_v, matched_edge, dist) {
                    pair_u[u] = v;
                    pair_v[v] = u;
                    matched_edge[u] = edge_idx;
                    return true;
                }
            }
        }
        
        dist[u] = INF;
        false
    }
    
    // Main Hopcroft-Karp loop
    while bfs(n, adj, &pair_u, &pair_v, &mut dist) {
        for u in 0..n {
            if pair_u[u] == NIL {
                dfs(u, n, adj, &mut pair_u, &mut pair_v, &mut matched_edge, &mut dist);
            }
        }
    }
    
    // Collect matched edges
    let mut matching = Vec::new();
    for u in 0..n {
        if matched_edge[u] != NIL {
            matching.push(matched_edge[u]);
        }
    }
    
    matching
}

/// Given an edge coloring with `num_colors` colors, collapse them into `target_colors`.
/// If num_colors <= target_colors, returns the original coloring.
/// Otherwise, maps colors 0..num_colors into 0..target_colors, trying to minimize overlap.
pub fn collapse_colors(
    coloring: &EdgeColoring,
    target_colors: usize,
) -> HashMap<usize, usize> {
    if coloring.num_colors <= target_colors {
        return coloring.edge_to_color.clone();
    }
    
    // Simple strategy: color % target_colors
    // This distributes load somewhat evenly
    coloring.edge_to_color
        .iter()
        .map(|(&edge_id, &color)| (edge_id, color % target_colors))
        .collect()
}