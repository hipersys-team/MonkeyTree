//! ILP-based migration solver for MonkeyTree
//!
//! Translates the bipartite rebalancing ILP from ilp.md to Rust using good_lp.
//!
//! ## Problem
//! Given a set of fragmented jobs spread across ToRs, find the minimum number
//! of worker moves needed to reduce per-ToR fragmentation below a threshold.
//!
//! ## Block-based Allocation
//! When `block_size > 1`, the ILP operates at block granularity:
//! - Workers are grouped into blocks (e.g., 8 GPUs per server)
//! - Migrations move entire blocks, not individual workers
//! - The ILP sees `workers / block_size` as the unit of allocation

use std::collections::HashMap;

// Debug flag for ILP migration tracking. Enable to trace block moves and allocations.
const DEBUG_ILP: bool = false;
use good_lp::{constraint, coin_cbc, variable, variables, Expression, Solution, SolverModel, Variable};
use crate::simulator::ml_job::JobId;
use crate::simulator::ml_worker::WorkerId;
use crate::simulator::system::JobMigration;
use crate::utils::DHashMap;

/// Status of ILP solve
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveStatus {
    Optimal,
    Infeasible,
    Timeout,
    Error(String),
}

/// Optional configuration enabling a cross-pod fragmentation constraint.
///
/// When attached to an `ILPInput` / `SegmentILPInput`, the solver adds, in
/// addition to the standard per-ToR constraint (5.4 / 6.6), a mirrored
/// constraint at the pod granularity:
///
///     sum_s ring_count[s] * z(s, p) - sum_s ring_count[s] * y_pod(s, p) <= pod_lambda
///
/// for every pod `p`, where
/// - `z(s, p) = 1` iff segment s has at least one worker in pod p (OR of x over tors in p),
/// - `y_pod(s, p) = 1` iff segment s is present in pod p AND pod p is the ONLY
///   pod where s is present (i.e., s is not cross-pod fragmented).
///
/// This leaves the original per-ToR constraint unchanged, so MonkeyTree (legacy)
/// and MonkeyTree3 can share the same solver with `pod_config = None` vs `Some`.
#[derive(Debug, Clone)]
pub struct PodConstraintConfig {
    /// Number of pods.
    pub num_pods: usize,
    /// Number of ToRs per pod. Pod of ToR `t` is `t / tors_per_pod`.
    pub tors_per_pod: usize,
    /// Cross-pod fragmentation threshold (same units as `target_lambda`:
    /// weighted by ring_count).
    pub pod_lambda: usize,
}

/// Input to the ILP solver (legacy, job-based)
#[deprecated(since = "0.2.0", note = "Use SegmentILPInput for pipeline-aware optimization")]
#[derive(Debug, Clone)]
pub struct ILPInput {
    /// Sources = fragmented jobs
    pub fragmented_jobs: Vec<JobId>,
    /// Number of ToRs
    pub num_tors: usize,
    /// Initial allocation: (job_id, tor_index) -> worker count (in blocks if block_size > 1)
    pub initial_allocation: HashMap<(JobId, usize), usize>,
    /// Total workers per job (in blocks if block_size > 1)
    pub workers_per_job: HashMap<JobId, usize>,
    /// Capacity per ToR (in blocks if block_size > 1)
    pub tor_capacity: usize,
    /// Target max fragmented jobs per ToR (λ threshold)
    pub target_lambda: usize,
    /// Non-fragmented workers per ToR (in blocks if block_size > 1)
    pub nonfrag_workers_per_tor: Vec<usize>,
    /// Block size for allocation (1 = individual workers, 8 = 8-GPU servers)
    pub block_size: usize,
    /// Number of independent rings per job.
    /// For strided ring with stride S, this equals S.
    /// For regular AllReduce or AllToAll, this is 1.
    /// Used in constraint 5.4 to weight each job's contribution to fragmentation.
    pub job_ring_count: HashMap<JobId, usize>,
    /// Optional cross-pod fragmentation constraint. When `None`, the solver
    /// behaves identically to the original formulation (preserves MonkeyTree
    /// and MonkeyTreePerfect behavior). When `Some`, a mirrored constraint is
    /// added at the pod granularity (used by MonkeyTree3).
    pub pod_config: Option<PodConstraintConfig>,
}

#[allow(deprecated)]
impl ILPInput {
    /// Create an ILPInput with block_size = 1 (individual workers)
    pub fn new_individual(
        fragmented_jobs: Vec<JobId>,
        num_tors: usize,
        initial_allocation: HashMap<(JobId, usize), usize>,
        workers_per_job: HashMap<JobId, usize>,
        tor_capacity: usize,
        target_lambda: usize,
        nonfrag_workers_per_tor: Vec<usize>,
        job_ring_count: HashMap<JobId, usize>,
    ) -> Self {
        Self {
            fragmented_jobs,
            num_tors,
            initial_allocation,
            workers_per_job,
            tor_capacity,
            target_lambda,
            nonfrag_workers_per_tor,
            block_size: 1,
            job_ring_count,
            pod_config: None,
        }
    }
    
    /// Create an ILPInput with block-based allocation.
    /// All counts are automatically divided by block_size.
    pub fn new_blocked(
        fragmented_jobs: Vec<JobId>,
        num_tors: usize,
        initial_allocation: HashMap<(JobId, usize), usize>,
        workers_per_job: HashMap<JobId, usize>,
        tor_capacity: usize,
        target_lambda: usize,
        nonfrag_workers_per_tor: Vec<usize>,
        block_size: usize,
        job_ring_count: HashMap<JobId, usize>,
    ) -> Self {
        // Scale all counts by block_size
        let scaled_initial: HashMap<(JobId, usize), usize> = initial_allocation
            .into_iter()
            .map(|(k, v)| (k, v / block_size))
            .collect();
        
        let scaled_workers: HashMap<JobId, usize> = workers_per_job
            .into_iter()
            .map(|(k, v)| (k, v / block_size))
            .collect();
        
        let scaled_nonfrag: Vec<usize> = nonfrag_workers_per_tor
            .into_iter()
            .map(|v| v / block_size)
            .collect();
        
        Self {
            fragmented_jobs,
            num_tors,
            initial_allocation: scaled_initial,
            workers_per_job: scaled_workers,
            tor_capacity: tor_capacity / block_size,
            target_lambda,
            nonfrag_workers_per_tor: scaled_nonfrag,
            block_size,
            job_ring_count,
            pod_config: None,
        }
    }
}

/// Output from the ILP solver (legacy, job-based)
#[deprecated(since = "0.2.0", note = "Use SegmentILPSolution for pipeline-aware optimization")]
#[derive(Debug, Clone)]
pub struct ILPSolution {
    pub status: SolveStatus,
    /// New allocation: (job_id, tor_index) -> worker count
    pub new_allocation: HashMap<(JobId, usize), usize>,
    /// Minimum number of worker moves required
    pub num_moves: usize,
}

/// Build and solve the ILP for minimum-move rebalancing (legacy, job-based)
#[deprecated(since = "0.2.0", note = "Use solve_segment_migration_ilp for pipeline-aware optimization")]
#[allow(deprecated)]
pub fn solve_migration_ilp(input: &ILPInput) -> Result<ILPSolution, String> {
    if input.fragmented_jobs.is_empty() {
        return Ok(ILPSolution {
            status: SolveStatus::Optimal,
            new_allocation: HashMap::new(),
            num_moves: 0,
        });
    }

    let jobs = &input.fragmented_jobs;
    let num_tors = input.num_tors;
    
    // Build edge set E: all (job, tor) pairs
    let edges: Vec<(JobId, usize)> = jobs.iter()
        .flat_map(|&j| (0..num_tors).map(move |t| (j, t)))
        .collect();
    
    // Compute R_s = total workers for each job
    let r: HashMap<JobId, usize> = input.workers_per_job.clone();
    
    // Compute Big-M bounds U[(s,t)] = min(R[s], beta[t])
    let beta = input.tor_capacity;
    let u: HashMap<(JobId, usize), usize> = edges.iter()
        .map(|&(s, t)| {
            let rs = r.get(&s).copied().unwrap_or(0);
            ((s, t), rs.min(beta))
        })
        .collect();
    
    // Initial allocation w0
    let w0 = &input.initial_allocation;
    
    // Create the ILP problem
    let mut vars = variables!();
    
    // Decision variables
    // w[(s,t)] = final worker count (integer >= 0)
    let w: HashMap<(JobId, usize), Variable> = edges.iter()
        .map(|&(s, t)| ((s, t), vars.add(variable().integer().min(0))))
        .collect();
    
    // x[(s,t)] = 1 if job s has any workers on ToR t (binary)
    let x: HashMap<(JobId, usize), Variable> = edges.iter()
        .map(|&(s, t)| ((s, t), vars.add(variable().binary())))
        .collect();
    
    // d[(s,t)] = deviation |w - w0| (integer >= 0)
    let d: HashMap<(JobId, usize), Variable> = edges.iter()
        .map(|&(s, t)| ((s, t), vars.add(variable().integer().min(0))))
        .collect();
    
    // y[(s,t)] = 1 if job s is present on ToR t AND t is the ONLY ToR where s is present (binary)
    let y: HashMap<(JobId, usize), Variable> = edges.iter()
        .map(|&(s, t)| ((s, t), vars.add(variable().binary())))
        .collect();

    // Optional cross-pod fragmentation variables (MonkeyTree3).
    // z[(s,p)]     = 1 iff segment s has any worker in pod p (pod-level presence, OR of x over tors in p)
    // y_pod[(s,p)] = 1 iff segment s is present in pod p AND p is the ONLY pod where s is present
    //                (i.e., s is not cross-pod fragmented)
    let (z_pod, y_pod): (HashMap<(JobId, usize), Variable>, HashMap<(JobId, usize), Variable>) =
        if let Some(pod_cfg) = &input.pod_config {
            let num_pods = pod_cfg.num_pods;
            let mut z_map = HashMap::new();
            let mut y_map = HashMap::new();
            for &s in jobs {
                for p in 0..num_pods {
                    z_map.insert((s, p), vars.add(variable().binary()));
                    y_map.insert((s, p), vars.add(variable().binary()));
                }
            }
            (z_map, y_map)
        } else {
            (HashMap::new(), HashMap::new())
        };

    // Objective: minimize total deviation
    let objective: Expression = d.values().map(|&v| v).sum();
    
    let mut problem = vars.minimise(objective).using(coin_cbc);
    // Suppress CBC solver output
    problem.set_parameter("log", "0");
    
    // Constraint 5.1: Source row-sum (preserve worker counts)
    for &s in jobs {
        let row_sum: Expression = (0..num_tors)
            .filter_map(|t| w.get(&(s, t)))
            .map(|&v| v)
            .sum();
        let rs = r.get(&s).copied().unwrap_or(0) as f64;
        problem = problem.with(constraint!(row_sum == rs));
    }
    
    // Constraint 5.2: Destination capacity (max workers per ToR)
    for t in 0..num_tors {
        let col_sum: Expression = jobs.iter()
            .filter_map(|&s| w.get(&(s, t)))
            .map(|&v| v)
            .sum();
        let nonfrag = input.nonfrag_workers_per_tor.get(t).copied().unwrap_or(0) as f64;
        let capacity = (beta as f64) - nonfrag;
        problem = problem.with(constraint!(col_sum <= capacity));
    }
    
    // Constraint 5.3: Activation linking (w <= U * x)
    for &(s, t) in &edges {
        let w_var = w[&(s, t)];
        let x_var = x[&(s, t)];
        let u_val = u.get(&(s, t)).copied().unwrap_or(0) as f64;
        problem = problem.with(constraint!(w_var <= u_val * x_var));
    }
    
    // Constraint 5.4 (updated per 6.6): Destination degree with consolidation
    // sum_s (ring_count[s] * x(s,t)) <= lambda + sum_s (ring_count[s] * y(s,t)) for this specific ToR t
    // Each job's contribution is weighted by its ring_count (number of independent rings it creates).
    // For strided ring jobs with stride S, this is S. For regular AllReduce/AllToAll, this is 1.
    for t in 0..num_tors {
        let x_sum: Expression = jobs.iter()
            .filter_map(|&s| {
                let ring_count = input.job_ring_count.get(&s).copied().unwrap_or(1) as f64;
                x.get(&(s, t)).map(|&v| ring_count * v)
            })
            .sum();
        let y_sum: Expression = jobs.iter()
            .filter_map(|&s| {
                let ring_count = input.job_ring_count.get(&s).copied().unwrap_or(1) as f64;
                y.get(&(s, t)).map(|&v| ring_count * v)
            })
            .sum();
        let lambda = input.target_lambda as f64;
        problem = problem.with(constraint!(x_sum - y_sum <= lambda));
    }
    
    // Constraint 5.5: Deviation linearization
    for &(s, t) in &edges {
        let w_var = w[&(s, t)];
        let d_var = d[&(s, t)];
        let w0_val = w0.get(&(s, t)).copied().unwrap_or(0) as f64;
        // d >= w - w0
        problem = problem.with(constraint!(d_var >= w_var - w0_val));
        // d >= w0 - w
        problem = problem.with(constraint!(d_var >= w0_val - w_var));
    }
    
    // Constraint 6.6a: y(s,t) = 1 requires x(s,t) = 1 (job must be present on that ToR)
    for &(s, t) in &edges {
        let y_var = y[&(s, t)];
        let x_var = x[&(s, t)];
        problem = problem.with(constraint!(y_var <= x_var));
    }
    
    // Constraint 6.6b: y(s,t) = 1 forces job to be on ONLY ToR t (no other ToRs)
    // sum_{t' != t} x(s,t') <= (|T|-1) * (1 - y(s,t))
    let num_tors_minus_1 = (num_tors - 1) as f64;
    for &s in jobs {
        for t in 0..num_tors {
            if let Some(&y_var) = y.get(&(s, t)) {
                // Sum x(s,t') for all t' != t
                let x_sum_others: Expression = (0..num_tors)
                    .filter(|&t_prime| t_prime != t)
                    .filter_map(|t_prime| x.get(&(s, t_prime)))
                    .map(|&v| v)
                    .sum();
                // x_sum_others <= (|T|-1) * (1 - y(s,t))
                // Rearranged: x_sum_others + (|T|-1) * y(s,t) <= |T|-1
                problem = problem.with(constraint!(x_sum_others + num_tors_minus_1 * y_var <= num_tors_minus_1));
            }
        }
    }

    // --------------------------------------------------------------------
    // Optional cross-pod fragmentation constraint (MonkeyTree3).
    // Enabled only when input.pod_config.is_some().
    // Mirrors the per-ToR (5.4 / 6.6) formulation at the pod granularity.
    // --------------------------------------------------------------------
    if let Some(pod_cfg) = &input.pod_config {
        let num_pods = pod_cfg.num_pods;
        let tors_per_pod = pod_cfg.tors_per_pod;
        let pod_lambda = pod_cfg.pod_lambda as f64;

        // (P.1) z(s, p) >= x(s, t) for every ToR t in pod p.
        // This makes z(s, p) act as an OR of x(s, t) over the ToRs in pod p:
        // if any ToR in pod p has segment s, then pod p has segment s.
        for &s in jobs {
            for p in 0..num_pods {
                let tor_start = p * tors_per_pod;
                let tor_end = ((p + 1) * tors_per_pod).min(num_tors);
                let z_var = match z_pod.get(&(s, p)) {
                    Some(&v) => v,
                    None => continue,
                };
                for t in tor_start..tor_end {
                    if let Some(&x_var) = x.get(&(s, t)) {
                        problem = problem.with(constraint!(z_var >= x_var));
                    }
                }
            }
        }

        // (P.2) y_pod(s, p) <= z(s, p): can only claim "not cross-pod fragmented, present in p"
        // if segment s is actually in pod p.
        for &s in jobs {
            for p in 0..num_pods {
                if let (Some(&z_var), Some(&yp_var)) = (z_pod.get(&(s, p)), y_pod.get(&(s, p))) {
                    problem = problem.with(constraint!(yp_var <= z_var));
                }
            }
        }

        // (P.3) y_pod(s, p) = 1 forces segment to be present in ONLY pod p (z=0 for all p' != p).
        //   sum_{p' != p} z(s, p') <= (|P| - 1) * (1 - y_pod(s, p))
        //   <=> sum_{p' != p} z(s, p') + (|P| - 1) * y_pod(s, p) <= |P| - 1
        if num_pods >= 2 {
            let num_pods_minus_1 = (num_pods - 1) as f64;
            for &s in jobs {
                for p in 0..num_pods {
                    let yp_var = match y_pod.get(&(s, p)) {
                        Some(&v) => v,
                        None => continue,
                    };
                    let z_sum_others: Expression = (0..num_pods)
                        .filter(|&pp| pp != p)
                        .filter_map(|pp| z_pod.get(&(s, pp)))
                        .map(|&v| v)
                        .sum();
                    problem = problem.with(constraint!(
                        z_sum_others + num_pods_minus_1 * yp_var <= num_pods_minus_1
                    ));
                }
            }
        }

        // (P.4) Per-pod cross-pod fragmentation budget.
        //   For every pod p:
        //     sum_s ring_count[s] * z(s, p) - sum_s ring_count[s] * y_pod(s, p) <= pod_lambda
        // Segments present in pod p contribute ring_count on the LHS. If a segment
        // is present in only p (y_pod = 1), its contribution is canceled, so only
        // cross-pod fragmented segments actually count toward the pod's budget.
        for p in 0..num_pods {
            let z_sum: Expression = jobs.iter()
                .filter_map(|&s| {
                    let rc = input.job_ring_count.get(&s).copied().unwrap_or(1) as f64;
                    z_pod.get(&(s, p)).map(|&v| rc * v)
                })
                .sum();
            let y_pod_sum: Expression = jobs.iter()
                .filter_map(|&s| {
                    let rc = input.job_ring_count.get(&s).copied().unwrap_or(1) as f64;
                    y_pod.get(&(s, p)).map(|&v| rc * v)
                })
                .sum();
            problem = problem.with(constraint!(z_sum - y_pod_sum <= pod_lambda));
        }
    }

    // Solve
    let solution = match problem.solve() {
        Ok(sol) => sol,
        Err(e) => {
            return Ok(ILPSolution {
                status: SolveStatus::Error(format!("{:?}", e)),
                new_allocation: HashMap::new(),
                num_moves: 0,
            });
        }
    };
    
    // Extract solution
    let mut new_allocation = HashMap::new();
    let mut total_deviation = 0.0;
    
    for &(s, t) in &edges {
        let val = solution.value(w[&(s, t)]).round() as usize;
        if val > 0 {
            new_allocation.insert((s, t), val);
        }
        total_deviation += solution.value(d[&(s, t)]);
    }
    
    let num_moves = (total_deviation / 2.0).round() as usize;
    
    // Log ILP solution details
    println!("[ILP] Solution: objective={:.1} (total deviation), num_moves={}", total_deviation, num_moves);
    println!("[ILP] Proposed allocation changes:");
    for &job_id in jobs {
        let mut job_changes: Vec<String> = Vec::new();
        for t in 0..num_tors {
            let old_val = w0.get(&(job_id, t)).copied().unwrap_or(0);
            let new_val = new_allocation.get(&(job_id, t)).copied().unwrap_or(0);
            if old_val != new_val {
                job_changes.push(format!("ToR{}: {} -> {}", t, old_val, new_val));
            }
        }
        if !job_changes.is_empty() {
            println!("  Job {}: {}", job_id, job_changes.join(", "));
        }
    }
    
    Ok(ILPSolution {
        status: SolveStatus::Optimal,
        new_allocation,
        num_moves,
    })
}

/// Convert ILP solution to concrete worker migrations.
/// 
/// When block_size > 1, migrations happen at block granularity:
/// - Workers are grouped into blocks of `block_size` consecutive hosts
/// - A "block move" means all workers in a block move together to a new block
/// 
/// Algorithm:
/// 1. Collect all moves needed (excess blocks + destinations)
/// 2. For each block move, find a free block on the destination ToR
/// 3. Move all workers in the block to the new block
#[deprecated(since = "0.2.0", note = "Use compute_segment_migrations for pipeline-aware optimization")]
#[allow(deprecated)]
pub fn compute_migrations(
    input: &ILPInput,
    solution: &ILPSolution,
    current_placements: &DHashMap<JobId, DHashMap<WorkerId, usize>>,
    hosts_per_leaf: usize,
) -> Vec<JobMigration> {
    let block_size = input.block_size;
    
    if block_size <= 1 {
        return compute_migrations_individual(input, solution, current_placements, hosts_per_leaf);
    }
    
    compute_migrations_blocked(input, solution, current_placements, hosts_per_leaf)
}

/// Compute migrations for individual worker allocation (block_size = 1)
#[allow(deprecated)]
fn compute_migrations_individual(
    input: &ILPInput,
    solution: &ILPSolution,
    current_placements: &DHashMap<JobId, DHashMap<WorkerId, usize>>,
    hosts_per_leaf: usize,
) -> Vec<JobMigration> {
    let num_hosts = input.num_tors * hosts_per_leaf;
    
    // Track host availability - initialize from current placements
    let mut host_available: Vec<bool> = vec![true; num_hosts];
    for worker_hosts in current_placements.values() {
        for &host in worker_hosts.values() {
            if host < num_hosts {
                host_available[host] = false;
            }
        }
    }
    
    // Pending moves: (job_id, worker_id, src_host, src_tor, dst_tor)
    let mut pending_moves: Vec<(JobId, WorkerId, usize, usize, usize)> = Vec::new();
    
    // Workers needed per (job, tor) - will be decremented as we assign
    let mut needed_per_job_tor: HashMap<(JobId, usize), usize> = HashMap::new();
    
    for &job_id in &input.fragmented_jobs {
        let worker_hosts = match current_placements.get(&job_id) {
            Some(wh) => wh,
            None => continue,
        };
        
        // Current workers per ToR for this job
        let mut current_per_tor: HashMap<usize, Vec<WorkerId>> = HashMap::new();
        for (&wid, &host) in worker_hosts.iter() {
            let tor = host / hosts_per_leaf;
            current_per_tor.entry(tor).or_default().push(wid);
        }
        
        // Collect excess and needs
        let mut excess_workers: Vec<(WorkerId, usize, usize)> = Vec::new();
        
        for tor in 0..input.num_tors {
            let current = current_per_tor.get(&tor).map(|v| v.len()).unwrap_or(0);
            let target = solution.new_allocation.get(&(job_id, tor)).copied().unwrap_or(0);
            
            if current > target {
                let mut workers = current_per_tor.get(&tor).cloned().unwrap_or_default();
                workers.sort_by(|a, b| b.cmp(a));
                for i in 0..(current - target) {
                    if let Some(&wid) = workers.get(i) {
                        let host = worker_hosts.get(&wid).copied().unwrap_or(0);
                        excess_workers.push((wid, host, tor));
                    }
                }
            } else if target > current {
                needed_per_job_tor.insert((job_id, tor), target - current);
            }
        }
        
        for (wid, host, src_tor) in excess_workers {
            let mut assigned_dst = None;
            for tor in 0..input.num_tors {
                let key = (job_id, tor);
                if let Some(count) = needed_per_job_tor.get_mut(&key) {
                    if *count > 0 {
                        *count -= 1;
                        assigned_dst = Some(tor);
                        break;
                    }
                }
            }
            
            if let Some(dst_tor) = assigned_dst {
                pending_moves.push((job_id, wid, host, src_tor, dst_tor));
                if host < num_hosts {
                    host_available[host] = true;
                }
            } else {
                println!("  [warning] job={} worker={}: excess but no destination", job_id, wid);
            }
        }
    }
    
    println!("[compute_migrations] {} pending moves to process", pending_moves.len());
    
    let mut migrations_dict: HashMap<JobId, DHashMap<WorkerId, usize>> = HashMap::new();
    
    for (job_id, wid, src_host, src_tor, dst_tor) in pending_moves {
        let tor_start = dst_tor * hosts_per_leaf;
        let tor_end = (dst_tor + 1) * hosts_per_leaf;
        
        let mut found_host = None;
        for host in tor_start..tor_end {
            if host < num_hosts && host_available[host] {
                found_host = Some(host);
                break;
            }
        }
        
        if let Some(new_host) = found_host {
            println!("  [move] job={} worker={}: host {} (ToR {}) -> host {} (ToR {})",
                job_id, wid, src_host, src_tor, new_host, dst_tor);
            
            migrations_dict.entry(job_id).or_default().insert(wid, new_host);
            host_available[new_host] = false;
        } else {
            println!("  [error] job={} worker={}: no free host on ToR {}!",
                job_id, wid, dst_tor);
        }
    }
    
    let mut migrations = Vec::new();
    for (job_id, worker_to_host) in migrations_dict {
        if !worker_to_host.is_empty() {
            migrations.push(JobMigration { job_id, worker_to_host });
        }
    }
    
    migrations
}

/// Compute migrations for block-based allocation (block_size > 1)
/// 
/// Workers are grouped into blocks: workers [8k, 8k+7] belong to block k.
/// When a block moves, all workers in it move to a new block on the destination ToR.
#[allow(deprecated)]
fn compute_migrations_blocked(
    input: &ILPInput,
    solution: &ILPSolution,
    current_placements: &DHashMap<JobId, DHashMap<WorkerId, usize>>,
    hosts_per_leaf: usize,
) -> Vec<JobMigration> {
    let block_size = input.block_size;
    let num_hosts = input.num_tors * hosts_per_leaf;
    let blocks_per_tor = hosts_per_leaf / block_size;
    
    // Track block availability (a block is free if all hosts in it are free)
    let mut block_available: Vec<bool> = vec![true; num_hosts / block_size];
    for worker_hosts in current_placements.values() {
        for &host in worker_hosts.values() {
            let block_idx = host / block_size;
            if block_idx < block_available.len() {
                block_available[block_idx] = false;
            }
        }
    }
    
    // Pending block moves: (job_id, src_block, src_tor, dst_tor, workers_in_block)
    let mut pending_block_moves: Vec<(JobId, usize, usize, usize, Vec<(WorkerId, usize)>)> = Vec::new();
    
    // Blocks needed per (job, tor)
    let mut needed_per_job_tor: HashMap<(JobId, usize), usize> = HashMap::new();
    
    for &job_id in &input.fragmented_jobs {
        let worker_hosts = match current_placements.get(&job_id) {
            Some(wh) => wh,
            None => continue,
        };
        
        // Group workers by block
        let mut blocks: HashMap<usize, Vec<(WorkerId, usize)>> = HashMap::new();
        for (&wid, &host) in worker_hosts.iter() {
            let block_idx = host / block_size;
            blocks.entry(block_idx).or_default().push((wid, host));
        }
        
        // Current blocks per ToR
        let mut blocks_per_tor_current: HashMap<usize, Vec<usize>> = HashMap::new();
        for &block_idx in blocks.keys() {
            let tor = (block_idx * block_size) / hosts_per_leaf;
            blocks_per_tor_current.entry(tor).or_default().push(block_idx);
        }
        
        // Collect excess blocks and needs (solution counts are already in blocks)
        let mut excess_blocks: Vec<(usize, usize)> = Vec::new(); // (block_idx, tor)
        
        for tor in 0..input.num_tors {
            let current_blocks = blocks_per_tor_current.get(&tor).map(|v| v.len()).unwrap_or(0);
            let target_blocks = solution.new_allocation.get(&(job_id, tor)).copied().unwrap_or(0);
            
            if current_blocks > target_blocks {
                let mut tor_blocks = blocks_per_tor_current.get(&tor).cloned().unwrap_or_default();
                tor_blocks.sort_by(|a, b| b.cmp(a)); // reverse order for determinism
                for i in 0..(current_blocks - target_blocks) {
                    if let Some(&block_idx) = tor_blocks.get(i) {
                        excess_blocks.push((block_idx, tor));
                    }
                }
            } else if target_blocks > current_blocks {
                needed_per_job_tor.insert((job_id, tor), target_blocks - current_blocks);
            }
        }
        
        // Match excess blocks to destination ToRs
        for (block_idx, src_tor) in excess_blocks {
            let mut assigned_dst = None;
            for tor in 0..input.num_tors {
                let key = (job_id, tor);
                if let Some(count) = needed_per_job_tor.get_mut(&key) {
                    if *count > 0 {
                        *count -= 1;
                        assigned_dst = Some(tor);
                        break;
                    }
                }
            }
            
            if let Some(dst_tor) = assigned_dst {
                let workers_in_block = blocks.get(&block_idx).cloned().unwrap_or_default();
                pending_block_moves.push((job_id, block_idx, src_tor, dst_tor, workers_in_block));
                // Mark source block as free
                if block_idx < block_available.len() {
                    block_available[block_idx] = true;
                }
            } else {
                println!("  [warning] job={} block={}: excess but no destination", job_id, block_idx);
            }
        }
    }
    
    println!("[compute_migrations] {} pending block moves to process", pending_block_moves.len());
    
    let mut migrations_dict: HashMap<JobId, DHashMap<WorkerId, usize>> = HashMap::new();
    
    for (job_id, src_block, src_tor, dst_tor, workers_in_block) in pending_block_moves {
        // Find a free block on dst_tor
        let block_start = dst_tor * blocks_per_tor;
        let block_end = block_start + blocks_per_tor;
        
        let mut found_block = None;
        for block_idx in block_start..block_end {
            if block_idx < block_available.len() && block_available[block_idx] {
                found_block = Some(block_idx);
                break;
            }
        }
        
        if let Some(new_block) = found_block {
            let new_block_host_start = new_block * block_size;
            
            println!("  [block move] job={}: block {} (ToR {}) -> block {} (ToR {}), {} workers",
                job_id, src_block, src_tor, new_block, dst_tor, workers_in_block.len());
            
            // Map each worker to the corresponding host in the new block
            // Workers at offset i in old block go to offset i in new block
            for (wid, old_host) in workers_in_block {
                let offset = old_host % block_size;
                let new_host = new_block_host_start + offset;
                migrations_dict.entry(job_id).or_default().insert(wid, new_host);
            }
            
            block_available[new_block] = false;
        } else {
            println!("  [error] job={} block={}: no free block on ToR {}!",
                job_id, src_block, dst_tor);
        }
    }
    
    let mut migrations = Vec::new();
    for (job_id, worker_to_host) in migrations_dict {
        if !worker_to_host.is_empty() {
            migrations.push(JobMigration { job_id, worker_to_host });
        }
    }
    
    migrations
}

// ============================================================================
// Segment-based ILP support (for pipeline-aware optimization)
// ============================================================================

use super::fragmentation::{SegmentId, JobSegment};

/// Input to the segment-based ILP solver.
/// Each segment (pipeline stage or whole job) is treated as an independent unit.
#[derive(Debug, Clone)]
pub struct SegmentILPInput {
    /// Fragmented segments
    pub fragmented_segments: Vec<SegmentId>,
    /// All segment info (for worker counts and ring counts)
    pub segments: HashMap<SegmentId, JobSegment>,
    /// Number of ToRs
    pub num_tors: usize,
    /// Initial allocation: (segment_id, tor_index) -> worker count
    pub initial_allocation: HashMap<(SegmentId, usize), usize>,
    /// Capacity per ToR
    pub tor_capacity: usize,
    /// Target max fragmented segments per ToR (λ threshold)
    pub target_lambda: usize,
    /// Non-fragmented workers per ToR
    pub nonfrag_workers_per_tor: Vec<usize>,
    /// Block size for allocation (1 = individual workers)
    pub block_size: usize,
    /// Optional cross-pod fragmentation constraint. `None` preserves the
    /// original per-ToR-only behavior (MonkeyTree / MonkeyTreePerfect).
    /// `Some(..)` enables the additional cross-pod constraint used by MonkeyTree3.
    pub pod_config: Option<PodConstraintConfig>,
}

/// Output from the segment-based ILP solver.
#[derive(Debug, Clone)]
pub struct SegmentILPSolution {
    pub status: SolveStatus,
    /// New allocation: (segment_id, tor_index) -> worker count
    pub new_allocation: HashMap<(SegmentId, usize), usize>,
    /// Minimum number of worker moves required
    pub num_moves: usize,
}

/// Solve the segment-based migration ILP.
/// Converts segment input to job-based input, solves, then converts back.
#[allow(deprecated)]  // Internally uses the legacy ILPInput/solve_migration_ilp
pub fn solve_segment_migration_ilp(input: &SegmentILPInput) -> Result<SegmentILPSolution, String> {
    if input.fragmented_segments.is_empty() {
        return Ok(SegmentILPSolution {
            status: SolveStatus::Optimal,
            new_allocation: HashMap::new(),
            num_moves: 0,
        });
    }
    
    // Create a mapping from SegmentId to virtual JobId
    // Use a simple encoding: index in the segments list
    let segment_to_virtual: HashMap<SegmentId, JobId> = input.fragmented_segments.iter()
        .enumerate()
        .map(|(idx, &seg)| (seg, idx))
        .collect();
    
    let virtual_to_segment: HashMap<JobId, SegmentId> = segment_to_virtual.iter()
        .map(|(&seg, &vid)| (vid, seg))
        .collect();
    
    // Convert to job-based input
    let fragmented_jobs: Vec<JobId> = input.fragmented_segments.iter()
        .map(|seg| segment_to_virtual[seg])
        .collect();
    
    let initial_allocation: HashMap<(JobId, usize), usize> = input.initial_allocation.iter()
        .filter_map(|((seg, tor), &count)| {
            segment_to_virtual.get(seg).map(|&vid| ((vid, *tor), count))
        })
        .collect();
    
    let workers_per_job: HashMap<JobId, usize> = input.fragmented_segments.iter()
        .filter_map(|seg| {
            input.segments.get(seg).map(|s| (segment_to_virtual[seg], s.num_workers()))
        })
        .collect();
    
    let job_ring_count: HashMap<JobId, usize> = input.fragmented_segments.iter()
        .filter_map(|seg| {
            input.segments.get(seg).map(|s| (segment_to_virtual[seg], s.ring_count))
        })
        .collect();
    
    println!("[ILP] Segment to job conversion: {} segments -> {} virtual jobs", 
        input.fragmented_segments.len(), fragmented_jobs.len());
    println!("[ILP] Workers per virtual job: {:?}", workers_per_job);
    println!("[ILP] Initial allocation: {:?}", initial_allocation);
    
    // Apply block scaling if needed
    let mut job_input = if input.block_size > 1 {
        ILPInput::new_blocked(
            fragmented_jobs,
            input.num_tors,
            initial_allocation,
            workers_per_job,
            input.tor_capacity,
            input.target_lambda,
            input.nonfrag_workers_per_tor.clone(),
            input.block_size,
            job_ring_count,
        )
    } else {
        ILPInput::new_individual(
            fragmented_jobs,
            input.num_tors,
            initial_allocation,
            workers_per_job,
            input.tor_capacity,
            input.target_lambda,
            input.nonfrag_workers_per_tor.clone(),
            job_ring_count,
        )
    };

    // Forward the optional cross-pod constraint config (None preserves legacy behavior).
    job_input.pod_config = input.pod_config.clone();

    // Solve using the existing job-based solver
    let job_solution = solve_migration_ilp(&job_input)?;
    
    println!("[ILP] Job solution has {} allocations (in blocks of {})", 
        job_solution.new_allocation.len(), input.block_size);
    for ((vid, tor), count) in &job_solution.new_allocation {
        let seg = virtual_to_segment.get(vid);
        println!("[ILP]   virtual_job={} tor={} count={} blocks ({} workers) -> segment={:?}", 
            vid, tor, count, count * input.block_size, seg);
    }
    
    // Convert solution back to segment-based
    // IMPORTANT: Scale counts back up by block_size since the ILP worked in blocks
    // but compute_segment_migrations works in individual workers
    let new_allocation: HashMap<(SegmentId, usize), usize> = job_solution.new_allocation.iter()
        .filter_map(|((vid, tor), &count)| {
            virtual_to_segment.get(vid).map(|&seg| ((seg, *tor), count * input.block_size))
        })
        .collect();
    
    println!("[ILP] Converted to {} segment allocations (scaled back to workers)", new_allocation.len());
    
    Ok(SegmentILPSolution {
        status: job_solution.status,
        new_allocation,
        num_moves: job_solution.num_moves,
    })
}

/// Compute migrations from segment-based ILP solution.
/// Returns migrations grouped by actual job ID.
/// 
/// Uses a two-phase algorithm to handle swaps correctly:
/// - Phase 1: Identify all blocks that will be vacated and mark them as free
/// - Phase 2: Assign vacating blocks to their destination ToRs using the free block pool
pub fn compute_segment_migrations(
    input: &SegmentILPInput,
    solution: &SegmentILPSolution,
    placements: &HashMap<JobId, DHashMap<WorkerId, usize>>,
    hosts_per_leaf: usize,
) -> Vec<JobMigration> {
    let block_size = input.block_size;
    let num_hosts = input.num_tors * hosts_per_leaf;
    let blocks_per_tor = hosts_per_leaf / block_size;
    
    if DEBUG_ILP {
        println!("[compute_segment_migrations] Starting: {} segments, block_size={}", 
            input.fragmented_segments.len(), block_size);
    }
    
    // Initialize block availability from ALL current placements
    let mut block_available: Vec<bool> = vec![true; num_hosts / block_size];
    #[allow(unused_variables)]
    let mut block_owner: Vec<Option<JobId>> = vec![None; num_hosts / block_size];
    for (&job_id, worker_hosts) in placements.iter() {
        for &host in worker_hosts.values() {
            let block_idx = host / block_size;
            if block_idx < block_available.len() {
                block_available[block_idx] = false;
                block_owner[block_idx] = Some(job_id);
            }
        }
    }
    
    // Pending block moves: (seg_id, src_block, src_tor, dst_tor, workers_in_block)
    let mut pending_block_moves: Vec<(SegmentId, usize, usize, usize, Vec<(WorkerId, usize)>)> = Vec::new();
    
    // Blocks needed per (segment, tor)
    let mut needed_per_seg_tor: HashMap<(SegmentId, usize), usize> = HashMap::new();
    
    // ========== PHASE 1: Identify excess blocks and mark them free ==========
    for &seg_id in &input.fragmented_segments {
        let segment = match input.segments.get(&seg_id) {
            Some(s) => s,
            None => continue,
        };
        
        let job_hosts = match placements.get(&seg_id.job_id) {
            Some(jh) => jh,
            None => continue,
        };
        
        // Get workers for this segment with their hosts
        let workers: Vec<(WorkerId, usize)> = segment.worker_ids.iter()
            .filter_map(|&wid| job_hosts.get(&wid).map(|&h| (wid, h)))
            .collect();
        
        if workers.is_empty() {
            continue;
        }
        
        // Group workers by block
        let mut blocks: HashMap<usize, Vec<(WorkerId, usize)>> = HashMap::new();
        for &(wid, host) in &workers {
            let block_idx = host / block_size;
            blocks.entry(block_idx).or_default().push((wid, host));
        }
        
        // Current blocks per ToR for this segment
        let mut blocks_per_tor_current: HashMap<usize, Vec<usize>> = HashMap::new();
        for &block_idx in blocks.keys() {
            let tor = (block_idx * block_size) / hosts_per_leaf;
            blocks_per_tor_current.entry(tor).or_default().push(block_idx);
        }
        
        // Get target allocation (in workers, need to convert to blocks)
        // Note: solution.new_allocation is already scaled back to workers
        let mut target_blocks_per_tor: HashMap<usize, usize> = HashMap::new();
        for tor in 0..input.num_tors {
            let worker_count = solution.new_allocation.get(&(seg_id, tor)).copied().unwrap_or(0);
            let block_count = worker_count / block_size;
            if block_count > 0 {
                target_blocks_per_tor.insert(tor, block_count);
            }
        }
        
        // Collect current blocks count per ToR
        #[allow(unused_variables)]
        let current_blocks_per_tor: HashMap<usize, usize> = blocks_per_tor_current.iter()
            .map(|(&tor, blocks)| (tor, blocks.len()))
            .collect();
        
        if DEBUG_ILP {
            println!("[ILP] Segment {:?}: current={:?}, target={:?}",
                seg_id, current_blocks_per_tor, target_blocks_per_tor);
        }
        
        // Identify excess blocks (evict lowest-indexed blocks for clean splits)
        let mut excess_blocks: Vec<(usize, usize)> = Vec::new(); // (block_idx, src_tor)
        
        for tor in 0..input.num_tors {
            let current_count = blocks_per_tor_current.get(&tor).map(|v| v.len()).unwrap_or(0);
            let target_count = target_blocks_per_tor.get(&tor).copied().unwrap_or(0);
            
            if current_count > target_count {
                let mut tor_blocks = blocks_per_tor_current.get(&tor).cloned().unwrap_or_default();
                // Sort descending - evict highest-indexed blocks (consistent with compute_migrations_blocked)
                tor_blocks.sort_by(|a, b| b.cmp(a));
                let to_evict = current_count - target_count;
                for &block_idx in tor_blocks.iter().take(to_evict) {
                    excess_blocks.push((block_idx, tor));
                }
            } else if target_count > current_count {
                needed_per_seg_tor.insert((seg_id, tor), target_count - current_count);
            }
        }
        
        // Match excess blocks to destination ToRs and mark source blocks as free
        for (block_idx, src_tor) in excess_blocks {
            let mut assigned_dst = None;
            for tor in 0..input.num_tors {
                let key = (seg_id, tor);
                if let Some(count) = needed_per_seg_tor.get_mut(&key) {
                    if *count > 0 {
                        *count -= 1;
                        assigned_dst = Some(tor);
                        break;
                    }
                }
            }
            
            if let Some(dst_tor) = assigned_dst {
                let workers_in_block = blocks.get(&block_idx).cloned().unwrap_or_default();
                pending_block_moves.push((seg_id, block_idx, src_tor, dst_tor, workers_in_block));
                // Mark source block as free (enables swap handling)
                if block_idx < block_available.len() {
                    block_available[block_idx] = true;
                }
            }
        }
    }
    
    // Collect which jobs are migrating for validation
    #[allow(unused_variables)]
    let migrating_segments: std::collections::HashSet<SegmentId> = pending_block_moves.iter()
        .map(|(seg_id, _, _, _, _)| *seg_id)
        .collect();
    let migrating_jobs: std::collections::HashSet<JobId> = migrating_segments.iter()
        .map(|seg| seg.job_id)
        .collect();
    
    // Phase 2: Assign blocks to their destination ToRs
    let mut migrations_by_job: HashMap<JobId, DHashMap<WorkerId, usize>> = HashMap::new();
    
    for (seg_id, _src_block, _src_tor, dst_tor, workers_in_block) in pending_block_moves {
        let block_start = dst_tor * blocks_per_tor;
        let block_end = block_start + blocks_per_tor;
        
        // Find a free block on the destination ToR
        let mut found_block = None;
        for block_idx in block_start..block_end {
            if block_idx < block_available.len() && block_available[block_idx] {
                found_block = Some(block_idx);
                break;
            }
        }
        
        if let Some(new_block) = found_block {
            let new_block_host_start = new_block * block_size;
            
            // Map workers to hosts in the new block (preserving offsets)
            for (wid, old_host) in workers_in_block {
                let offset = old_host % block_size;
                let new_host = new_block_host_start + offset;
                migrations_by_job
                    .entry(seg_id.job_id)
                    .or_default()
                    .insert(wid, new_host);
            }
            
            block_available[new_block] = false;
        }
    }
    
    // Convert to JobMigration list
    let result: Vec<JobMigration> = migrations_by_job.into_iter()
        .filter(|(_, m)| !m.is_empty())
        .map(|(job_id, worker_to_host)| JobMigration { job_id, worker_to_host })
        .collect();
    
    // Validate: ensure no target host is occupied by a non-migrating job
    let mut host_to_job: Vec<Option<JobId>> = vec![None; num_hosts];
    for (&job_id, worker_hosts) in placements.iter() {
        for &host in worker_hosts.values() {
            if host < num_hosts {
                host_to_job[host] = Some(job_id);
            }
        }
    }
    
    for job_mig in &result {
        for (&_worker_id, &target_host) in job_mig.worker_to_host.iter() {
            if let Some(current_occupant) = host_to_job[target_host] {
                if current_occupant != job_mig.job_id && !migrating_jobs.contains(&current_occupant) {
                    panic!("[ILP] Invalid migration: job {} targets host {} occupied by non-migrating job {}",
                        job_mig.job_id, target_host, current_occupant);
                }
            }
        }
    }
    
    result
}
