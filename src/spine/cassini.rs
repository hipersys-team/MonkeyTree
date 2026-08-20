use crate::system_modules::cassini::{
    CassiniSystemModule as BaseCassiniSystemModule,
    JobProfiler, GeometricAbstraction, ILPOptimizer, AffinityGraph,
    JobProfile, PlacementCandidate, CassiniSchedule, TimeShift, CompatibilityScore
};
use crate::spine::topology::{SpineTreeTopology, SpineTree, SpineTreeRouter};
use crate::network::topology::Topology;
use crate::spine::routing::SpineRouteOracle;
use crate::simulator::{
    ml_job::{MLJob, JobId},
    job_scheduler::JobScheduler,
    flow_scheduler::FlowScheduler,
    ml_simulator::MLContext,
    system::SystemModule
};
use crate::flow_scheduler::CassiniFlowScheduler;
use std::collections::{HashMap, HashSet};

/// Spine-specific Cassini system module that handles network-aware job scheduling
/// for spine-leaf topologies
pub struct SpineCassiniSystemModule {
    /// Base Cassini functionality
    _base: BaseCassiniSystemModule,
    
    /// Job profiler
    profiler: JobProfiler,
    
    /// Geometric abstraction handler
    geometry: GeometricAbstraction,
    
    /// ILP optimizer
    optimizer: ILPOptimizer,
    
    /// Current job profiles
    job_profiles: HashMap<JobId, JobProfile>,
    
    /// Current placement candidates being evaluated
    current_candidates: Vec<PlacementCandidate>,
    
    /// Link capacity in bytes/ms (assume all links have same capacity)
    link_capacity: u64,

    /// When true, do not select placements; use job-provided static placements
    use_static_placement: bool,

    /// Persisted placements per job: worker_id -> host_index
    _job_placements: HashMap<JobId, HashMap<usize, usize>>, // TODO: this is never used
}

impl SpineCassiniSystemModule {
    pub fn new() -> Self {
        Self {
            _base: BaseCassiniSystemModule::new(),
            profiler: JobProfiler::new(),
            geometry: GeometricAbstraction::new(),
            optimizer: ILPOptimizer::new(),
            job_profiles: HashMap::new(),
            current_candidates: Vec::new(),
            link_capacity: 10_000, // Default ~10 MB/s (≈80 Mbps)
            use_static_placement: false,
            _job_placements: HashMap::new(),
        }
    }
   
    // TODO: this method seems useless?
    pub fn with_link_capacity(mut self, capacity_bytes_per_us: u64) -> Self {
        self.link_capacity = capacity_bytes_per_us;
        self
    }
    
    /// Sync bandwidth parameters from the provided topology.
    /// Sets both the profiler throughput baseline and our link_capacity (bytes/ms).
    fn update_bandwidth_from_topology<R: SpineTreeRouter>(&mut self, topology: &SpineTree<R>) {
        let bps: f64 = topology.link_bandwidth_bps();
        // bytes/us = bps / 8 / 1_000_000 = bps / 8_000_000
        let bytes_per_us: u64 = (bps / 8_000_000.0).floor().max(1.0) as u64;
        self.link_capacity = bytes_per_us;
        self.profiler.set_baseline_throughput_bytes_per_us(bytes_per_us);
    }

    /// Enable static placement mode (system module will not pick placements)
    pub fn with_static_placement(mut self) -> Self {
        self.use_static_placement = true;
        self
    }
    
    /// Evaluates placement candidates for a job using Cassini compatibility analysis
    pub fn evaluate_placement_candidates<R: SpineTreeRouter + SpineRouteOracle, S: JobScheduler, FS: FlowScheduler>(
        &mut self,
        job: &MLJob,
        candidates: Vec<PlacementCandidate>,
        topology: &SpineTree<R>,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) -> Option<PlacementCandidate> {
        if candidates.is_empty() { return candidates.into_iter().next(); }
        
        // Ensure profiler/link capacity reflect the current topology bandwidth
        self.update_bandwidth_from_topology(topology);

        // Profile the job if not already done
        if !self.job_profiles.contains_key(&job.id) {
            if let Some(profile) = self.profiler.profile_job(job) {
                self.job_profiles.insert(job.id, profile);
            } else {
                // Create default profile
                let default_profile = self.profiler.create_default_profile(job, 1000);
                self.job_profiles.insert(job.id, default_profile);
            }
        }
        
        let mut best_candidate = None;
        let mut best_score = CompatibilityScore::incompatible();
        
        // Evaluate each placement candidate
        for mut candidate in candidates {
            let score = self.evaluate_single_placement(job, &candidate, topology);
            candidate.compatibility_score = Some(score.clone());
            
            if score.score > best_score.score {
                best_score = score;
                best_candidate = Some(candidate);
            }
        }
        
        best_candidate
    }
    
    /// Evaluates a single placement candidate
    fn evaluate_single_placement<R: SpineTreeRouter + SpineRouteOracle>(
        &self,
        _job: &MLJob,
        candidate: &PlacementCandidate,
        topology: &SpineTree<R>,
    ) -> CompatibilityScore {
        // Build affinity graph for this placement
        let mut affinity_graph = AffinityGraph::new();
        
        // Map job workers to topology paths and identify shared links
        let links_used = self.identify_links_used(candidate, topology);
        
        if links_used.is_empty() {
            return CompatibilityScore::fully_compatible();
        }
        // Add job and links to affinity graph
        affinity_graph.add_job(candidate.job_id);
        for &link_id in &links_used {
            affinity_graph.add_link(link_id);
            affinity_graph.add_job_link_edge(candidate.job_id, link_id, 0); // Time shift will be computed later
        }
        
        // Check for conflicts with existing jobs
        let mut total_score = 0.0;
        let mut evaluated_links = 0;
        for &link_id in &links_used {
            let jobs_on_link = affinity_graph.get_jobs_on_link(link_id);
            if jobs_on_link.len() > 1 {
                // Multiple jobs on this link - compute compatibility
                let score = self.compute_link_compatibility(&jobs_on_link, link_id);
                total_score += score.score;
                evaluated_links += 1;
            } else {
                // Single job on link - fully compatible
                total_score += 1.0;
                evaluated_links += 1;
            }
        }
        
        if evaluated_links == 0 {
            CompatibilityScore::fully_compatible()
        } else {
            CompatibilityScore::new(total_score / evaluated_links as f64)
        }
    }
    
    /// Identifies which links in the spine topology would be used by a job placement
    fn identify_links_used<R: SpineTreeRouter + SpineRouteOracle>(
        &self,
        candidate: &PlacementCandidate,
        topology: &SpineTree<R>,
    ) -> Vec<usize> {
        let mut links_used = HashSet::new();

        // Predict per-pair ECMP path via the router oracle (stateless)
        let hosts_per_leaf = topology.hosts_per_leaf();
        for (&worker1_id, &host1) in &candidate.worker_to_host {
            for (&worker2_id, &host2) in &candidate.worker_to_host {
                if worker1_id == worker2_id { continue; }

                let leaf_idx1 = host1 / hosts_per_leaf;
                let host_off1 = host1 % hosts_per_leaf;
                let leaf_idx2 = host2 / hosts_per_leaf;
                let host_off2 = host2 % hosts_per_leaf;
                let src = topology.get_host(leaf_idx1, host_off1).expect("invalid src host");
                let dst = topology.get_host(leaf_idx2, host_off2).expect("invalid dst host");

                // Predict path based on job_id for iteration-stable ECMP
                let path = topology.router.borrow().predict_path(topology, src, dst, candidate.job_id);
                links_used.extend(path);
            }
        }

        links_used.into_iter().collect()
    }
    
    /// Computes compatibility score for jobs sharing a link
    fn compute_link_compatibility(&self, job_ids: &[JobId], _link_id: usize) -> CompatibilityScore {
        if job_ids.len() <= 1 {
            return CompatibilityScore::fully_compatible();
        }
        
        // Collect job profiles for these jobs
        let mut profiles = Vec::new();
        for &job_id in job_ids {
            if let Some(profile) = self.job_profiles.get(&job_id) {
                profiles.push(profile.clone());
            }
        }
        
        if profiles.is_empty() {
            return CompatibilityScore::incompatible();
        }
        
        // Calculate LCM of iteration times
        let iteration_times: Vec<u64> = profiles.iter().map(|p| p.iteration_time_us).collect();
        let lcm_period = self.geometry.calculate_lcm(&iteration_times);
        
        // Create unified circles
        let circles: Vec<_> = profiles.iter()
            .map(|profile| self.geometry.create_unified_circle(profile, lcm_period))
            .collect();
        
        // Compute compatibility using ILP optimizer
        match self.optimizer.optimize_compatibility(&circles, self.link_capacity) {
            Ok((_rotations, score)) => score,
            Err(_) => CompatibilityScore::incompatible(),
        }
    }
    
    /// Applies Cassini schedule to flow scheduler
    fn apply_schedule_to_flow_scheduler(
        &self,
        now_us: u64,
        schedule: CassiniSchedule,
        flow_scheduler: &mut CassiniFlowScheduler,
    ) {
        // Print a human-readable summary of the schedule being applied
        self.print_schedule_summary(now_us, &schedule);

        // Apply the schedule directly to the Cassini flow scheduler
        flow_scheduler.apply_cassini_schedule(now_us, schedule);
    }
}

impl Default for SpineCassiniSystemModule {
    fn default() -> Self {
        Self::new()
    }
}

// Implementation of SystemModule trait for spine topology integration
impl<R, S> SystemModule<SpineTree<R>, S, CassiniFlowScheduler>
    for SpineCassiniSystemModule
where
    R: SpineTreeRouter + SpineRouteOracle,
    S: JobScheduler + 'static,
{
    fn on_job_scheduled(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        job: &MLJob,
        topology: &SpineTree<R>,
        scheduler: &mut S,
        flow_scheduler: &mut CassiniFlowScheduler,
    ) {
        if self.use_static_placement {
            // Profile the job if missing
            if !self.job_profiles.contains_key(&job.id) {
                self.update_bandwidth_from_topology(topology);
                if let Some(profile) = self.profiler.profile_job(job) {
                    self.job_profiles.insert(job.id, profile);
                } else {
                    let default_profile = self.profiler.create_default_profile(job, 1000);
                    self.job_profiles.insert(job.id, default_profile);
                }
            }
            return;
        }

        // Ask the scheduler for candidates (may return empty for snapshot-like schedulers)
        let num_hosts = topology.num_leaves() * topology.hosts_per_leaf();
        let available_hosts = vec![false; num_hosts];
        let candidates = scheduler.generate_placement_candidates(job, topology, &available_hosts, 10);
        if let Some(best_candidate) = self.evaluate_placement_candidates(
            job,
            candidates,
            topology,
            scheduler,
            flow_scheduler,
        ) {
            self.current_candidates = vec![best_candidate];
        }
    }

    fn on_job_completed(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        job_id: JobId,
        _job: &MLJob,
        _topology: &SpineTree<R>,
        _scheduler: &mut S,
        _flow_scheduler: &mut CassiniFlowScheduler,
    ) {
        self.job_profiles.remove(&job_id);
    }

    fn on_reconfigure(
        &mut self,
        now_us: u64,
        ctx: &MLContext,
        topology: &SpineTree<R>,
        _scheduler: &mut S,
        flow_scheduler: &mut CassiniFlowScheduler,
    ) -> Option<crate::simulator::system::MigrationPlan> {
        if self.job_profiles.is_empty() {
            flow_scheduler.clear_cassini_schedule();
            return None;
        }

        let mut placements: std::collections::HashMap<JobId, std::collections::HashMap<usize, usize>> = std::collections::HashMap::new();
        if let Ok(p) = ctx.placements.try_borrow() {
            for (jid, w2h) in p.iter() {
                let mut map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                for (wid, host) in w2h.iter() { map.insert(*wid, *host); }
                placements.insert(*jid, map);
            }
        }

        if placements.is_empty() {
            flow_scheduler.clear_cassini_schedule();
            return None;
        }

        match self.compute_topology_aware_schedule_from_placements(now_us, topology, &placements) {
            Ok(schedule) => {
                self.apply_schedule_to_flow_scheduler(now_us, schedule, flow_scheduler);
            }
            Err(err) => {
                println!("{} Cassini: cannot install schedule: {}", now_us, err);
                flow_scheduler.clear_cassini_schedule();
            }
        }
        None
    }
}

impl SpineCassiniSystemModule {
    /// Identify links used by a mapping (worker -> host) for a given job using router's ECMP prediction
    fn identify_links_used_for_mapping<R: SpineTreeRouter + SpineRouteOracle>(
        &self,
        job_id: JobId,
        mapping: &std::collections::HashMap<usize, usize>,
        topology: &SpineTree<R>,
    ) -> Vec<usize> {
        // Reuse the same logic as `identify_links_used` by creating a temporary placement
        let candidate = PlacementCandidate { job_id, worker_to_host: mapping.clone(), compatibility_score: None };
        self.identify_links_used(&candidate, topology)
    }

    /// Computes a Cassini schedule using topology-aware per-link optimization and the affinity graph
    /// from an explicit mapping of job placements. Returns an error if the affinity graph contains loops.
    fn compute_topology_aware_schedule_from_placements<R: SpineTreeRouter + SpineRouteOracle>(
        &mut self,
        now_us: u64,
        topology: &SpineTree<R>,
        placements: &std::collections::HashMap<JobId, std::collections::HashMap<usize, usize>>,
    ) -> Result<CassiniSchedule, String> {
        // Ensure profiler/link capacity reflect the current topology bandwidth
        self.update_bandwidth_from_topology(topology);

        // Build link -> jobs map via predicted paths from provided placements
        if placements.is_empty() {
            // Nothing to place; return empty schedule
            let mut schedule = CassiniSchedule::new();
            schedule.computed_at_us = now_us;
            return Ok(schedule);
        }

        let mut affinity = AffinityGraph::new();

        // Collect which jobs traverse which links
        let mut all_links: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (&job_id, mapping) in placements.iter() {
            let links_used = self.identify_links_used_for_mapping(job_id, mapping, topology);
            for link_id in links_used {
                all_links.insert(link_id);
                affinity.add_job_link_edge(job_id, link_id, 0);
            }
        }

        // If the affinity graph has loops, bail with an explicit message
        if affinity.has_loops() {
            return Err("Affinity graph contains a loop; cannot compute a unique schedule".to_string());
        }

        // For each congested link (with >1 job), compute per-link time shifts from geometric optimization
        for link_id in all_links.into_iter() {
            let jobs_on_link = affinity.get_jobs_on_link(link_id);
            if jobs_on_link.len() <= 1 { continue; }

            // Gather job profiles for these jobs
            let mut profiles: Vec<JobProfile> = Vec::new();
            for jid in jobs_on_link.iter() {
                if let Some(p) = self.job_profiles.get(jid) { profiles.push(p.clone()); }
            }
            if profiles.len() <= 1 { continue; }

            // Determine unified circle perimeter using LCM of iteration times for jobs on this link
            let iteration_times: Vec<u64> = profiles.iter().map(|p| p.iteration_time_us).collect();
            let lcm_period = self.geometry.calculate_lcm(&iteration_times);
            if lcm_period == 0 { continue; }

            // Create unified circles and run optimization
            let circles: Vec<_> = profiles.iter()
                .map(|profile| self.geometry.create_unified_circle(profile, lcm_period))
                .collect();

            let result = self.optimizer.optimize_complete(&circles, self.link_capacity);
            if result.success {
                // Convert and assign per-link time shifts as edge weights
                for ts in result.time_shifts {
                    affinity.set_link_time_shift(ts.job_id, link_id, ts.shift_us);
                }
            } else {
                // If optimization fails, fall back to zero shifts on this link
                // (edges already initialized to 0)
            }
        }

        // Compute unique per-job time shifts via affinity graph BFS consolidation
        let mut iter_time_map: std::collections::HashMap<JobId, u64> = std::collections::HashMap::new();
        for (&job_id, profile) in self.job_profiles.iter() {
            iter_time_map.insert(job_id, profile.iteration_time_us);
        }
        let mut unique_shifts = affinity.compute_unique_time_shifts(&iter_time_map);

        // Assemble final schedule; include isolated jobs with zero shift
        let mut schedule = CassiniSchedule::new();
        schedule.computed_at_us = now_us;
        for (&job_id, profile) in self.job_profiles.iter() {
            let ts = unique_shifts.remove(&job_id).unwrap_or(TimeShift { job_id, shift_us: 0, rotation_angle: 0.0 });
            schedule.add_time_shift(ts);
            schedule.job_periods.insert(job_id, profile.iteration_time_us);
        }

        Ok(schedule)
    }
    /// Prints a human-readable summary of a Cassini schedule
    fn print_schedule_summary(&self, now_us: u64, schedule: &CassiniSchedule) {
        let mut job_ids: Vec<_> = schedule.time_shifts.keys().copied().collect();
        job_ids.sort();

        println!(
            "Applying Cassini schedule (version {}, computed_at={} ms) at now={} ms:",
            schedule.version,
            schedule.computed_at_us,
            now_us
        );

        if job_ids.is_empty() {
            println!("  - No jobs scheduled (empty schedule)");
        } else {
            for job_id in job_ids {
                if let Some(ts) = schedule.time_shifts.get(&job_id) {
                    let period = schedule.job_periods.get(&job_id).copied().unwrap_or(0);
                    let pct = if period > 0 { (ts.shift_us as f64 / period as f64) * 100.0 } else { 0.0 };
                    println!(
                        "  - job {}: shift={} ms ({:.1}% of {} ms period), rotation={:.2} rad",
                        job_id,
                        ts.shift_us,
                        pct,
                        period,
                        ts.rotation_angle
                    );
                }
            }
        }
    }

    /// Computes a global Cassini schedule for all active jobs
    fn _compute_global_schedule(&self, now_us: u64) -> CassiniSchedule {
        if self.job_profiles.is_empty() {
            let mut schedule = CassiniSchedule::new();
            schedule.computed_at_us = now_us;
            return schedule;
        }

        // Construct schedule without relying on a global LCM period.
        // Use per-job iteration periods and unique time shifts.
        let mut schedule = CassiniSchedule::new();
        schedule.computed_at_us = now_us;
        
        // For now, create simple time shifts (could be enhanced with full affinity graph analysis)
        for (i, (&job_id, profile)) in self.job_profiles.iter().enumerate() {
            let time_shift = TimeShift {
                job_id,
                // Simple stagger based on quarter-iteration, reduced modulo the job's own period
                shift_us: (i as u64 * profile.iteration_time_us / 4) % profile.iteration_time_us,
                rotation_angle: 0.0,
            };
            schedule.add_time_shift(time_shift);
            schedule.job_periods.insert(job_id, profile.iteration_time_us);
        }
        
        schedule
    }

    /// Computes a Cassini schedule using persisted static placements
    fn _compute_schedule_from_static_placements<R: SpineTreeRouter + SpineRouteOracle>(
        &self,
        now_us: u64,
        _topology: &SpineTree<R>,
    ) -> CassiniSchedule {
        // Start with the same baseline approach as compute_global_schedule.
        // Placement-aware refinements can be added later using identify_links_used.
        // TODO: you cannot be serious
        if self.job_profiles.is_empty() {
            let mut schedule = CassiniSchedule::new();
            schedule.computed_at_us = now_us;
            return schedule;
        }

        let mut schedule = CassiniSchedule::new();
        schedule.computed_at_us = now_us;

        for (i, (&job_id, profile)) in self.job_profiles.iter().enumerate() {
            let time_shift = TimeShift {
                job_id,
                // Simple stagger based on quarter-iteration, reduced modulo the job's own period
                shift_us: (i as u64 * profile.iteration_time_us / 4) % profile.iteration_time_us,
                rotation_angle: 0.0,
            };
            schedule.add_time_shift(time_shift);
            schedule.job_periods.insert(job_id, profile.iteration_time_us);
        }

        schedule
    }
}