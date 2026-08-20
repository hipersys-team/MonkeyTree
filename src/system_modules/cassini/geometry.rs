use crate::system_modules::cassini::types::{JobProfile, UnifiedCircle, CompatibilityScore};
use std::f64::consts::PI;

/// Manages geometric abstractions for Cassini scheduling
pub struct GeometricAbstraction {
    /// Angular resolution in degrees for bandwidth sampling
    angular_resolution_deg: f64,
}

impl GeometricAbstraction {
    pub fn new() -> Self {
        Self {
            angular_resolution_deg: 5.0, // Default 5 degree resolution
        }
    }
    
    pub fn with_angular_resolution(mut self, resolution_deg: f64) -> Self {
        self.angular_resolution_deg = resolution_deg.max(0.1).min(90.0);
        self
    }
    
    /// Creates a unified circle for a job with the given LCM period
    pub fn create_unified_circle(&self, job_profile: &JobProfile, lcm_period_us: u64) -> UnifiedCircle {
        let num_iterations = lcm_period_us / job_profile.iteration_time_us;
        let num_samples = (360.0 / self.angular_resolution_deg) as usize;
        let mut bandwidth_samples = vec![0u64; num_samples];
        
        // Map communication phases to angular positions
        for iteration in 0..num_iterations {
            let iteration_start_angle = (iteration as f64 * 2.0 * PI) / num_iterations as f64;
            self.map_phases_to_circle(
                &job_profile.communication_phases,
                job_profile.iteration_time_us,
                iteration_start_angle,
                &mut bandwidth_samples,
            );
        }
        
        UnifiedCircle {
            job_id: job_profile.job_id,
            perimeter_us: lcm_period_us,
            num_iterations,
            bandwidth_samples,
            angular_resolution_deg: self.angular_resolution_deg,
        }
    }
    
    /// Maps communication phases to angular positions on the circle
    fn map_phases_to_circle(
        &self,
        phases: &[crate::system_modules::cassini::types::CommunicationPhase],
        iteration_time_us: u64,
        start_angle: f64,
        bandwidth_samples: &mut [u64],
    ) {
        let mut current_time = 0u64;
        
        for phase in phases {
            let phase_start_angle = start_angle + (current_time as f64 * 2.0 * PI) / iteration_time_us as f64;
            let phase_end_angle = start_angle + ((current_time + phase.duration_us) as f64 * 2.0 * PI) / iteration_time_us as f64;
            
            // Map the phase to bandwidth samples
            self.add_phase_to_samples(
                phase_start_angle,
                phase_end_angle,
                phase.bandwidth_demand,
                bandwidth_samples,
            );
            
            current_time += phase.duration_us;
        }
    }
    
    /// Adds a communication phase to the bandwidth samples
    fn add_phase_to_samples(
        &self,
        start_angle: f64,
        end_angle: f64,
        bandwidth_demand: u64,
        bandwidth_samples: &mut [u64],
    ) {
        let num_samples = bandwidth_samples.len();
        let angle_per_sample = 2.0 * PI / num_samples as f64;
        
        // Normalize angles to [0, 2π)
        let start_angle = start_angle % (2.0 * PI);
        let end_angle = end_angle % (2.0 * PI);
        
        if start_angle <= end_angle {
            // Normal case: phase doesn't wrap around
            let start_sample = (start_angle / angle_per_sample) as usize;
            let end_sample = (end_angle / angle_per_sample) as usize;
            
            for i in start_sample..=end_sample.min(num_samples - 1) {
                bandwidth_samples[i] = bandwidth_samples[i].max(bandwidth_demand);
            }
        } else {
            // Wrap-around case: phase crosses the 0/2π boundary
            // From start_angle to 2π
            let start_sample = (start_angle / angle_per_sample) as usize;
            for i in start_sample..num_samples {
                bandwidth_samples[i] = bandwidth_samples[i].max(bandwidth_demand);
            }
            
            // From 0 to end_angle
            let end_sample = (end_angle / angle_per_sample) as usize;
            for i in 0..=end_sample {
                bandwidth_samples[i] = bandwidth_samples[i].max(bandwidth_demand);
            }
        }
    }
    
    /// Computes the compatibility score between jobs on a link
    /// Returns the optimal rotation angle and compatibility score
    pub fn compute_compatibility(
        &self,
        circles: &[UnifiedCircle],
        link_capacity: u64,
    ) -> (Vec<f64>, CompatibilityScore) {
        if circles.is_empty() {
            return (vec![], CompatibilityScore::fully_compatible());
        }
        
        if circles.len() == 1 {
            // Single job is always fully compatible
            return (vec![0.0], CompatibilityScore::fully_compatible());
        }
        
        // Try different rotation angles for all jobs except the first (reference job)
        let mut best_score = CompatibilityScore::incompatible();
        let mut best_rotations = vec![0.0; circles.len()];
        
        // Use a coarse search first, then refine
        let angle_step = self.angular_resolution_deg * PI / 180.0; // Convert to radians
        let max_angles_to_try = (360.0 / self.angular_resolution_deg) as usize;
        
        self.search_rotations(&circles, link_capacity, angle_step, max_angles_to_try, &mut best_rotations, &mut best_score);
        
        (best_rotations, best_score)
    }
    
    /// Searches for optimal rotation angles using a grid search approach
    fn search_rotations(
        &self,
        circles: &[UnifiedCircle],
        link_capacity: u64,
        angle_step: f64,
        max_steps: usize,
        best_rotations: &mut [f64],
        best_score: &mut CompatibilityScore,
    ) {
        if circles.len() <= 1 {
            return;
        }
        
        // Generate rotation combinations for jobs 1..n (job 0 is reference)
        let num_jobs = circles.len();
        let mut current_rotations = vec![0.0; num_jobs];
        
        // Use recursive search for simplicity (could be optimized for many jobs)
        self.recursive_rotation_search(
            circles,
            link_capacity,
            angle_step,
            max_steps,
            &mut current_rotations,
            1, // Start with job 1 (job 0 is reference)
            best_rotations,
            best_score,
        );
    }
    
    /// Recursive helper for rotation search
    fn recursive_rotation_search(
        &self,
        circles: &[UnifiedCircle],
        link_capacity: u64,
        angle_step: f64,
        max_steps: usize,
        current_rotations: &mut [f64],
        job_index: usize,
        best_rotations: &mut [f64],
        best_score: &mut CompatibilityScore,
    ) {
        if job_index >= circles.len() {
            // Evaluate this rotation combination
            let score = self.evaluate_rotation_combination(circles, current_rotations, link_capacity);
            if score.score > best_score.score {
                *best_score = score;
                best_rotations.copy_from_slice(current_rotations);
            }
            return;
        }
        
        // Try different rotation angles for this job
        for step in 0..max_steps {
            current_rotations[job_index] = step as f64 * angle_step;
            
            self.recursive_rotation_search(
                circles,
                link_capacity,
                angle_step,
                max_steps,
                current_rotations,
                job_index + 1,
                best_rotations,
                best_score,
            );
        }
    }
    
    /// Evaluates a specific rotation combination
    fn evaluate_rotation_combination(
        &self,
        circles: &[UnifiedCircle],
        rotations: &[f64],
        link_capacity: u64,
    ) -> CompatibilityScore {
        if circles.is_empty() {
            return CompatibilityScore::fully_compatible();
        }
        
        let num_samples = circles[0].bandwidth_samples.len();
        let mut total_excess = 0u64;
        let mut total_samples = 0usize;
        
        // Check each angular position
        for sample_idx in 0..num_samples {
            let mut total_demand = 0u64;
            
            // Sum bandwidth demands from all jobs at this angular position
            for (job_idx, circle) in circles.iter().enumerate() {
                let rotated_sample_idx = self.apply_rotation_to_sample(
                    sample_idx,
                    rotations[job_idx],
                    num_samples,
                );
                
                if rotated_sample_idx < circle.bandwidth_samples.len() {
                    total_demand += circle.bandwidth_samples[rotated_sample_idx];
                }
            }
            
            // Calculate excess demand
            if total_demand > link_capacity {
                total_excess += total_demand - link_capacity;
            }
            total_samples += 1;
        }
        
        // Calculate compatibility score
        if total_samples == 0 {
            return CompatibilityScore::fully_compatible();
        }
        
        let average_excess = total_excess as f64 / (total_samples as f64 * link_capacity as f64);
        let score = 1.0 - average_excess;
        
        CompatibilityScore::new(score)
    }
    
    /// Applies rotation to a sample index
    fn apply_rotation_to_sample(
        &self,
        sample_idx: usize,
        rotation_angle: f64,
        num_samples: usize,
    ) -> usize {
        let rotation_samples = (rotation_angle * num_samples as f64 / (2.0 * PI)) as i32;
        let rotated_idx = (sample_idx as i32 - rotation_samples) % num_samples as i32;
        
        if rotated_idx < 0 {
            (rotated_idx + num_samples as i32) as usize
        } else {
            rotated_idx as usize
        }
    }
    
    /// Calculates the Least Common Multiple of iteration times
    pub fn calculate_lcm(&self, iteration_times: &[u64]) -> u64 {
        if iteration_times.is_empty() {
            return 1;
        }
        
        iteration_times.iter().fold(iteration_times[0], |acc, &time| {
            self.lcm(acc, time)
        })
    }
    
    /// Calculates LCM of two numbers
    fn lcm(&self, a: u64, b: u64) -> u64 {
        if a == 0 || b == 0 {
            return 0;
        }
        
        (a * b) / self.gcd(a, b)
    }
    
    /// Calculates GCD of two numbers using Euclidean algorithm
    fn gcd(&self, mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }
}

impl Default for GeometricAbstraction {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to get bandwidth demand at a specific angle from a unified circle
impl UnifiedCircle {
    pub fn get_bandwidth_at_angle(&self, angle: f64) -> u64 {
        let normalized_angle = angle % (2.0 * PI);
        let num_samples = self.bandwidth_samples.len();
        let sample_idx = (normalized_angle * num_samples as f64 / (2.0 * PI)) as usize;
        
        if sample_idx < self.bandwidth_samples.len() {
            self.bandwidth_samples[sample_idx]
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_modules::cassini::types::{JobProfile, CommunicationPhase};
    
    #[test]
    fn test_lcm_calculation() {
        let geom = GeometricAbstraction::new();
        
        assert_eq!(geom.calculate_lcm(&[]), 1);
        assert_eq!(geom.calculate_lcm(&[60]), 60);
        assert_eq!(geom.calculate_lcm(&[40, 60]), 120);
        assert_eq!(geom.calculate_lcm(&[15, 25, 35]), 525);
    }
    
    #[test]
    fn test_unified_circle_creation() {
        let geom = GeometricAbstraction::new();
        
        // Create a simple job profile
        let profile = JobProfile {
            job_id: 1,
            iteration_time_us: 100,
            communication_phases: vec![
                CommunicationPhase {
                    duration_us: 70,
                    bandwidth_demand: 0,
                    is_up_phase: false,
                },
                CommunicationPhase {
                    duration_us: 30,
                    bandwidth_demand: 1000,
                    is_up_phase: true,
                },
            ],
            num_workers: 2,
            name: Some("test_job".to_string()),
        };
        
        let circle = geom.create_unified_circle(&profile, 100);
        
        assert_eq!(circle.job_id, 1);
        assert_eq!(circle.perimeter_us, 100);
        assert_eq!(circle.num_iterations, 1);
        assert!(!circle.bandwidth_samples.is_empty());
        
        // Check that some samples have bandwidth > 0 (communication phases)
        let has_communication = circle.bandwidth_samples.iter().any(|&bw| bw > 0);
        assert!(has_communication);
    }
    
    #[test]
    fn test_single_job_compatibility() {
        let geom = GeometricAbstraction::new();
        
        let profile = JobProfile {
            job_id: 1,
            iteration_time_us: 100,
            communication_phases: vec![
                CommunicationPhase {
                    duration_us: 100,
                    bandwidth_demand: 500,
                    is_up_phase: true,
                },
            ],
            num_workers: 1,
            name: None,
        };
        
        let circle = geom.create_unified_circle(&profile, 100);
        let (rotations, score) = geom.compute_compatibility(&[circle], 1000);
        
        assert_eq!(rotations.len(), 1);
        assert_eq!(rotations[0], 0.0);
        assert!(score.is_fully_compatible);
    }
    
    #[test]
    fn test_two_job_compatibility() {
        let geom = GeometricAbstraction::new();
        
        // Create two jobs with different patterns
        let profile1 = JobProfile {
            job_id: 1,
            iteration_time_us: 100,
            communication_phases: vec![
                CommunicationPhase {
                    duration_us: 50,
                    bandwidth_demand: 800,
                    is_up_phase: true,
                },
                CommunicationPhase {
                    duration_us: 50,
                    bandwidth_demand: 0,
                    is_up_phase: false,
                },
            ],
            num_workers: 2,
            name: None,
        };
        
        let profile2 = JobProfile {
            job_id: 2,
            iteration_time_us: 100,
            communication_phases: vec![
                CommunicationPhase {
                    duration_us: 50,
                    bandwidth_demand: 0,
                    is_up_phase: false,
                },
                CommunicationPhase {
                    duration_us: 50,
                    bandwidth_demand: 800,
                    is_up_phase: true,
                },
            ],
            num_workers: 2,
            name: None,
        };
        
        let circle1 = geom.create_unified_circle(&profile1, 100);
        let circle2 = geom.create_unified_circle(&profile2, 100);
        
        let (rotations, score) = geom.compute_compatibility(&[circle1, circle2], 1000);
        
        assert_eq!(rotations.len(), 2);
        // These jobs should be compatible since their communication phases are complementary
        assert!(score.score > 0.8);
    }
}