use crate::network::topology::Topology;
use crate::simulator::{
    ml_job::{MLJob, JobId},
    job_scheduler::JobScheduler,
    flow_scheduler::FlowScheduler,
    ml_simulator::MLContext,
    system::SystemModule,
};

/// Base Cassini system module with topology-agnostic functionality
/// This provides common functionality that can be extended for specific topologies
#[derive(Debug)]
pub struct CassiniSystemModule {
    /// Version counter for tracking schedule updates
    schedule_version: u64,
}

impl CassiniSystemModule {
    pub fn new() -> Self {
        Self {
            schedule_version: 0,
        }
    }
    
    pub fn next_schedule_version(&mut self) -> u64 {
        self.schedule_version += 1;
        self.schedule_version
    }
    
    pub fn current_schedule_version(&self) -> u64 {
        self.schedule_version
    }
}

impl Default for CassiniSystemModule {
    fn default() -> Self {
        Self::new()
    }
}

// Default implementation of SystemModule trait (no-op)
impl<T: Topology, S: JobScheduler, FS: FlowScheduler> SystemModule<T, S, FS> for CassiniSystemModule {
    fn on_init(&mut self, _ctx: &MLContext, _topo: &T, _scheduler: &mut S, _flow_scheduler: &mut FS) {
        // Base implementation does nothing - specific topology modules will override
    }
    
    fn on_job_scheduled(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        _job: &MLJob,
        _topo: &T,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // Base implementation does nothing - specific topology modules will override
    }
    
    fn on_job_completed(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        _job: &MLJob,
        _topo: &T,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
        // Base implementation does nothing - specific topology modules will override
    }
    
    fn on_reconfigure(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _topo: &T,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) -> Option<crate::simulator::system::MigrationPlan> {
        // Base implementation does nothing - specific topology modules will override
        None
    }
}