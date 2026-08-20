use crate::network::topology::Topology;
use crate::simulator::ml_job::{MLJob, JobId};
use crate::simulator::job_scheduler::JobScheduler;
use crate::simulator::flow_scheduler::FlowScheduler;
use crate::simulator::ml_simulator::MLContext;
use crate::utils::DHashMap;
use crate::simulator::ml_worker::WorkerId;

/// Unique identifier for a system timer.
pub type TimerId = u64;

/// A pluggable system-wide module that can observe and influence scheduling.
///
/// The trait is generic over the simulator's `Topology`, `JobScheduler`, and
/// `FlowScheduler` types. Implementations may add stronger bounds using where-clauses
/// (e.g., require a specific topology or a specialized flow scheduler).
pub trait SystemModule<T: Topology, S: JobScheduler, FS: FlowScheduler> {
    /// One-time initialization hook. Called when the simulator is constructed.
    /// Implementations may inspect the context and topology or mutate the schedulers.
    fn on_init(&mut self, _ctx: &MLContext, _topo: &T, _scheduler: &mut S, _flow_scheduler: &mut FS) {}

    /// Called when a job transitions to Scheduled/Running.
    /// Implementations may reconfigure routing and flow scheduling for future flows.
    /// Active in-flight flows continue on their existing paths.
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
    }

    /// Called when a job completes.
    /// Implementations may reconfigure routing and flow scheduling for future flows.
    /// Active in-flight flows continue on their existing paths.
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
    }

    /// Called when a reconfiguration is needed (e.g., after job changes).
    /// Implementations may reconfigure routing and flow scheduling for future flows.
    /// Active in-flight flows continue on their existing paths.
    /// Optionally returns a MigrationPlan to be applied.
    fn on_reconfigure(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _topo: &T,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) -> Option<MigrationPlan> {
        None
    }

    /// Called when a job's migration is complete.
    /// The job has been un-paused and will resume execution.
    /// Implementations may recompute routes based on new placements.
    /// 
    /// The `job` parameter contains the job after rank reassignment, allowing
    /// implementations to re-record flow templates with updated worker IDs.
    fn on_migration_end(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _job_id: JobId,
        _job: &MLJob,
        _topo: &T,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
    }

    /// Called when a system timer fires.
    /// 
    /// System modules can schedule timers by pushing to `ctx.pending_timers`.
    /// The `timer_id` is the ID that was provided when scheduling the timer.
    /// 
    /// # Example
    /// ```ignore
    /// // In on_init or another hook:
    /// ctx.schedule_timer(100, 1); // Fire timer ID 1 in 100 microseconds
    /// 
    /// // In on_timer:
    /// fn on_timer(&mut self, now_us: u64, ctx: &MLContext, timer_id: TimerId, ...) {
    ///     if timer_id == 1 {
    ///         // Handle timer, optionally reschedule
    ///         ctx.schedule_timer(100, 1); // Periodic timer
    ///     }
    /// }
    /// ```
    fn on_timer(
        &mut self,
        _now_us: u64,
        _ctx: &MLContext,
        _timer_id: TimerId,
        _topo: &T,
        _scheduler: &mut S,
        _flow_scheduler: &mut FS,
    ) {
    }
}

/// Default no-op implementation of `SystemModule`.
#[derive(Debug, Default)]
pub struct NoopSystemModule;

impl<T: Topology, S: JobScheduler, FS: FlowScheduler> SystemModule<T, S, FS> for NoopSystemModule {}

/// Complete migration plan returned by system module.
/// Migration time is determined by actual network flows transferring worker model data.
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    pub jobs: Vec<JobMigration>,
}

/// Per-job worker placement in a migration
#[derive(Debug, Clone)]
pub struct JobMigration {
    pub job_id: JobId,
    /// Mapping from worker ID to new host index
    pub worker_to_host: DHashMap<WorkerId, usize>,
}

/// Information about a single worker migration (computed by the simulator)
#[derive(Debug, Clone)]
pub struct WorkerMigrationInfo {
    pub job_id: JobId,
    pub worker_id: WorkerId,
    pub src_host: usize,
    pub dst_host: usize,
    pub model_size_bytes: u64,
}

/// Per-job pending migration state, used for synchronizing migrations with iteration barriers.
#[derive(Debug, Clone)]
pub struct PendingJobMigration {
    /// Workers that will move with their migration info
    pub moves: Vec<WorkerMigrationInfo>,
    /// Jobs that must reach their iteration barrier before this job's migration can start
    /// (because this job wants to migrate to hosts currently occupied by those jobs)
    pub waiting_for: crate::utils::DHashSet<JobId>,
    /// Has this job reached its iteration barrier?
    pub at_barrier: bool,
    /// Have migration flows been started for this job?
    pub flows_started: bool,
}

/// Base index for migration flow templates in routers.
/// Migration flow for worker W uses job_flow_idx = MIGRATION_FLOW_IDX_BASE + W.
/// This ensures migration flows don't conflict with regular job flow indices.
pub const MIGRATION_FLOW_IDX_BASE: usize = 1_000_000;

/// Compute the migration flow index for a given worker.
#[inline]
pub fn migration_flow_idx(worker_id: WorkerId) -> usize {
    MIGRATION_FLOW_IDX_BASE + worker_id
}

