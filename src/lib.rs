pub mod network;
pub mod simulator;
pub mod routing;
pub mod spine;
pub mod fat;
pub mod rail;
pub mod flow_scheduler;
pub mod utils;
pub mod schedulers;
pub mod job_schedulers;
pub mod system_modules;
pub mod monkeytree;
pub mod collectives;

// make the main primitives available at the crate root for convenience
pub use network::{Simulator, EventKind};
pub use simulator::{MLSimulator, JobScheduler, FifoScheduler, MLJob, ml_job::MLJobBuilder, MLWorker, WorkerEvent, FlowScheduler, ImmediateFlowScheduler};
pub use job_schedulers::SimpleScheduler;
pub use system_modules::CassiniSystemModule;
pub use schedulers::{BlockScheduler, ClusterScheduler, FifoBlockScheduler, RandomBlockScheduler, SnapshotScheduler, DEFAULT_BLOCK_SIZE};
pub use flow_scheduler::release_scheduler::{ReleaseFlowScheduler, FlowReleaseSchedule, FlowReleaseSpec};
pub use routing::{EcmpRouter};
pub use spine::{SpineTree, SpineTreeTopology, SpineTreeRouter, SpineEcmpRouter, SpineCruxRouter, SGLBRouter, SGLBSystemModule, SGLBConfig};
pub use monkeytree::{MonkeyTreeSystem, MonkeyTreeEcmp, MonkeyTreeCrux, MonkeyTreePerfect, MonkeyTreeSGLB, SpinePerfectRouter, MonkeyTreeConfig};
pub use monkeytree::{RailMonkeyTreeSystem, RailMonkeyTreeCrux, RailMonkeyTreePerfect};
pub use fat::{FatTreePerfectRouter, FatTreePerfectSystem, FatTreeCruxRouter, FatTreeCruxSystem, FatTreeMonkeyTreePerfect, FatTreeMonkeyTree3, FatTreeSGLBRouter, FatTreeSGLBSystem};
pub use rail::{RailTree, RailTopology, RailTreeRouter, RailEcmpRouter, RailCruxRouter, RailCruxSystemModule, RailPerfectRouter, RailPerfectRoutingSystem};
pub use collectives::{CollectiveJobBuilder, CollectiveOp, JobPhase};
pub use utils::validation::{validate_job_for_block_scheduler, validate_jobs_for_block_scheduler, validate_worker_count, validate_topology_for_blocks};