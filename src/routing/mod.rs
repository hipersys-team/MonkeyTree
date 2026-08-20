//! Routing algorithm implementations for network simulation.
//! 
//! This module contains various routing implementations that work with
//! the routing traits defined in the parent routing module.

pub mod ecmp;
pub mod crux;
pub mod system_router;

// Re-export the implementations for easy access
pub use ecmp::EcmpRouter;
pub use crux::CruxRouter;
pub use system_router::SystemRouter;
// pub use debug::DebugRouter;  // If moved here
