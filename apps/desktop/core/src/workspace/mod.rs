//! The workspace: one `WorkFs` per session, what can be grafted into it, and the file
//! operations a tree is drawn from.

pub mod connectors;
pub mod fsops;
pub mod manager;
pub mod mount;
pub mod shared;

pub use manager::*;
pub use mount::WorkspaceMount;
pub use shared::SharedFs;
