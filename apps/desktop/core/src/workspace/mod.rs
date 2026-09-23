//! The workspace: one `ContextFs` per session, what can be grafted into it, and the file
//! operations a tree is drawn from.

pub mod connectors;
pub mod fsops;
pub mod manager;
pub mod shared;

pub use manager::*;
pub use shared::SharedFs;
