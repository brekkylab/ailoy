//! The workspace: one `WorkFs` per session, what can be grafted into it, and the file
//! operations a tree is drawn from.

pub mod connectors;
pub mod fsops;
pub mod shared;

pub use shared::SharedFs;
