//! Virtual filesystem over external providers (S3, Notion, Google Drive).
//!
//! A single [`Vfs`] holds the provider [`Resource`]s and routes virtual paths
//! to them. It is the single source of truth for provider access; FUSE
//! frontends (host `fuser` mount, guest `mfusepy` forwarder) sit on top.
//!
//! See `docs/vfs-integration-report.md` for the design rationale.

pub mod accessor;
pub mod forward;
pub mod fuse;
pub mod guest;
pub mod path;
pub mod resource;
pub mod session;

#[allow(clippy::module_inception)]
mod vfs;

pub use accessor::{GDriveConfig, NotionConfig, S3Config};
pub use forward::VfsForward;
pub use fuse::VfsMount;
pub use guest::bootstrap_guest_forwarder;
pub use path::VPath;
pub use resource::{DirEntry, FileKind, FileStat, Resource, S3Resource};
pub use session::AgentVfs;
pub use vfs::{Mount, MountSpec, ProviderConfig, Vfs, VfsConfig};
