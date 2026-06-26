//! Virtual filesystem over external providers (S3, Notion, Google Drive).
//!
//! A single [`Vfs`] holds the provider [`Resource`]s and routes virtual paths
//! to them. It is the single source of truth for provider access; FUSE
//! frontends (host `fuser` mount, static in-guest forwarder) sit on top.
//!
//! See `docs/vfs-integration-report.md` for the design rationale.

pub mod accessor;
mod cache;
pub mod fuse;
pub mod path;
pub mod resource;

/// Sandbox-only frontend (host forward server + in-guest forwarder).
#[cfg(feature = "sandbox")]
pub mod sandbox;

#[allow(clippy::module_inception)]
mod vfs;

pub use accessor::{GDriveConfig, NotionConfig, S3Config};
pub use fuse::VfsMount;
pub use path::VPath;
pub use resource::{DirEntry, FileKind, FileStat, Resource, S3Resource};
// Keep the public paths `crate::vfs::{VfsForward, bootstrap_guest_forwarder}`
// stable; they live in the sandbox submodule but are only present with the feature.
#[cfg(feature = "sandbox")]
pub use sandbox::{VfsForward, bootstrap_guest_forwarder};
pub use vfs::{AgentVfs, Mount, MountSpec, ProviderConfig, Vfs, VfsConfig};
