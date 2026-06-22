use std::sync::Arc;

use crate::vfs::accessor::{GDriveConfig, NotionConfig, S3Config};
use crate::vfs::path::VPath;
use crate::vfs::resource::{Resource, S3Resource};

/// Per-mount provider configuration (carries credentials, host-only).
#[derive(Clone)]
pub enum ProviderConfig {
    S3(S3Config),
    Notion(NotionConfig),
    GDrive(GDriveConfig),
}

/// One mount: a virtual top-level prefix bound to a provider config. The same
/// provider type may appear multiple times with different credentials.
#[derive(Clone)]
pub struct MountSpec {
    pub prefix: String,
    pub provider: ProviderConfig,
}

#[derive(Clone, Default)]
pub struct VfsConfig {
    pub mounts: Vec<MountSpec>,
}

/// A live mount: prefix bound to an instantiated [`Resource`].
pub struct Mount {
    pub prefix: String,
    pub resource: Arc<dyn Resource>,
}

/// Routes virtual paths to mounts by longest-prefix match. Single source of
/// truth for provider access shared by both FUSE frontends.
pub struct Vfs {
    mounts: Vec<Mount>,
}

impl Vfs {
    /// Build from live mounts. Rejects duplicate or empty prefixes; sorts by
    /// prefix length descending for longest-match routing.
    pub fn new(mut mounts: Vec<Mount>) -> anyhow::Result<Self> {
        for m in &mounts {
            if m.prefix.is_empty() || !m.prefix.starts_with('/') {
                anyhow::bail!("mount prefix must be absolute and non-empty: {:?}", m.prefix);
            }
        }
        for i in 0..mounts.len() {
            for j in (i + 1)..mounts.len() {
                if mounts[i].prefix == mounts[j].prefix {
                    anyhow::bail!("duplicate mount prefix: {}", mounts[i].prefix);
                }
            }
        }
        mounts.sort_by(|a, b| b.prefix.len().cmp(&a.prefix.len()));
        Ok(Self { mounts })
    }

    /// Instantiate resources from a [`VfsConfig`] and build the VFS.
    pub fn from_config(config: VfsConfig) -> anyhow::Result<Self> {
        let mut mounts = Vec::with_capacity(config.mounts.len());
        for spec in config.mounts {
            let resource: Arc<dyn Resource> = match spec.provider {
                ProviderConfig::S3(c) => Arc::new(S3Resource::new(&c)?),
                ProviderConfig::Notion(_) => {
                    anyhow::bail!("notion adapter not implemented yet")
                }
                ProviderConfig::GDrive(_) => {
                    anyhow::bail!("gdrive adapter not implemented yet")
                }
            };
            mounts.push(Mount {
                prefix: spec.prefix,
                resource,
            });
        }
        Vfs::new(mounts)
    }

    pub fn mounts(&self) -> &[Mount] {
        &self.mounts
    }

    /// Top-level mount names (prefix without leading `/`), for VFS-root readdir.
    pub fn mount_names(&self) -> Vec<String> {
        self.mounts
            .iter()
            .map(|m| m.prefix.trim_start_matches('/').to_string())
            .collect()
    }

    /// Resolve an absolute virtual path (e.g. `/s3-prod/data/x.csv`) to its
    /// owning resource and the mount-relative [`VPath`].
    pub fn route(&self, abs: &str) -> Option<(&Arc<dyn Resource>, VPath)> {
        let abs = abs.trim_end_matches('/');
        for m in &self.mounts {
            if abs == m.prefix {
                return Some((&m.resource, VPath::root()));
            }
            if let Some(rest) = abs.strip_prefix(&m.prefix) {
                if rest.starts_with('/') {
                    return Some((&m.resource, VPath::new(rest)));
                }
            }
        }
        None
    }
}
