//! Connectors: a `MountConfig` becomes a `FileSystem`, after one request proves it can answer.
//!
//! **A connection is confirmed before it is mounted.** Both network stores build offline —
//! `S3Fs::new` assembles a client and sends nothing, and `NotionFs::new` builds an HTTP client —
//! so a wrong key, region or bucket would otherwise be invisible until the tree was clicked, and
//! would arrive there as an `EIO` on a listing with nothing to say about which field was wrong.
//! So each connector makes exactly one request here, and the caller mounts only what answered.
//!
//! The store comes back as an `Arc<dyn FileSystem>` rather than a `Box`: cortex implements
//! `FileSystem` for `Arc<T: FileSystem + ?Sized>` and not for `Box<dyn FileSystem>`, and
//! `ContextFs::mount` needs a `FileSystem` by value.

use std::{path::Path, sync::Arc, time::Duration};

use cortex::fs::{FileSystem, NotionConfig, NotionFs, PassthroughFs, S3Config, S3Fs};

use crate::{
    error::{EngineError, Result},
    types::{MountConfig, MountKind},
};

/// How long a connector waits for the confirming request before calling the connection bad.
const PROBE_TIMEOUT: Duration = Duration::from_secs(15);

/// A mount path as the sidebar spells it: `/`-rooted, no trailing slash, never the root itself.
///
/// The path is **rebuilt from its components**, the way `ContextFs` keys its mount table: split on
/// `/`, empty and `.` segments dropped, the rest joined back under a leading `/`. That is what
/// keeps the row in the mount list and the key in the tree the same string — `"/mem/."` and
/// `"/a//b"` are `ContextFs`'s `"mem"` and `"a/b"` whatever the list says, so a row that kept the
/// user's spelling would name a mount nobody could detach.
///
/// A `..` segment is refused rather than resolved: it is never what a mount point means, and
/// resolving it would let `"/mem/.."` name the root.
///
/// The root itself is excluded because the workspace's own files are there and unmounting them
/// would leave the tree with nowhere to write — an empty workspace is the one thing a launch
/// guarantees, and a connector is not the thing that gets to take it away.
pub fn normalize_mount_path(path: &str) -> Result<String> {
    let mut segments = Vec::new();
    for segment in path.trim().split('/') {
        match segment {
            "" | "." => continue,
            ".." => return Err(EngineError::Invalid("경로에 '..' 을 쓸 수 없습니다".into())),
            s => segments.push(s),
        }
    }
    if segments.is_empty() {
        return Err(EngineError::Invalid(
            "연결할 경로를 입력해 주세요 (예: /notion)".into(),
        ));
    }
    Ok(format!("/{}", segments.join("/")))
}

/// What a config looks like in the sidebar: its kind, the line under the name, and whether the
/// store implements the write half.
pub fn describe(config: &MountConfig) -> (MountKind, String, bool) {
    match config {
        MountConfig::Root => (MountKind::Root, "워크스페이스 파일".into(), true),
        MountConfig::Local { host_root } => {
            // Read-write: `PassthroughFs` serves the host's own permissions, so what this
            // promises is that the store implements the write half — not that every file under
            // it is writable.
            (MountKind::Local, host_root.display().to_string(), true)
        }
        MountConfig::Notion { .. } => (
            MountKind::Notion,
            "Notion workspace · 읽기 전용".into(),
            false,
        ),
        MountConfig::S3 {
            bucket,
            region,
            endpoint,
            ..
        } => (
            MountKind::S3,
            match endpoint {
                Some(e) => format!("s3://{bucket} · {e}"),
                None => format!("s3://{bucket} · {region}"),
            },
            false,
        ),
    }
}

/// The store `config` names, once one request has proved it answers.
pub async fn build_and_probe(config: &MountConfig) -> Result<Arc<dyn FileSystem>> {
    match config {
        MountConfig::Root => Err(EngineError::Invalid(
            "루트는 커넥터로 추가할 수 없습니다".into(),
        )),
        MountConfig::Local { host_root } => {
            let meta = tokio::fs::metadata(host_root)
                .await
                .map_err(|e| EngineError::Invalid(format!("{}: {e}", host_root.display())))?;
            if !meta.is_dir() {
                return Err(EngineError::Invalid("디렉터리를 선택해 주세요".into()));
            }
            Ok(Arc::new(PassthroughFs::new(host_root.clone())))
        }
        MountConfig::Notion { api_key } => {
            let api_key = api_key.trim().to_string();
            if api_key.is_empty() {
                return Err(EngineError::Invalid(
                    "Notion 통합 토큰을 입력해 주세요".into(),
                ));
            }
            let store = NotionFs::new(&NotionConfig { api_key })?;
            // The confirming request. A listing of the root is what the tree would ask for
            // first anyway, so a token that cannot do it is a connection worth refusing now.
            tokio::time::timeout(PROBE_TIMEOUT, store.list(Path::new("")))
                .await
                .map_err(|_| EngineError::Invalid("Notion 응답이 없습니다 (15초)".into()))?
                .map_err(|e| EngineError::Invalid(format!("Notion에 연결하지 못했습니다: {e}")))?;
            Ok(Arc::new(store))
        }
        MountConfig::S3 {
            bucket,
            region,
            access_key_id,
            secret_access_key,
            endpoint,
            key_prefix,
        } => {
            let cfg = S3Config {
                bucket: bucket.trim().to_string(),
                region: region.trim().to_string(),
                access_key_id: access_key_id.trim().to_string(),
                secret_access_key: secret_access_key.clone(),
                endpoint: endpoint
                    .clone()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty()),
                key_prefix: key_prefix
                    .clone()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty()),
            };
            if cfg.bucket.is_empty() {
                return Err(EngineError::Invalid("버킷 이름을 입력해 주세요".into()));
            }
            let store = S3Fs::new(&cfg)?;
            tokio::time::timeout(PROBE_TIMEOUT, store.check_reachable())
                .await
                .map_err(|_| EngineError::Invalid("S3 응답이 없습니다 (15초)".into()))?
                .map_err(|e| EngineError::Invalid(format!("버킷에 연결하지 못했습니다: {e}")))?;
            Ok(Arc::new(store))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mount_paths_are_normalized() {
        assert_eq!(normalize_mount_path(" notion/ ").unwrap(), "/notion");
        assert_eq!(normalize_mount_path("/a/b").unwrap(), "/a/b");
        // The spellings `ContextFs` collapses on its own: the row has to collapse them too, or
        // the list names one path and the tree is keyed by another.
        assert_eq!(normalize_mount_path("/mem/.").unwrap(), "/mem");
        assert_eq!(normalize_mount_path("/a//b").unwrap(), "/a/b");
        assert!(normalize_mount_path("/").is_err());
        assert!(normalize_mount_path("/.").is_err(), "the root, spelled out");
        assert!(normalize_mount_path("/../x").is_err());
    }

    #[tokio::test]
    async fn local_connector_needs_a_directory() {
        let f = tempfile::NamedTempFile::new().unwrap();
        assert!(
            build_and_probe(&MountConfig::Local {
                host_root: f.path().to_path_buf()
            })
            .await
            .is_err()
        );
        let d = tempfile::tempdir().unwrap();
        assert!(
            build_and_probe(&MountConfig::Local {
                host_root: d.path().to_path_buf()
            })
            .await
            .is_ok()
        );
    }
}
