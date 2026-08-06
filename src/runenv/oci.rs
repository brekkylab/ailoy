//! Minimal OCI Distribution v2 image pull: an image reference (`python:3.12-slim`)
//! becomes a read-only **EROFS image** the sandbox mounts as the overlay lower.
//!
//! We handle the registry protocol (Docker Hub token auth, manifest-list host-arch
//! selection); the heavy lifting — decompressing each layer tar, applying OCI
//! whiteouts, merging the layers, and encoding the result as EROFS — is reused
//! from `microsandbox-image` (`ingest_compressed_tar` + `FileTree::merge_layer` +
//! `write_erofs`), the same crate that formats the ext4 upper. Non-Hub registries
//! are still an explicit error.

use std::io;
use std::path::Path;

use microsandbox_image::erofs::write_erofs;
use microsandbox_image::tar::{Compression, ingest_compressed_tar};
use microsandbox_image::tree::{FileTree, ResourceLimits};
use serde_json::Value;

const DOCKER_REGISTRY: &str = "registry-1.docker.io";
const DOCKER_AUTH: &str = "https://auth.docker.io/token";
const DOCKER_AUTH_SERVICE: &str = "registry.docker.io";

// Manifest media types we ask for and recognise.
const ACCEPT: &str = "application/vnd.oci.image.index.v1+json, \
     application/vnd.docker.distribution.manifest.list.v2+json, \
     application/vnd.oci.image.manifest.v1+json, \
     application/vnd.docker.distribution.manifest.v2+json";

/// A parsed image reference: `registry/repo:tag`.
#[derive(Debug)]
struct ImageRef {
    registry: String,
    repo: String,
    /// A tag or a `sha256:…` digest — both address `/manifests/<reference>`.
    reference: String,
}

/// Split `python:3.12-slim`, `ghcr.io/org/img@sha256:…`, etc. into its parts,
/// applying Docker's defaults (Hub registry, `library/` namespace, `latest`).
fn parse_ref(input: &str) -> ImageRef {
    // A leading component is a registry only if it looks like a host: has a dot
    // or a port colon, or is `localhost`. Otherwise the whole thing is a Hub repo.
    let (registry, remainder) = match input.split_once('/') {
        Some((head, rest))
            if head == "localhost" || head.contains('.') || head.contains(':') =>
        {
            (head.to_string(), rest.to_string())
        }
        _ => (DOCKER_REGISTRY.to_string(), input.to_string()),
    };

    // A digest (`@sha256:…`) wins over a tag; else split the last `:` that is not
    // part of a registry port (there is no `/` left in `remainder` after the repo).
    let (repo, reference) = if let Some((r, d)) = remainder.split_once('@') {
        (r.to_string(), d.to_string())
    } else if let Some((r, t)) = remainder.rsplit_once(':') {
        (r.to_string(), t.to_string())
    } else {
        (remainder.clone(), "latest".to_string())
    };

    // Hub shorthand: a bare name is in the `library` namespace.
    let repo = if registry == DOCKER_REGISTRY && !repo.contains('/') {
        format!("library/{repo}")
    } else {
        repo
    };

    ImageRef {
        registry,
        repo,
        reference,
    }
}

/// The OCI architecture string for the host (`aarch64` -> `arm64`).
fn host_arch() -> &'static str {
    match std::env::consts::ARCH {
        "aarch64" => "arm64",
        "x86_64" => "amd64",
        other => other,
    }
}

fn other<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

/// Fetch a Docker Hub pull token for `repo`. Anonymous pulls still need one; a
/// non-Hub registry that needs auth is out of scope for this spike.
async fn docker_token(client: &reqwest::Client, repo: &str) -> io::Result<String> {
    let resp = client
        .get(DOCKER_AUTH)
        .query(&[
            ("service", DOCKER_AUTH_SERVICE),
            ("scope", &format!("repository:{repo}:pull")),
        ])
        .send()
        .await
        .map_err(other)?
        .error_for_status()
        .map_err(other)?;
    let v: Value = resp.json().await.map_err(other)?;
    v.get("token")
        .and_then(|t| t.as_str())
        .map(str::to_string)
        .ok_or_else(|| other("docker auth response had no token"))
}

/// GET `/v2/<repo>/manifests/<reference>` as parsed JSON.
async fn get_manifest(
    client: &reqwest::Client,
    img: &ImageRef,
    token: &str,
    reference: &str,
) -> io::Result<Value> {
    let url = format!("https://{}/v2/{}/manifests/{reference}", img.registry, img.repo);
    let resp = client
        .get(&url)
        .bearer_auth(token)
        .header(reqwest::header::ACCEPT, ACCEPT)
        .send()
        .await
        .map_err(other)?
        .error_for_status()
        .map_err(other)?;
    resp.json().await.map_err(other)
}

/// Resolve a manifest that may be an index/list down to a single image manifest
/// for the host platform.
async fn resolve_image_manifest(
    client: &reqwest::Client,
    img: &ImageRef,
    token: &str,
) -> io::Result<Value> {
    let manifest = get_manifest(client, img, token, &img.reference).await?;

    // An index/list carries per-platform child manifests; an image manifest
    // carries `layers` directly.
    if let Some(children) = manifest.get("manifests").and_then(|m| m.as_array()) {
        let want = host_arch();
        let digest = children
            .iter()
            .find(|m| {
                let p = m.get("platform");
                let os = p.and_then(|p| p.get("os")).and_then(|o| o.as_str());
                let arch = p
                    .and_then(|p| p.get("architecture"))
                    .and_then(|a| a.as_str());
                os == Some("linux") && arch == Some(want)
            })
            .and_then(|m| m.get("digest"))
            .and_then(|d| d.as_str())
            .ok_or_else(|| other(format!("no linux/{want} image in the manifest list")))?;
        return get_manifest(client, img, token, digest).await;
    }

    Ok(manifest)
}

/// Download one layer blob and merge its tar into `tree`. `ingest_compressed_tar`
/// decompresses (gzip/zstd), applies OCI whiteouts, and skips unsupported nodes.
async fn merge_layer(
    client: &reqwest::Client,
    img: &ImageRef,
    token: &str,
    layer: &Value,
    tree: &mut FileTree,
    limits: &ResourceLimits,
) -> io::Result<()> {
    let media = layer.get("mediaType").and_then(|m| m.as_str()).unwrap_or("");
    let compression = if media.contains("zstd") {
        Compression::Zstd
    } else if media.contains("gzip") {
        Compression::Gzip
    } else {
        // Bare `application/vnd.oci.image.layer.v1.tar` (uncompressed).
        Compression::None
    };
    let digest = layer
        .get("digest")
        .and_then(|d| d.as_str())
        .ok_or_else(|| other("layer had no digest"))?;

    let url = format!("https://{}/v2/{}/blobs/{digest}", img.registry, img.repo);
    let bytes = client
        .get(&url)
        .bearer_auth(token)
        .send()
        .await
        .map_err(other)?
        .error_for_status()
        .map_err(other)?
        .bytes()
        .await
        .map_err(other)?;

    let res = ingest_compressed_tar(&bytes[..], compression, limits, None)
        .await
        .map_err(|e| other(format!("ingest layer {digest}: {e:?}")))?;
    tree.merge_layer(res.tree);
    Ok(())
}

/// Pull `reference` and write its merged rootfs as a read-only EROFS image at
/// `dest`. Idempotent: an existing `dest` is reused, so a caller can key it by
/// reference/digest as a cache.
pub async fn pull_erofs(reference: &str, dest: &Path) -> io::Result<()> {
    if dest.exists() {
        return Ok(());
    }

    let img = parse_ref(reference);
    if img.registry != DOCKER_REGISTRY {
        return Err(other(format!(
            "only Docker Hub is supported for now (got registry {})",
            img.registry
        )));
    }

    let client = reqwest::Client::builder()
        .user_agent("ailoy-oci")
        .build()
        .map_err(other)?;

    let token = docker_token(&client, &img.repo).await?;
    let manifest = resolve_image_manifest(&client, &img, &token).await?;

    let layers = manifest
        .get("layers")
        .and_then(|l| l.as_array())
        .ok_or_else(|| other("image manifest had no layers"))?
        .clone();

    let limits = ResourceLimits::default();
    let mut tree = FileTree::new();
    for layer in &layers {
        merge_layer(&client, &img, &token, layer, &mut tree, &limits).await?;
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Write to a temp path then rename, so a concurrent puller never sees a
    // half-written EROFS at `dest`. `write_erofs` is sync and writes MBs.
    let tmp = dest.with_extension("erofs.tmp");
    let tmp2 = tmp.clone();
    tokio::task::spawn_blocking(move || write_erofs(&tree, &tmp2).map_err(|e| other(format!("{e:?}"))))
        .await
        .map_err(other)??;
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_docker_hub_shorthand() {
        let r = parse_ref("python:3.12-slim");
        assert_eq!(r.registry, DOCKER_REGISTRY);
        assert_eq!(r.repo, "library/python");
        assert_eq!(r.reference, "3.12-slim");
    }

    #[test]
    fn parses_namespaced_and_defaults_tag() {
        let r = parse_ref("nurikim/agent");
        assert_eq!(r.repo, "nurikim/agent");
        assert_eq!(r.reference, "latest");
    }

    #[test]
    fn parses_registry_with_port_and_digest() {
        let r = parse_ref("localhost:5000/foo@sha256:abcd");
        assert_eq!(r.registry, "localhost:5000");
        assert_eq!(r.repo, "foo");
        assert_eq!(r.reference, "sha256:abcd");
    }

    /// Pull a real public image and confirm we produced a non-trivial EROFS.
    /// Ignored (network + tens of MB).
    #[tokio::test]
    #[ignore = "network: pulls python:3.12-slim from Docker Hub"]
    async fn pulls_python_rootfs() {
        let dir = tempfile::tempdir().unwrap();
        let erofs = dir.path().join("python.erofs");
        pull_erofs("python:3.12-slim", &erofs)
            .await
            .expect("pull python:3.12-slim");
        let size = std::fs::metadata(&erofs).map(|m| m.len()).unwrap_or(0);
        assert!(size > 4 << 20, "expected a multi-MB EROFS, got {size} bytes");
    }
}
