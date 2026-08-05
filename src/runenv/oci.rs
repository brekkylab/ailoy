//! Minimal OCI Distribution v2 image pull: an image reference (`python:3.12-slim`)
//! becomes an unpacked rootfs *directory*, which the sandbox later serves over
//! virtio-fs exactly like the provisioned Alpine tree.
//!
//! Deliberately small — passthrough means there is no block image, overlay, or
//! content-addressed store to build, only "fetch the layers and stack them".
//! Scope of this spike: public registries (anonymous or Docker Hub token auth),
//! gzip layers, the host architecture picked out of a manifest list. zstd layers
//! and authenticated/private registries are explicit `Unsupported` errors rather
//! than silent wrong behaviour.

use std::io::{self, Cursor};
use std::path::Path;

use flate2::read::GzDecoder;
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

/// Download one layer blob and stack it onto `dest`, honouring OCI whiteouts.
async fn apply_layer(
    client: &reqwest::Client,
    img: &ImageRef,
    token: &str,
    layer: &Value,
    dest: &Path,
) -> io::Result<()> {
    let media = layer.get("mediaType").and_then(|m| m.as_str()).unwrap_or("");
    if media.contains("zstd") {
        return Err(other("zstd layers are not supported by this spike yet"));
    }
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

    // Unpack synchronously: gunzip + tar over an in-memory blob (layers are tens
    // of MB — fine for a spike; stream later if it matters).
    let dest = dest.to_path_buf();
    tokio::task::spawn_blocking(move || unpack_tar_gz(&bytes, &dest))
        .await
        .map_err(other)?
}

/// Stack a gzipped tar onto `dest`, applying `.wh.` whiteouts and skipping device
/// nodes (which would need privileges we do not have).
fn unpack_tar_gz(blob: &[u8], dest: &Path) -> io::Result<()> {
    let mut archive = tar::Archive::new(GzDecoder::new(Cursor::new(blob)));
    archive.set_preserve_permissions(true);
    archive.set_overwrite(true);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();

        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            // `.wh..wh..opq`: everything already in this directory from lower
            // layers is hidden.
            if name == ".wh..wh..opq" {
                if let Some(parent) = path.parent() {
                    clear_dir(&dest.join(parent));
                }
                continue;
            }
            // `.wh.<name>`: delete `<name>` from the merged tree.
            if let Some(victim) = name.strip_prefix(".wh.") {
                let target = match path.parent() {
                    Some(p) => dest.join(p).join(victim),
                    None => dest.join(victim),
                };
                remove_any(&target);
                continue;
            }
        }

        let kind = entry.header().entry_type();
        if kind.is_character_special() || kind.is_block_special() || kind.is_fifo() {
            continue;
        }
        entry.unpack_in(dest)?;
    }
    Ok(())
}

/// Remove a path whether it is a file, symlink, or directory (best-effort).
fn remove_any(p: &Path) {
    if p.is_dir() && !p.is_symlink() {
        let _ = std::fs::remove_dir_all(p);
    } else {
        let _ = std::fs::remove_file(p);
    }
}

/// Empty a directory's contents without removing the directory itself.
fn clear_dir(dir: &Path) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            remove_any(&e.path());
        }
    }
}

/// Pull `reference` into `dest`, leaving an unpacked rootfs tree there.
///
/// Idempotent by directory: if `dest` already holds an unpacked image (marked by
/// a `.ailoy-oci-done` sentinel) it is reused, so a caller can point this at a
/// cache path keyed by digest/reference.
pub async fn pull(reference: &str, dest: &Path) -> io::Result<()> {
    let done = dest.join(".ailoy-oci-done");
    if done.exists() {
        return Ok(());
    }

    let img = parse_ref(reference);
    if img.registry != DOCKER_REGISTRY {
        // Anonymous non-Hub pulls sometimes work, but the token flow here is
        // Hub-specific; keep the spike honest about it.
        return Err(other(format!(
            "only Docker Hub is supported by this spike (got registry {})",
            img.registry
        )));
    }

    let client = reqwest::Client::builder()
        .user_agent("ailoy-oci-spike")
        .build()
        .map_err(other)?;

    let token = docker_token(&client, &img.repo).await?;
    let manifest = resolve_image_manifest(&client, &img, &token).await?;

    let layers = manifest
        .get("layers")
        .and_then(|l| l.as_array())
        .ok_or_else(|| other("image manifest had no layers"))?
        .clone();

    std::fs::create_dir_all(dest)?;
    for layer in &layers {
        apply_layer(&client, &img, &token, layer, dest).await?;
    }

    std::fs::write(&done, reference.as_bytes())?;
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

    /// The spike's proof: pull a real public image into a temp dir and confirm we
    /// got a usable Python rootfs. Ignored (network + tens of MB).
    #[tokio::test]
    #[ignore = "network: pulls python:3.12-slim from Docker Hub"]
    async fn pulls_python_rootfs() {
        let dir = tempfile::tempdir().unwrap();
        pull("python:3.12-slim", dir.path())
            .await
            .expect("pull python:3.12-slim");
        let py = dir.path().join("usr/local/bin/python3.12");
        assert!(
            py.exists(),
            "expected python3.12 in the pulled rootfs at {}",
            py.display()
        );
    }
}
