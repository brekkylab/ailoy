use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};

/// Download URL template for uv GitHub releases.
const UV_RELEASE_BASE: &str = "https://github.com/astral-sh/uv/releases/latest/download";

/// Returns the path to the `uv` binary to use.
///
/// Resolution order:
/// 1. `UV` environment variable (override for testing / CI)
/// 2. `uv` found anywhere on `PATH`
/// 3. `<cache_dir>/ailoy/bin/uv[.exe]` — the managed install location
///
/// The platform cache directory follows XDG / OS conventions via [`dirs::cache_dir`]:
/// - Linux:   `$XDG_CACHE_HOME/ailoy/bin/uv`  (defaults to `~/.cache/ailoy/bin/uv`)
/// - macOS:   `~/Library/Caches/ailoy/bin/uv`
/// - Windows: `%LOCALAPPDATA%\ailoy\bin\uv.exe`
///
/// Steps 1 and 2 only check that the path exists; they do not verify the
/// binary is executable or the correct version.
pub fn resolve_uv_path() -> Result<PathBuf> {
    // 1. Explicit override (useful in tests and CI).
    if let Ok(path) = std::env::var("UV") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Ok(p);
        }
    }

    // 2. Walk PATH.
    let uv_name = uv_binary_name();
    if let Ok(path) = which::which(uv_name) {
        return Ok(path);
    }

    // 3. Managed install location.
    Ok(managed_uv_path()?)
}

/// Ensure `uv` is available, downloading it if necessary.
///
/// Calls [`resolve_uv_path`] first.  If the resolved path does not exist on
/// disk (i.e. it fell through to the managed location and the binary has not
/// been downloaded yet), downloads the latest `uv` release from GitHub.
pub async fn ensure_uv() -> Result<PathBuf> {
    let path = resolve_uv_path()?;
    if path.exists() {
        return Ok(path);
    }
    log::info!("uv not found — downloading to {}", path.display());
    download_uv(&path).await?;
    Ok(path)
}

/// Path where ailoy will place its own `uv` download.
///
/// Returns `<cache_dir>/ailoy/bin/uv[.exe]` using the platform cache directory:
/// - Linux:   `$XDG_CACHE_HOME`  (default `~/.cache`)
/// - macOS:   `~/Library/Caches`
/// - Windows: `%LOCALAPPDATA%`
pub fn managed_uv_path() -> Result<PathBuf> {
    let cache = dirs::cache_dir().context("cannot determine cache directory")?;
    Ok(cache.join("ailoy").join("bin").join(uv_binary_name()))
}

// ── download logic ────────────────────────────────────────────────────────────

/// Download the latest `uv` binary for the current platform and place it at
/// `dest`.  Creates parent directories as needed.
async fn download_uv(dest: &Path) -> Result<()> {
    let target = platform_target()?;
    let archive_name = archive_name(target);
    let url = format!("{UV_RELEASE_BASE}/{archive_name}");

    log::info!("Downloading {}", url);
    let response = reqwest::get(&url).await.context("failed to download uv")?;

    if !response.status().is_success() {
        bail!(
            "failed to download uv: HTTP {} from {}",
            response.status(),
            url,
        );
    }

    let bytes = response
        .bytes()
        .await
        .context("failed to read uv archive body")?;

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).context("failed to create directory for uv binary")?;
    }

    extract_uv_binary(&bytes, dest)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(dest, std::fs::Permissions::from_mode(0o755))
            .context("failed to set executable permission on uv")?;
    }

    log::info!("uv installed at {}", dest.display());
    Ok(())
}

/// Extract the `uv` binary from the downloaded archive into `dest`.
///
/// On Unix the archive is `.tar.gz` (extracted via flate2 + tar);
/// on Windows it is `.zip` (extracted via the zip crate).
fn extract_uv_binary(data: &[u8], dest: &Path) -> Result<()> {
    let want = uv_binary_name();

    #[cfg(not(target_os = "windows"))]
    {
        use flate2::read::GzDecoder;
        use tar::Archive;

        let decoder = GzDecoder::new(data);
        let mut archive = Archive::new(decoder);

        for entry in archive.entries().context("failed to read tar entries")? {
            let mut entry = entry.context("failed to read tar entry")?;
            let path = entry.path().context("failed to read entry path")?;

            // Match any entry whose file-name component is exactly `uv`.
            // The archive layout is typically `uv-{target}/uv`.
            if path.file_name().and_then(|n| n.to_str()) == Some(want) {
                entry
                    .unpack(dest)
                    .context("failed to extract uv binary from archive")?;
                return Ok(());
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        use std::io::Read as _;

        let cursor = std::io::Cursor::new(data);
        let mut archive =
            zip::ZipArchive::new(cursor).context("failed to open zip archive")?;

        for i in 0..archive.len() {
            let mut file = archive.by_index(i).context("failed to read zip entry")?;
            let path = match file.enclosed_name() {
                Some(p) => p.to_path_buf(),
                None => continue,
            };

            // Match any entry whose file-name component is exactly `uv.exe`.
            // The archive layout is typically `uv-{target}/uv.exe`.
            if path.file_name().and_then(|n| n.to_str()) == Some(want) {
                let mut out = std::fs::File::create(dest)
                    .context("failed to create uv binary file")?;
                std::io::copy(&mut file, &mut out)
                    .context("failed to write uv binary from zip")?;
                return Ok(());
            }
        }
    }

    bail!(
        "uv binary ({}) not found in archive — is the release format correct?",
        want,
    )
}

// ── helpers ────────────────────────────────────────────────────────────────────

/// Platform-specific binary name.
fn uv_binary_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "uv.exe"
    } else {
        "uv"
    }
}

/// Archive file name for the given target triple.
fn archive_name(target: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("uv-{target}.zip")
    } else {
        format!("uv-{target}.tar.gz")
    }
}

/// Determine the platform target triple used by uv release artifacts.
///
/// Matches ailoy's CI/supported platforms:
///   - Linux x86_64 glibc  (ubuntu-latest)
///   - macOS aarch64        (macos-15, Apple Silicon)
///   - Windows x86_64       (windows-2022)
fn platform_target() -> Result<&'static str> {
    // macOS — ailoy CI: macos-15 (Apple Silicon)
    if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        Ok("aarch64-apple-darwin")
    } else if cfg!(target_os = "macos") && cfg!(target_arch = "x86_64") {
        Ok("x86_64-apple-darwin")
    }
    // Linux — ailoy CI: ubuntu-latest (x86_64 glibc)
    else if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        Ok("x86_64-unknown-linux-gnu")
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "aarch64") {
        Ok("aarch64-unknown-linux-gnu")
    }
    // Windows — ailoy CI: windows-2022 (x86_64)
    else if cfg!(target_os = "windows") && cfg!(target_arch = "x86_64") {
        Ok("x86_64-pc-windows-msvc")
    } else if cfg!(target_os = "windows") && cfg!(target_arch = "aarch64") {
        Ok("aarch64-pc-windows-msvc")
    } else {
        bail!("unsupported platform for uv download")
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_managed_uv_path_is_under_cache_dir() {
        let path = managed_uv_path().unwrap();
        let cache = dirs::cache_dir().unwrap();
        assert!(
            path.starts_with(&cache),
            "managed uv path {:?} should be under cache dir {:?}",
            path,
            cache,
        );
        // Must end with the platform binary name.
        assert_eq!(path.file_name().unwrap(), uv_binary_name());
    }

    #[test]
    fn test_managed_uv_path_structure() {
        let path = managed_uv_path().unwrap();
        let cache = dirs::cache_dir().unwrap();
        // Expected: <cache_dir>/ailoy/bin/uv[.exe]
        let expected = cache.join("ailoy").join("bin").join(uv_binary_name());
        assert_eq!(path, expected);
    }

    #[test]
    fn test_resolve_uv_path_env_override() {
        // Point UV at an existing file (use the test binary itself).
        let current_exe = std::env::current_exe().unwrap();
        // SAFETY: test-only env mutation — tests run in separate processes.
        unsafe { std::env::set_var("UV", current_exe.to_str().unwrap()) };
        let result = resolve_uv_path().unwrap();
        assert_eq!(result, current_exe);
        unsafe { std::env::remove_var("UV") };
    }

    #[test]
    fn test_resolve_uv_path_env_override_missing_file_falls_through() {
        unsafe { std::env::set_var("UV", "/this/path/definitely/does/not/exist/uv") };
        // Should not error — must fall through to managed path.
        let result = resolve_uv_path();
        assert!(result.is_ok(), "should return managed path as fallback");
        unsafe { std::env::remove_var("UV") };
    }

    #[test]
    fn test_uv_binary_name_platform() {
        let name = uv_binary_name();
        if cfg!(target_os = "windows") {
            assert_eq!(name, "uv.exe");
        } else {
            assert_eq!(name, "uv");
        }
    }

    #[test]
    fn test_platform_target_is_known() {
        let target = platform_target().unwrap();
        assert!(
            target.contains("apple") || target.contains("linux") || target.contains("windows"),
            "unexpected target: {}",
            target,
        );
    }

    #[test]
    fn test_archive_name_format() {
        let target = platform_target().unwrap();
        let name = archive_name(target);
        if cfg!(target_os = "windows") {
            assert!(name.ends_with(".zip"));
        } else {
            assert!(name.ends_with(".tar.gz"));
        }
        assert!(name.contains(target));
    }

    #[tokio::test]
    #[ignore = "requires network, downloads ~25MB"]
    async fn test_download_uv_and_verify() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("bin").join(uv_binary_name());

        download_uv(&dest).await.unwrap();

        assert!(dest.exists(), "uv binary should exist at {:?}", dest);
        assert!(
            dest.metadata().unwrap().len() > 1_000_000,
            "uv binary should be > 1MB",
        );

        // Verify it's actually executable by running `uv --version`.
        let output = std::process::Command::new(&dest)
            .arg("--version")
            .output()
            .unwrap();
        assert!(output.status.success());
        let version = String::from_utf8_lossy(&output.stdout);
        assert!(version.starts_with("uv "), "got: {}", version);
    }
}
