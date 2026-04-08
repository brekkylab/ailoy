use std::path::PathBuf;

use anyhow::{Context as _, Result};

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
/// binary is executable or the correct version.  Step 3 returns the
/// *expected* path regardless of whether the file is there yet — callers
/// are responsible for downloading when it is absent.
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

/// Platform-specific binary name.
fn uv_binary_name() -> &'static str {
    if cfg!(target_os = "windows") { "uv.exe" } else { "uv" }
}

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
}
