use std::sync::atomic::{AtomicU8, Ordering};

use anyhow::Context as _;
use cortex::console::{Error, ExecResult};
use tokio::sync::Notify;

use crate::tool::Console;

/// All ailoy-managed files live under `$XDG_CACHE_HOME/ailoy` (default `~/.cache/ailoy`):
///   - uv binary : `$AILOY_CACHE/bin/uv`  (symlink if system uv exists, else downloaded)
///   - Python venv: `$AILOY_CACHE/venv`
///
/// One-time setup (all steps idempotent):
///   1. Ensure python3 + ca-certificates (apt-get fallback for bare ubuntu images).
///   2. Resolve uv: symlink from PATH, or download the platform binary via Python urllib.
///   3. Create the venv via uv.
const SETUP_CMD: &str = r#"set -e
AILOY_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/ailoy"
mkdir -p "$AILOY_CACHE/bin"
command -v python3 >/dev/null 2>&1 \
    || { DEBIAN_FRONTEND=noninteractive apt-get update -qq 2>&1 \
         && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 ca-certificates 2>&1; } \
    || { echo "python3 not found and apt-get install failed"; exit 1; }
if ! test -x "$AILOY_CACHE/bin/uv"; then
    if command -v uv >/dev/null 2>&1; then
        ln -sf "$(command -v uv)" "$AILOY_CACHE/bin/uv"
    else
        python3 - <<'PYEOF'
import urllib.request, os, tarfile, io, platform
cache_home = os.environ.get('XDG_CACHE_HOME', os.path.join(os.path.expanduser('~'), '.cache'))
bin_dir = os.path.join(cache_home, 'ailoy', 'bin')
os.makedirs(bin_dir, exist_ok=True)
machine = platform.machine()
arch = 'aarch64' if machine in ('arm64', 'aarch64') else machine
system = platform.system().lower()
target = f'uv-{arch}-apple-darwin' if system == 'darwin' else f'uv-{arch}-unknown-linux-gnu'
url = f'https://github.com/astral-sh/uv/releases/latest/download/{target}.tar.gz'
req = urllib.request.Request(url, headers={'User-Agent': 'curl/8.0.0'})
with urllib.request.urlopen(req) as r:
    data = r.read()
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tf:
    for m in tf.getmembers():
        if m.name.split('/')[-1] == 'uv' and m.isfile():
            f = tf.extractfile(m)
            if f:
                p = os.path.join(bin_dir, 'uv')
                with open(p, 'wb') as out:
                    out.write(f.read())
                os.chmod(p, 0o755)
                break
PYEOF
    fi
fi
[ -d "$AILOY_CACHE/venv" ] \
    || "$AILOY_CACHE/bin/uv" venv "$AILOY_CACHE/venv""#;

const SETUP_TIMEOUT_MS: u64 = 300_000;

/// Python interpreter inside the ailoy venv. Resolved by sh at runtime via XDG_CACHE_HOME.
const AILOY_PYTHON: &str = r#""${XDG_CACHE_HOME:-$HOME/.cache}/ailoy/venv/bin/python3""#;

/// uv pip install targeting the ailoy venv. Both paths resolved by sh at runtime.
const AILOY_PIP_INSTALL: &str = r#""${XDG_CACHE_HOME:-$HOME/.cache}/ailoy/bin/uv" pip install --python "${XDG_CACHE_HOME:-$HOME/.cache}/ailoy/venv/bin/python3""#;

/// `sh -c cmd` as the argv cortex takes — it consults no shell of its own.
fn sh(cmd: &str) -> [String; 3] {
    ["sh".to_string(), "-c".to_string(), cmd.to_string()]
}

async fn run_setup(console: &mut Console) -> anyhow::Result<()> {
    // Milliseconds: cortex counts in them, and this is the one call here long
    // enough that the units are worth reading twice.
    let r = match console.exec(sh(SETUP_CMD), Some(SETUP_TIMEOUT_MS)).await {
        Ok(r) => r,
        // A killed execution has no result to inspect, so it arrives as a refusal
        // rather than as a flag on one.
        Err(e) if e.code() == Some(Error::TIMED_OUT) => {
            anyhow::bail!(
                "Python runtime setup timed out after {}s",
                SETUP_TIMEOUT_MS / 1000
            );
        }
        Err(e) => return Err(e).context("Python runtime setup failed"),
    };

    if r.code != 0 {
        let stdout = String::from_utf8_lossy(&r.stdout);
        let stderr = String::from_utf8_lossy(&r.stderr);
        anyhow::bail!(
            "Python runtime setup failed (exit {}): {}",
            r.code,
            if stderr.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            }
        );
    }
    Ok(())
}

const UNINITIALIZED: u8 = 0;
const INITIALIZING: u8 = 1;
const INITIALIZED: u8 = 2;

pub struct PythonReplRunner {
    state: AtomicU8,
    notify: Notify,
}

impl PythonReplRunner {
    pub fn new() -> Self {
        Self {
            state: AtomicU8::new(UNINITIALIZED),
            notify: Notify::new(),
        }
    }

    async fn ensure_ready(&self, console: &mut Console) -> anyhow::Result<()> {
        loop {
            // Create notified future before the CAS to avoid missing a wakeup.
            let notified = self.notify.notified();
            match self.state.compare_exchange(
                UNINITIALIZED,
                INITIALIZING,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let result = run_setup(console).await;
                    self.state.store(
                        if result.is_ok() {
                            INITIALIZED
                        } else {
                            UNINITIALIZED
                        },
                        Ordering::Release,
                    );
                    self.notify.notify_waiters();
                    return result;
                }
                Err(INITIALIZING) => notified.await,
                Err(INITIALIZED) => return Ok(()),
                Err(_) => unreachable!(),
            }
        }
    }

    /// Install pip packages into the ailoy venv via uv.
    pub async fn install_packages(
        &self,
        console: &mut Console,
        packages: &[&str],
    ) -> anyhow::Result<ExecResult> {
        if packages.is_empty() {
            return Ok(ExecResult::default());
        }
        self.ensure_ready(console).await?;
        let quoted: Vec<String> = packages.iter().map(|p| format!("'{p}'")).collect();
        let cmd = format!("{AILOY_PIP_INSTALL} {}", quoted.join(" "));
        Ok(console.exec(sh(&cmd), None).await?)
    }

    /// Execute a Python script with optional env vars.
    pub async fn run(
        &self,
        console: &mut Console,
        source: &str,
        env: &[(&str, &str)],
    ) -> anyhow::Result<ExecResult> {
        self.run_with_timeout(console, source, env, 0).await
    }

    /// Like [`run`] but with a per-execution timeout (`0` = no timeout).
    pub async fn run_with_timeout(
        &self,
        console: &mut Console,
        source: &str,
        env: &[(&str, &str)],
        timeout_secs: u64,
    ) -> anyhow::Result<ExecResult> {
        self.ensure_ready(console).await?;

        let script_path = format!("/tmp/__ailoy_{}.py", uuid::Uuid::new_v4());

        // Under `/tmp`, so nothing above the file needs creating — cortex's `write`
        // makes the file and not the path.
        console
            .write(&script_path, source.as_bytes().to_vec(), None)
            .await
            .context("failed to write script")?;

        let env_prefix = env
            .iter()
            .map(|(k, v)| format!("{k}='{v}'"))
            .collect::<Vec<_>>()
            .join(" ");
        let cmd = if env_prefix.is_empty() {
            format!("{AILOY_PYTHON} {script_path}")
        } else {
            format!("{env_prefix} {AILOY_PYTHON} {script_path}")
        };

        // The tool's parameter is seconds; cortex's bound is milliseconds. `0` is
        // "no timeout", which for cortex is `None`.
        let timeout_ms = (timeout_secs > 0).then(|| timeout_secs.saturating_mul(1_000));
        let result = console.exec(sh(&cmd), timeout_ms).await;

        // Best effort, and deliberately after the result is in hand: the script is
        // gone either way, and a cleanup failure should not mask what ran.
        let _ = console
            .exec(sh(&format!("rm -f {script_path}")), None)
            .await;

        result.context("script execution error")
    }
}
