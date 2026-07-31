use std::sync::atomic::{AtomicU8, Ordering};

use anyhow::Context as _;
use tokio::sync::Notify;

use crate::runenv::{Console, ExecResult};

/// All ailoy-managed files live under `$XDG_CACHE_HOME/ailoy` (default `~/.cache/ailoy`):
///   - uv binary : `$AILOY_CACHE/bin/uv`  (symlink if system uv exists, else downloaded)
///   - Python venv: `$AILOY_CACHE/venv`
///
/// One-time setup (all steps idempotent):
///   1. Ensure python3 + ca-certificates (apt-get fallback for bare ubuntu images).
///   2. Resolve uv: symlink from PATH, or download the platform binary via Python urllib.
///   3. Create the venv via uv.
///
/// # Network reach this needs
///
/// Every step above fetches something, so on a [`Console`] backed by a sandbox
/// this setup runs under the guest network policy. The default posture
/// (`SandboxNetwork::HostOnly` with no host ports) permits only gateway DNS, and
/// setup fails there on its first step — `apt-get` cannot reach a mirror, so
/// `python3` never installs.
///
/// A caller that binds `python_repl` to a sandbox has to widen the posture. The
/// narrow way is to name the destinations rather than take the whole internet:
///
/// ```ignore
/// SandboxNetwork::HostOnly.with_domain_suffixes([
///     "ubuntu.com",             // apt mirrors, for images without python3
///     "github.com",             // the uv release
///     "githubusercontent.com",  // where that release redirects to
///     "pypi.org",               // uv pip install
///     "pythonhosted.org",       // where the wheels themselves live
/// ])
/// ```
///
/// The redirect host and the wheel host are separate entries because a suffix
/// matches the name actually connected to, not the one first requested. An image
/// that already ships `python3` and `uv` needs only the last two.
/// `SandboxNetwork::Public` also works and is the blunt version of the same
/// grant. `sandbox_setup_under_domain_allowlist`, in the parent module's tests,
/// runs this script against a real VM under exactly the posture above. It is
/// `#[ignore]`d because it installs from three services this repo does not
/// control, so changing either the posture above or the script below means
/// running it by hand:
///
/// ```text
/// cargo test --features sandbox --lib -- --ignored sandbox_setup_under_domain_allowlist
/// ```
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

const SETUP_TIMEOUT_SECS: u64 = 300;

/// Python interpreter inside the ailoy venv. Resolved by sh at runtime via XDG_CACHE_HOME.
const AILOY_PYTHON: &str = r#""${XDG_CACHE_HOME:-$HOME/.cache}/ailoy/venv/bin/python3""#;

/// uv pip install targeting the ailoy venv. Both paths resolved by sh at runtime.
const AILOY_PIP_INSTALL: &str = r#""${XDG_CACHE_HOME:-$HOME/.cache}/ailoy/bin/uv" pip install --python "${XDG_CACHE_HOME:-$HOME/.cache}/ailoy/venv/bin/python3""#;

fn sh(cmd: &str) -> (String, Vec<String>) {
    ("sh".to_string(), vec!["-c".to_string(), cmd.to_string()])
}

async fn run_setup(console: &dyn Console) -> anyhow::Result<()> {
    let (prog, args) = sh(SETUP_CMD);
    let r = console
        .exec(prog, args, Some(SETUP_TIMEOUT_SECS))
        .await
        .context("Python runtime setup failed")?;
    if r.timed_out {
        anyhow::bail!("Python runtime setup timed out after {SETUP_TIMEOUT_SECS}s");
    }
    if r.exit_code != 0 {
        anyhow::bail!(
            "Python runtime setup failed (exit {}): {}",
            r.exit_code,
            if r.stderr.trim().is_empty() {
                r.stdout.trim()
            } else {
                r.stderr.trim()
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

    async fn ensure_ready(&self, console: &dyn Console) -> anyhow::Result<()> {
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
        console: &dyn Console,
        packages: &[&str],
    ) -> anyhow::Result<ExecResult> {
        if packages.is_empty() {
            return Ok(ExecResult {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: 0,
                timed_out: false,
            });
        }
        self.ensure_ready(console).await?;
        let quoted: Vec<String> = packages.iter().map(|p| format!("'{p}'")).collect();
        let cmd = format!("{AILOY_PIP_INSTALL} {}", quoted.join(" "));
        let (prog, args) = sh(&cmd);
        console.exec(prog, args, None).await
    }

    /// Execute a Python script with optional env vars.
    pub async fn run(
        &self,
        console: &dyn Console,
        source: &str,
        env: &[(&str, &str)],
    ) -> anyhow::Result<ExecResult> {
        self.run_with_timeout(console, source, env, 0).await
    }

    /// Like [`run`] but with a per-execution timeout (`0` = no timeout).
    pub async fn run_with_timeout(
        &self,
        console: &dyn Console,
        source: &str,
        env: &[(&str, &str)],
        timeout_secs: u64,
    ) -> anyhow::Result<ExecResult> {
        self.ensure_ready(console).await?;

        let script_path = format!("/tmp/__ailoy_{}.py", uuid::Uuid::new_v4());

        console
            .write(std::path::Path::new(&script_path), source.as_bytes())
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

        let timeout = (timeout_secs > 0).then_some(timeout_secs);
        let (prog, args) = sh(&cmd);
        let result = console.exec(prog, args, timeout).await;

        let (cleanup_prog, cleanup_args) = sh(&format!("rm -f {script_path}"));
        let _ = console.exec(cleanup_prog, cleanup_args, None).await;

        result.context("script execution error")
    }
}
