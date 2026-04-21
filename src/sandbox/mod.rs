#![cfg(feature = "sandbox-microvm")]
//! Thin wrapper around the `microsandbox` crate, exposing an ailoy-internal
//! `Sandbox` type so the public API is not coupled to the underlying library.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use microsandbox::Sandbox as MsbSandbox;
use microsandbox::sandbox::{ExecOptionsBuilder, PullPolicy};
use uuid::Uuid;

//--------------------------------------------------------------------------------------------------
// Types
//--------------------------------------------------------------------------------------------------

/// Configuration for creating a new sandbox.
pub struct SandboxConfig {
    /// Unique sandbox name (e.g. "ailoy-{uuid}").
    pub name: String,

    /// OCI container image. Default: `"python:3.12-slim"`.
    pub image: String,

    /// Number of virtual CPUs. Default: `2`.
    pub cpus: u8,

    /// Guest memory in MiB. Default: `512`.
    pub memory_mib: u32,

    /// Default working directory inside the sandbox. Default: `"/workspace"`.
    pub workdir: String,

    /// Environment variables passed to every command.
    pub env: HashMap<String, String>,

    /// When `true`, disable all network access. Default: `false`.
    pub disable_network: bool,

    /// Idle shutdown timeout in seconds. Default: `300`.
    pub idle_timeout_secs: u64,

    /// Per-exec timeout in seconds. Default: `60`.
    pub default_timeout_secs: u64,

    /// Maximum characters to keep from stdout/stderr. Default: `8000`.
    pub max_output_chars: usize,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            name: format!("ailoy-{}", Uuid::new_v4()),
            image: "python:3.12-slim".to_string(),
            cpus: 2,
            memory_mib: 512,
            workdir: "/workspace".to_string(),
            env: HashMap::new(),
            disable_network: false,
            idle_timeout_secs: 300,
            default_timeout_secs: 60,
            max_output_chars: 8000,
        }
    }
}

/// The result of running a command inside a sandbox.
pub struct ExecResult {
    /// Captured stdout (possibly truncated to `max_output_chars`).
    pub stdout: String,
    /// Captured stderr (possibly truncated to `max_output_chars`).
    pub stderr: String,
    /// Process exit code.
    pub exit_code: i32,
    /// Whether the command was killed due to a timeout.
    pub timed_out: bool,
}

/// A running sandbox wrapping a `microsandbox::Sandbox`.
pub struct Sandbox {
    inner: MsbSandbox,
    default_timeout_secs: u64,
    max_output_chars: usize,
}

impl std::fmt::Debug for Sandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sandbox")
            .field("default_timeout_secs", &self.default_timeout_secs)
            .field("max_output_chars", &self.max_output_chars)
            .finish_non_exhaustive()
    }
}

//--------------------------------------------------------------------------------------------------
// Methods
//--------------------------------------------------------------------------------------------------

impl Sandbox {
    /// Boot a new sandbox from the given configuration.
    pub async fn new(config: SandboxConfig) -> anyhow::Result<Self> {
        let mut builder = MsbSandbox::builder(&config.name)
            .image(config.image.as_str())
            .cpus(config.cpus)
            .memory(config.memory_mib)
            .workdir(config.workdir.as_str())
            .idle_timeout(config.idle_timeout_secs)
            .pull_policy(PullPolicy::IfMissing);

        for (k, v) in &config.env {
            builder = builder.env(k.as_str(), v.as_str());
        }

        if config.disable_network {
            builder = builder.disable_network();
        }

        let inner = builder.create().await?;

        Ok(Self {
            inner,
            default_timeout_secs: config.default_timeout_secs,
            max_output_chars: config.max_output_chars,
        })
    }

    /// Run a command and wait for completion.
    pub async fn exec(&self, cmd: &str, args: &[&str]) -> anyhow::Result<ExecResult> {
        let timeout = Duration::from_secs(self.default_timeout_secs);
        let owned_args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let result = self
            .inner
            .exec_with(cmd, |b: ExecOptionsBuilder| {
                b.args(owned_args.iter().map(|s| s.as_str()))
                    .timeout(timeout)
            })
            .await;

        self.handle_exec_result(result)
    }

    /// Run a shell command via the sandbox's default shell.
    pub async fn shell(&self, script: &str) -> anyhow::Result<ExecResult> {
        let result = self.inner.shell(script).await;
        self.handle_exec_result(result)
    }

    /// Write raw bytes to a file inside the sandbox.
    pub async fn write_file(&self, guest_path: &str, data: &[u8]) -> anyhow::Result<()> {
        self.inner.fs().write(guest_path, data).await?;
        Ok(())
    }

    /// Read a file from the sandbox as a UTF-8 string.
    pub async fn read_file(&self, guest_path: &str) -> anyhow::Result<String> {
        let s = self.inner.fs().read_to_string(guest_path).await?;
        Ok(s)
    }

    /// Copy a file from the host filesystem into the sandbox.
    pub async fn copy_from_host(
        &self,
        host: &Path,
        guest: &str,
    ) -> anyhow::Result<()> {
        self.inner.fs().copy_from_host(host, guest).await?;
        Ok(())
    }

    /// Copy a file from the sandbox to the host filesystem.
    pub async fn copy_to_host(
        &self,
        guest: &str,
        host: &Path,
    ) -> anyhow::Result<()> {
        self.inner.fs().copy_to_host(guest, host).await?;
        Ok(())
    }

    /// Stop the sandbox and wait for the process to exit.
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        self.inner.stop_and_wait().await?;
        Ok(())
    }

    //----------------------------------------------------------------------------------------------
    // Helpers
    //----------------------------------------------------------------------------------------------

    fn handle_exec_result(
        &self,
        result: Result<microsandbox::ExecOutput, microsandbox::MicrosandboxError>,
    ) -> anyhow::Result<ExecResult> {
        use microsandbox::MicrosandboxError;

        match result {
            Ok(output) => {
                let stdout = truncate_output(
                    output.stdout().unwrap_or_default(),
                    self.max_output_chars,
                );
                let stderr = truncate_output(
                    output.stderr().unwrap_or_default(),
                    self.max_output_chars,
                );
                Ok(ExecResult {
                    stdout,
                    stderr,
                    exit_code: output.status().code,
                    timed_out: false,
                })
            }
            Err(MicrosandboxError::ExecTimeout(_)) => Ok(ExecResult {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: -1,
                timed_out: true,
            }),
            Err(e) => Err(e.into()),
        }
    }
}

//--------------------------------------------------------------------------------------------------
// Free functions
//--------------------------------------------------------------------------------------------------

/// Truncate a `String` to at most `max_chars` Unicode scalar values.
fn truncate_output(s: String, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s
    } else {
        s.chars().take(max_chars).collect()
    }
}
