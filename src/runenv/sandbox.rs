//! Thin wrapper around the `microsandbox` crate, exposing an ailoy-internal
//! `Sandbox` type so the public API is not coupled to the underlying library.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Weak},
};

use super::{ExecResult, RunEnv};

use microsandbox::{Sandbox as MsbSandbox, sandbox::PullPolicy};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

/// A volume mount attached to a sandbox at creation time.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VolumeMount {
    /// Bind-mount a host directory into the guest.
    Bind {
        /// Absolute or relative host path.
        host: PathBuf,
        /// Absolute guest path (e.g. `/data`).
        guest: String,
        /// When `true`, the guest cannot write to this mount.
        #[serde(default)]
        readonly: bool,
    },
    /// Mount a microsandbox named volume (`~/.microsandbox/volumes/<name>/`).
    /// The volume persists across sandbox restarts and can be shared between sandboxes.
    Named {
        /// Name of the pre-existing microsandbox volume.
        name: String,
        /// Absolute guest path.
        guest: String,
        /// When `true`, the guest cannot write to this mount.
        #[serde(default)]
        readonly: bool,
    },
    /// Memory-backed temporary filesystem. Disappears when the sandbox stops.
    Tmpfs {
        /// Absolute guest path.
        guest: String,
        /// Size limit in MiB. `None` means no limit.
        size_mib: Option<u32>,
    },
}

/// Configuration for creating a new sandbox.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SandboxConfig {
    /// OCI container image. Default: `"ubuntu:latest"`.
    pub image: String,

    /// Number of virtual CPUs. Default: `2`.
    pub cpus: u8,

    /// Guest memory in MiB. Default: `2048`.
    pub memory_mib: u32,

    /// Default working directory inside the sandbox. Default: `"/workspace"`.
    /// Created automatically after boot if it does not already exist.
    pub workdir: String,

    /// Environment variables passed to every command.
    pub env: HashMap<String, String>,

    /// When `true`, disable all network access. Default: `false`.
    pub disable_network: bool,

    /// Maximum characters to keep from stdout/stderr. Default: `30_000`.
    pub max_output_chars: usize,

    /// Volume mounts attached at sandbox creation time.
    #[serde(default)]
    pub volumes: Vec<VolumeMount>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            image: "ubuntu:latest".to_string(),
            cpus: 2,
            memory_mib: 2048,
            workdir: "/workspace".to_string(),
            env: HashMap::new(),
            disable_network: false,
            max_output_chars: 30_000,
            volumes: Vec::new(),
        }
    }
}

pub struct Sandbox {
    name: String,
    /// Weak ref to the currently-running sandbox; dead when stopped.
    active: Mutex<Weak<MsbSandbox>>,
    max_output_chars: usize,
}

#[async_trait::async_trait]
impl RunEnv for Sandbox {
    async fn exec(
        &self,
        program: String,
        args: Vec<String>,
        timeout: Option<u64>,
    ) -> anyhow::Result<ExecResult> {
        use microsandbox::MicrosandboxError;
        use microsandbox::sandbox::ExecOptionsBuilder;
        use std::time::Duration;
        let max_output_chars = self.max_output_chars;
        let sb = self.acquire().await?;
        let result = sb
            .exec_with(&program, |b: ExecOptionsBuilder| {
                let b = b.args(args.iter().map(|s| s.as_str()));
                if let Some(secs) = timeout {
                    b.timeout(Duration::from_secs(secs))
                } else {
                    b
                }
            })
            .await;
        self.release(sb).await?;
        match result {
            Ok(output) => Ok(ExecResult {
                stdout: truncate_output(output.stdout().unwrap_or_default(), max_output_chars),
                stderr: truncate_output(output.stderr().unwrap_or_default(), max_output_chars),
                exit_code: output.status().code,
                timed_out: false,
            }),
            Err(MicrosandboxError::ExecTimeout(_)) => Ok(ExecResult {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: -1,
                timed_out: true,
            }),
            Err(e) => Ok(ExecResult {
                stdout: String::new(),
                stderr: e.to_string(),
                exit_code: -1,
                timed_out: false,
            }),
        }
    }

    async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
        let path = path.to_string_lossy().into_owned();
        let sb = self.acquire().await?;
        let rv = sb.fs().read(&path).await?.to_vec();
        self.release(sb).await?;
        Ok(rv)
    }

    async fn write(&self, path: &Path, content: &[u8]) -> anyhow::Result<()> {
        let path_str = path.to_string_lossy().into_owned();
        let sb = self.acquire().await?;
        sb.fs().write(&path_str, content).await?;
        self.release(sb).await?;
        Ok(())
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let name = self.name.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = MsbSandbox::remove(&name).await;
            });
        }
    }
}

impl Sandbox {
    pub async fn new(config: SandboxConfig) -> anyhow::Result<Self> {
        if !microsandbox::setup::is_installed() {
            log::warn!(
                "microsandbox runtime not found — downloading to ~/.microsandbox, this may take a moment"
            );
            microsandbox::setup::install().await?;
            log::info!("microsandbox runtime installed");
        }

        let name = Uuid::new_v4().to_string();
        let max_output_chars = config.max_output_chars;

        let mut builder = MsbSandbox::builder(&name)
            .image(config.image.as_str())
            .cpus(config.cpus)
            .memory(config.memory_mib)
            // .idle_timeout(config.idle_timeout_secs)
            .pull_policy(PullPolicy::IfMissing);
        for (k, v) in &config.env {
            builder = builder.env(k.as_str(), v.as_str());
        }
        if config.disable_network {
            builder = builder.disable_network();
        }
        for mount in &config.volumes {
            builder = apply_volume_mount(builder, mount);
        }
        let inner = builder.create().await?;
        inner
            .exec_with("sh", |b| {
                b.args(["-c", &format!("mkdir -p {}", config.workdir)])
            })
            .await?;
        inner.stop_and_wait().await?;

        Ok(Self {
            name,
            max_output_chars,
            active: Mutex::new(Weak::new()),
        })
    }

    /// Acquire a handle to the sandbox, starting it if not currently running.
    async fn acquire(&self) -> anyhow::Result<Arc<MsbSandbox>> {
        let mut active = self.active.lock().await;
        match active.upgrade() {
            Some(arc) => Ok(arc),
            None => {
                let arc = Arc::new(MsbSandbox::start(&self.name).await?);
                *active = Arc::downgrade(&arc);
                Ok(arc)
            }
        }
    }

    async fn release(&self, v: Arc<MsbSandbox>) -> anyhow::Result<()> {
        let mut active = self.active.lock().await;
        if active.strong_count() == 1 {
            // Clear before stopping so concurrent acquire() sees a dead Weak
            // and won't hand out an Arc to a sandbox mid-stop.
            // Hold the lock across stop_and_wait to block new starts until done.
            *active = Weak::new();
            v.stop_and_wait().await?;
        }
        Ok(())
    }
}

fn apply_volume_mount(
    builder: microsandbox::sandbox::SandboxBuilder,
    mount: &VolumeMount,
) -> microsandbox::sandbox::SandboxBuilder {
    match mount {
        VolumeMount::Bind {
            host,
            guest,
            readonly,
        } => {
            let host = host.clone();
            let ro = *readonly;
            builder.volume(guest, move |m| {
                let m = m.bind(host);
                if ro { m.readonly() } else { m }
            })
        }
        VolumeMount::Named {
            name,
            guest,
            readonly,
        } => {
            let name = name.clone();
            let ro = *readonly;
            builder.volume(guest, move |m| {
                let m = m.named(name);
                if ro { m.readonly() } else { m }
            })
        }
        VolumeMount::Tmpfs { guest, size_mib } => {
            let size = *size_mib;
            builder.volume(guest, move |m| {
                let m = m.tmpfs();
                if let Some(s) = size { m.size(s) } else { m }
            })
        }
    }
}

fn truncate_output(s: String, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        return s;
    }
    let head = max_chars / 2;
    let tail = max_chars - head;
    let omitted = chars.len() - head - tail;
    let head_str: String = chars[..head].iter().collect();
    let tail_str: String = chars[chars.len() - tail..].iter().collect();
    format!("{head_str}\n\n... [{omitted} characters omitted] ...\n\n{tail_str}")
}

//--------------------------------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    async fn make_sandbox() -> Sandbox {
        Sandbox::new(SandboxConfig::default())
            .await
            .expect("failed to create sandbox")
    }

    fn sh(cmd: &str) -> (String, Vec<String>) {
        ("sh".to_string(), vec!["-c".to_string(), cmd.to_string()])
    }

    /// exec() runs a command and returns its output.
    #[tokio::test]
    async fn test_exec_basic() {
        let sb = make_sandbox().await;
        let (prog, args) = sh("echo hello world");
        let result = sb.exec(prog, args, None).await.unwrap();
        assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
        assert!(
            result.stdout.eq("hello world\n"),
            "stdout: {:?}",
            result.stdout
        );
    }

    /// exec() captures a non-zero exit code.
    #[tokio::test]
    async fn test_exec_exit_code() {
        let sb = make_sandbox().await;
        let (prog, args) = sh("exit 42");
        let result = sb.exec(prog, args, None).await.unwrap();
        assert_eq!(result.exit_code, 42);
    }

    /// exec() times out and sets timed_out flag.
    #[tokio::test]
    async fn test_exec_timeout() {
        let sb = make_sandbox().await;
        let (prog, args) = sh("sleep 10");
        let result = sb.exec(prog, args, Some(1)).await.unwrap();
        assert!(result.timed_out, "expected timed_out, got: {result:?}");
    }

    /// exec() + read() round-trip inside the sandbox.
    #[tokio::test]
    async fn test_write_read_roundtrip() {
        let sb = make_sandbox().await;
        let (prog, args) = sh("printf 'roundtrip' > /workspace/test.txt");
        assert_eq!(sb.exec(prog, args, None).await.unwrap().exit_code, 0);
        let bytes = sb
            .read(Path::new("/workspace/test.txt"))
            .await
            .expect("read failed");
        assert_eq!(bytes, b"roundtrip");
    }

    /// Bind mount: a file written to the host directory is visible inside the guest.
    #[tokio::test]
    async fn test_bind_mount_host_to_guest() {
        let host_dir = tempfile::tempdir().expect("failed to create temp dir");
        let host_file = host_dir.path().join("hello.txt");
        std::fs::write(&host_file, "bind_mount_works").expect("failed to write host file");

        let sb = Sandbox::new(SandboxConfig {
            volumes: vec![VolumeMount::Bind {
                host: host_dir.path().to_path_buf(),
                guest: "/mnt/host".to_string(),
                readonly: false,
            }],
            ..SandboxConfig::default()
        })
        .await
        .expect("failed to create sandbox");

        let (prog, args) = sh("cat /mnt/host/hello.txt");
        let result = sb.exec(prog, args, None).await.unwrap();
        assert_eq!(result.exit_code, 0, "cat failed, stderr: {}", result.stderr);
        assert!(
            result.stdout.contains("bind_mount_works"),
            "guest should see host file, stdout: {:?}",
            result.stdout
        );
    }

    /// Bind mount readonly: guest write is rejected.
    #[tokio::test]
    async fn test_bind_mount_readonly_rejects_guest_write() {
        let host_dir = tempfile::tempdir().expect("failed to create temp dir");

        let sb = Sandbox::new(SandboxConfig {
            volumes: vec![VolumeMount::Bind {
                host: host_dir.path().to_path_buf(),
                guest: "/mnt/ro".to_string(),
                readonly: true,
            }],
            ..SandboxConfig::default()
        })
        .await
        .expect("failed to create sandbox");

        let (prog, args) = sh("echo should_fail > /mnt/ro/file.txt");
        let result = sb.exec(prog, args, None).await.unwrap();
        assert_ne!(result.exit_code, 0, "write to read-only mount should fail");
    }

    /// Bind mount: a file written inside the guest is visible on the host.
    #[tokio::test]
    async fn test_bind_mount_guest_write_visible_on_host() {
        let host_dir = tempfile::tempdir().expect("failed to create temp dir");

        let sb = Sandbox::new(SandboxConfig {
            volumes: vec![VolumeMount::Bind {
                host: host_dir.path().to_path_buf(),
                guest: "/mnt/shared".to_string(),
                readonly: false,
            }],
            ..SandboxConfig::default()
        })
        .await
        .expect("failed to create sandbox");

        let (prog, args) = sh("echo guest_wrote > /mnt/shared/out.txt");
        let result = sb.exec(prog, args, None).await.unwrap();
        assert_eq!(
            result.exit_code, 0,
            "guest write failed, stderr: {}",
            result.stderr
        );

        let host_content =
            std::fs::read_to_string(host_dir.path().join("out.txt")).expect("host file not found");
        assert!(
            host_content.contains("guest_wrote"),
            "host should see guest-written file, got: {host_content:?}"
        );
    }

    /// Tmpfs: writable in-memory filesystem visible at the specified guest path.
    #[tokio::test]
    async fn test_tmpfs_mount_is_writable() {
        let sb = Sandbox::new(SandboxConfig {
            volumes: vec![VolumeMount::Tmpfs {
                guest: "/mnt/tmp".to_string(),
                size_mib: Some(64),
            }],
            ..SandboxConfig::default()
        })
        .await
        .expect("failed to create sandbox");

        let (prog, args) = sh("echo tmpfs_ok > /mnt/tmp/test.txt && cat /mnt/tmp/test.txt");
        let result = sb.exec(prog, args, None).await.unwrap();
        assert_eq!(
            result.exit_code, 0,
            "tmpfs write/read failed, stderr: {}",
            result.stderr
        );
        assert!(
            result.stdout.contains("tmpfs_ok"),
            "tmpfs should be writable, stdout: {:?}",
            result.stdout
        );
    }

    /// Named volume: data written in one sandbox instance is visible in another using the same volume.
    #[tokio::test]
    async fn test_named_volume_persists_across_sandboxes() {
        use microsandbox::Volume;

        let vol_name = format!("ailoy-test-vol-{}", Uuid::new_v4());

        Volume::builder(&vol_name)
            .create()
            .await
            .expect("failed to create named volume");

        let write_result = {
            let sb = Sandbox::new(SandboxConfig {
                volumes: vec![VolumeMount::Named {
                    name: vol_name.clone(),
                    guest: "/mnt/vol".to_string(),
                    readonly: false,
                }],
                ..SandboxConfig::default()
            })
            .await
            .expect("failed to create first sandbox");
            {
                let (prog, args) = sh("echo named_vol_works > /mnt/vol/data.txt");
                sb.exec(prog, args, None).await.unwrap()
            }
        };

        assert_eq!(
            write_result.exit_code, 0,
            "write to named volume failed, stderr: {}",
            write_result.stderr
        );

        let read_result = {
            let sb = Sandbox::new(SandboxConfig {
                volumes: vec![VolumeMount::Named {
                    name: vol_name.clone(),
                    guest: "/mnt/vol".to_string(),
                    readonly: false,
                }],
                ..SandboxConfig::default()
            })
            .await
            .expect("failed to create second sandbox");
            {
                let (prog, args) = sh("cat /mnt/vol/data.txt");
                sb.exec(prog, args, None).await.unwrap()
            }
        };

        if let Ok(handle) = Volume::get(&vol_name).await {
            let _ = handle.remove().await;
        }

        assert_eq!(
            read_result.exit_code, 0,
            "read from named volume failed, stderr: {}",
            read_result.stderr
        );
        assert!(
            read_result.stdout.contains("named_vol_works"),
            "second sandbox should see data written by first, stdout: {:?}",
            read_result.stdout
        );
    }
}
