use std::{collections::HashMap, path::PathBuf, sync::Arc};

use tokio::sync::Mutex;

use super::{ExecRequest, ExecResult, Sandbox};

#[derive(Clone, Debug)]
pub struct KrunSandboxConfig {
    pub image: String,
    pub ncpu: u8,
    pub mem: u32,
    pub cwd: PathBuf,
    pub env: HashMap<String, String>,
    pub default_timeout_secs: u64,
    pub max_output_chars: usize,
}

impl Default for KrunSandboxConfig {
    fn default() -> Self {
        // Provide a standard PATH so commands like python3, pip, sh builtins are
        // found regardless of image. The VM starts as a non-login shell (chroot
        // /bin/sh -c), which inherits only what the caller passes in.
        let mut env = HashMap::new();
        env.insert(
            "PATH".to_string(),
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_string(),
        );
        Self {
            image: "alpine:3.21".to_string(),
            ncpu: 2,
            mem: 512,
            cwd: PathBuf::from("/root"),
            env,
            default_timeout_secs: 60,
            max_output_chars: 8_000,
        }
    }
}

pub struct KrunSandbox {
    config: KrunSandboxConfig,
    image: onlybots::Image,
    snapshot: Arc<Mutex<Option<Vec<u8>>>>,
}

impl KrunSandbox {
    pub async fn new(config: KrunSandboxConfig) -> anyhow::Result<Self> {
        let image_ref = config.image.clone();
        let rootfs = tokio::task::spawn_blocking(move || onlybots::pull(&image_ref)).await??;
        let image = onlybots::list_images()?
            .into_iter()
            .find(|i| i.rootfs == rootfs)
            .ok_or_else(|| anyhow::anyhow!("pulled image not found in list"))?;
        Ok(Self {
            config,
            image,
            snapshot: Arc::new(Mutex::new(None)),
        })
    }
}

fn truncate(s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[output truncated at {} chars]", &s[..end], max)
}

#[async_trait::async_trait]
impl Sandbox for KrunSandbox {
    async fn exec(&self, request: ExecRequest) -> anyhow::Result<ExecResult> {
        let snapshot = self.snapshot.lock().await.take();
        let config = self.config.clone();
        let image = self.image.clone();

        let result = tokio::task::spawn_blocking(move || {
            let mut vm = onlybots::VM::new(image)
                .ncpu(config.ncpu)
                .mem(config.mem)
                .cwd(config.cwd.clone());

            for (k, v) in &config.env {
                vm = vm.env(k, v);
            }
            if let Some(snap) = snapshot {
                vm = vm.snapshot(snap);
            }

            vm.run(&request.command)
        })
        .await??;

        *self.snapshot.lock().await = Some(result.archive);

        let stdout = truncate(
            String::from_utf8_lossy(&result.output.stdout).into_owned(),
            config.max_output_chars,
        );
        let stderr = truncate(
            String::from_utf8_lossy(&result.output.stderr).into_owned(),
            config.max_output_chars,
        );
        let exit_code = result.output.status.code().unwrap_or(-1);

        Ok(ExecResult {
            stdout,
            stderr,
            exit_code,
            timed_out: false,
        })
    }

    async fn shutdown(&self) -> anyhow::Result<()> {
        *self.snapshot.lock().await = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn alpine_sandbox() -> KrunSandbox {
        KrunSandbox::new(KrunSandboxConfig::default())
            .await
            .expect("KrunSandbox::new failed")
    }

    #[tokio::test]
    async fn test_krun_sandbox_basic_exec() {
        let sandbox = alpine_sandbox().await;
        let result = sandbox
            .exec(ExecRequest {
                command: "echo hello".to_string(),
                timeout_secs: 60,
            })
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.stdout.trim(), "hello");
    }

    #[tokio::test]
    async fn test_krun_sandbox_snapshot_persistence() {
        let sandbox = alpine_sandbox().await;
        sandbox
            .exec(ExecRequest {
                command: "echo persisted > /root/data.txt".to_string(),
                timeout_secs: 30,
            })
            .await
            .unwrap();
        let result = sandbox
            .exec(ExecRequest {
                command: "cat /root/data.txt".to_string(),
                timeout_secs: 30,
            })
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(
            result.stdout.contains("persisted"),
            "stdout: {:?}",
            result.stdout
        );
    }

    async fn python_sandbox() -> KrunSandbox {
        KrunSandbox::new(KrunSandboxConfig {
            image: "python:3.12-alpine".to_string(),
            ..Default::default()
        })
        .await
        .expect("KrunSandbox::new failed")
    }

    #[tokio::test]
    async fn test_krun_sandbox_python_exec() {
        let sandbox = python_sandbox().await;
        let result = sandbox
            .exec(ExecRequest {
                command: "python3 -c 'print(1 + 1)'".to_string(),
                timeout_secs: 60,
            })
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
        assert_eq!(result.stdout.trim(), "2");
    }

    #[tokio::test]
    async fn test_krun_sandbox_python_stdlib_and_math() {
        let sandbox = python_sandbox().await;

        // Exercise stdlib (json, math) in a single exec to avoid snapshot issues
        let result = sandbox
            .exec(ExecRequest {
                command:
                    "python3 -c 'import json, math; d={\"pi\": math.pi}; print(json.dumps(d))'"
                        .to_string(),
                timeout_secs: 60,
            })
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
        let stdout = result.stdout.trim();
        assert!(
            stdout.contains("\"pi\""),
            "expected JSON with pi, got: {stdout:?}"
        );
    }
}
