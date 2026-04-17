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
        Self {
            image: "alpine:3.21".to_string(),
            ncpu: 2,
            mem: 512,
            cwd: PathBuf::from("/workspace"),
            env: HashMap::new(),
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
        let rootfs = tokio::task::spawn_blocking(move || onlybots::pull(&image_ref))
            .await??;
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
}
