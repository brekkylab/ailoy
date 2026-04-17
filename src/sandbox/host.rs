use tokio::process::Command;

use super::{ExecRequest, ExecResult, Sandbox};

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

#[derive(Clone, Debug)]
pub struct HostSandboxConfig {
    /// Hard cap on execution time. 0 means no cap; request.timeout_secs is used as-is.
    pub timeout_secs: u64,
    pub max_output_chars: usize,
}

impl Default for HostSandboxConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 0,
            max_output_chars: 8_000,
        }
    }
}

pub struct HostSandbox {
    config: HostSandboxConfig,
}

impl HostSandbox {
    pub fn new(config: HostSandboxConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl Sandbox for HostSandbox {
    async fn exec(&self, request: ExecRequest) -> anyhow::Result<ExecResult> {
        let effective_timeout_secs = if self.config.timeout_secs > 0 {
            self.config.timeout_secs.min(request.timeout_secs)
        } else {
            request.timeout_secs
        };
        let timeout = tokio::time::Duration::from_secs(effective_timeout_secs);

        let mut child = Command::new("sh")
            .arg("-c")
            .arg(&request.command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        let stdout_pipe = child.stdout.take().expect("stdout piped");
        let stderr_pipe = child.stderr.take().expect("stderr piped");

        let out_task = tokio::spawn(async move {
            use tokio::io::AsyncReadExt as _;
            let mut buf = Vec::new();
            tokio::io::BufReader::new(stdout_pipe)
                .read_to_end(&mut buf)
                .await?;
            Ok::<_, std::io::Error>(buf)
        });
        let err_task = tokio::spawn(async move {
            use tokio::io::AsyncReadExt as _;
            let mut buf = Vec::new();
            tokio::io::BufReader::new(stderr_pipe)
                .read_to_end(&mut buf)
                .await?;
            Ok::<_, std::io::Error>(buf)
        });

        match tokio::time::timeout(timeout, child.wait()).await {
            Ok(Ok(status)) => {
                let stdout_bytes = out_task.await.unwrap_or(Ok(vec![])).unwrap_or_default();
                let stderr_bytes = err_task.await.unwrap_or(Ok(vec![])).unwrap_or_default();
                Ok(ExecResult {
                    stdout: truncate(
                        String::from_utf8_lossy(&stdout_bytes).into_owned(),
                        self.config.max_output_chars,
                    ),
                    stderr: truncate(
                        String::from_utf8_lossy(&stderr_bytes).into_owned(),
                        self.config.max_output_chars,
                    ),
                    exit_code: status.code().unwrap_or(-1),
                    timed_out: false,
                })
            }
            Ok(Err(e)) => {
                out_task.abort();
                err_task.abort();
                Err(anyhow::anyhow!("failed to wait for process: {}", e))
            }
            Err(_) => {
                let _ = child.kill().await;
                out_task.abort();
                err_task.abort();
                Ok(ExecResult {
                    stdout: String::new(),
                    stderr: format!("command timed out after {}s", effective_timeout_secs),
                    exit_code: -1,
                    timed_out: true,
                })
            }
        }
    }

    async fn shutdown(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::ExecRequest;

    #[tokio::test]
    async fn test_host_sandbox_echo() {
        let sandbox = HostSandbox::new(HostSandboxConfig::default());
        let result = sandbox
            .exec(ExecRequest {
                command: "echo hello".to_string(),
                timeout_secs: 10,
            })
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.trim() == "hello");
    }

    #[tokio::test]
    async fn test_host_sandbox_stderr() {
        let sandbox = HostSandbox::new(HostSandboxConfig::default());
        let result = sandbox
            .exec(ExecRequest {
                command: "echo err >&2".to_string(),
                timeout_secs: 10,
            })
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stderr.trim() == "err");
    }

    #[tokio::test]
    async fn test_host_sandbox_exit_code() {
        let sandbox = HostSandbox::new(HostSandboxConfig::default());
        let result = sandbox
            .exec(ExecRequest {
                command: "exit 42".to_string(),
                timeout_secs: 10,
            })
            .await
            .unwrap();
        assert_eq!(result.exit_code, 42);
    }

    #[tokio::test]
    async fn test_host_sandbox_timeout() {
        let sandbox = HostSandbox::new(HostSandboxConfig::default());
        let result = sandbox
            .exec(ExecRequest {
                command: "sleep 100".to_string(),
                timeout_secs: 1,
            })
            .await
            .unwrap();
        assert!(result.timed_out);
        assert_eq!(result.exit_code, -1);
    }

    #[tokio::test]
    async fn test_config_timeout_secs_caps_request() {
        // config.timeout_secs=1 should override request.timeout_secs=100
        let sandbox = HostSandbox::new(HostSandboxConfig {
            timeout_secs: 1,
            ..Default::default()
        });
        let result = sandbox
            .exec(ExecRequest {
                command: "sleep 100".to_string(),
                timeout_secs: 100,
            })
            .await
            .unwrap();
        assert!(result.timed_out);
        assert!(result.stderr.contains("1s"));
    }

    #[tokio::test]
    async fn test_config_timeout_secs_zero_means_no_cap() {
        // config.timeout_secs=0 should not override request.timeout_secs
        let sandbox = HostSandbox::new(HostSandboxConfig {
            timeout_secs: 0,
            ..Default::default()
        });
        let result = sandbox
            .exec(ExecRequest {
                command: "echo ok".to_string(),
                timeout_secs: 5,
            })
            .await
            .unwrap();
        assert!(!result.timed_out);
        assert_eq!(result.stdout.trim(), "ok");
    }

    #[tokio::test]
    async fn test_config_max_output_chars_truncates() {
        let sandbox = HostSandbox::new(HostSandboxConfig {
            max_output_chars: 10,
            ..Default::default()
        });
        let result = sandbox
            .exec(ExecRequest {
                command: "printf '%0.s1234567890' {1..5}".to_string(),
                timeout_secs: 5,
            })
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("[output truncated at 10 chars]"));
        assert!(result.stdout.starts_with("1234567890"));
    }
}
