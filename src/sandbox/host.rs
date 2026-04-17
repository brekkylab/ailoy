use tokio::process::Command;

use super::{ExecRequest, ExecResult, Sandbox};

const MAX_OUTPUT_CHARS: usize = 8_000;

fn truncate(s: String) -> String {
    if s.len() <= MAX_OUTPUT_CHARS {
        return s;
    }
    let mut end = MAX_OUTPUT_CHARS;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[output truncated at {} chars]", &s[..end], MAX_OUTPUT_CHARS)
}

#[derive(Clone, Debug, Default)]
pub struct HostSandboxConfig {
    pub timeout_secs: u64,
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
        let timeout = tokio::time::Duration::from_secs(request.timeout_secs);

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
            tokio::io::BufReader::new(stdout_pipe).read_to_end(&mut buf).await?;
            Ok::<_, std::io::Error>(buf)
        });
        let err_task = tokio::spawn(async move {
            use tokio::io::AsyncReadExt as _;
            let mut buf = Vec::new();
            tokio::io::BufReader::new(stderr_pipe).read_to_end(&mut buf).await?;
            Ok::<_, std::io::Error>(buf)
        });

        match tokio::time::timeout(timeout, child.wait()).await {
            Ok(Ok(status)) => {
                let stdout_bytes = out_task.await.unwrap_or(Ok(vec![])).unwrap_or_default();
                let stderr_bytes = err_task.await.unwrap_or(Ok(vec![])).unwrap_or_default();
                Ok(ExecResult {
                    stdout: truncate(String::from_utf8_lossy(&stdout_bytes).into_owned()),
                    stderr: truncate(String::from_utf8_lossy(&stderr_bytes).into_owned()),
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
                    stderr: format!("command timed out after {}s", request.timeout_secs),
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
