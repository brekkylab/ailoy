use std::{
    fmt::{self, Debug, Formatter},
    sync::Arc,
};

use ailoy_macros::multi_platform_async_trait;

use crate::{
    tool::Tool,
    value::{Part, ToolCallArg, ToolDesc},
};

#[derive(Clone)]
pub struct BuiltinTool {
    desc: ToolDesc,

    #[cfg(not(target_arch = "wasm32"))]
    f: Arc<dyn Fn(ToolCallArg) -> Part + Send + Sync + 'static>,

    #[cfg(target_arch = "wasm32")]
    f: Arc<dyn Fn(ToolCallArg) -> Part + 'static>,
}

impl Debug for BuiltinTool {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuiltinTool")
            .field("desc", &self.desc)
            .field("f", &"(Function)")
            .finish()
    }
}

impl BuiltinTool {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(
        desc: ToolDesc,
        f: Arc<dyn Fn(ToolCallArg) -> Part + Send + Sync + 'static>,
    ) -> Self {
        BuiltinTool { desc, f }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new(desc: ToolDesc, f: Arc<dyn Fn(ToolCallArg) -> Part + 'static>) -> Self {
        BuiltinTool { desc, f }
    }
}

#[multi_platform_async_trait]
impl Tool for BuiltinTool {
    fn get_description(&self) -> ToolDesc {
        self.desc.clone()
    }

    async fn run(&self, args: ToolCallArg) -> Result<Vec<Part>, String> {
        let tool_func = self.f.clone();
        Ok(vec![tool_func(args)])
    }
}

#[cfg(any(target_family = "unix", target_family = "windows"))]
pub fn create_terminal_tool() -> BuiltinTool {
    use std::{
        collections::HashMap,
        process::{Command, Stdio},
    };

    use crate::value::ToolDescArg;

    let current_shell = {
        #[cfg(target_family = "unix")]
        {
            "bash"
        }
        #[cfg(target_family = "windows")]
        {
            "powershell"
        }
    };

    let desc = ToolDesc::new(
        "execute_command",
        format!(
            "Executes a command on the current system using the default shell. Current shell: {}. \
            Optional fields: cwd (string), env (object), stdin (string)",
            current_shell
        ),
        ToolDescArg::new_object().with_properties(
            [
                (
                    "command",
                    ToolDescArg::new_string().with_desc("The command to execute."),
                ),
                (
                    "cwd",
                    ToolDescArg::new_string().with_desc("Optional working directory."),
                ),
                (
                    "env",
                    ToolDescArg::new_object()
                        .with_desc("Optional environment variables as key-value pairs."),
                ),
                (
                    "stdin",
                    ToolDescArg::new_string().with_desc("Optional string to send to STDIN."),
                ),
            ],
            ["command"],
        ),
        Some(ToolDescArg::new_object().with_properties(
            [
                (
                    "stdout",
                    ToolDescArg::new_string().with_desc("Captured STDOUT"),
                ),
                (
                    "stderr",
                    ToolDescArg::new_string().with_desc("Captured STDERR"),
                ),
                (
                    "exit_code",
                    ToolDescArg::new_number().with_desc("Process exit code"),
                ),
            ],
            ["stdout", "stderr", "exit_code"],
        )),
    );

    let f = Arc::new(|args: ToolCallArg| {
        let args = match args.as_object() {
            Some(a) => a,
            None => {
                return Part::Text(
                    serde_json::json!({
                        "stdout": "",
                        "stderr": "Invalid arguments: expected object",
                        "exit_code": -1
                    })
                    .to_string(),
                );
            }
        };

        let cmd_str = match args.get("command").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return Part::Text(
                    serde_json::json!({
                        "stdout": "",
                        "stderr": "Missing required 'command' string",
                        "exit_code": -1
                    })
                    .to_string(),
                );
            }
        };

        // optional cwd
        let cwd = args.get("cwd").and_then(|v| v.as_str());

        // optional env
        let env_map: Option<HashMap<String, String>> =
            args.get("env").and_then(|v| v.as_object()).map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            });

        // optional stdin
        let stdin_data = args.get("stdin").and_then(|v| v.as_str());

        // Prepare command
        #[cfg(target_family = "unix")]
        let mut command = Command::new("bash");
        #[cfg(target_family = "unix")]
        {
            command.args(["-lc", cmd_str]);
        }
        #[cfg(target_family = "windows")]
        let mut command = Command::new("powershell");
        #[cfg(target_family = "windows")]
        {
            command.args(["-NoProfile", "-Command", cmd_str]);
        }
        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        // Apply cwd
        if let Some(dir) = cwd
            && !dir.is_empty()
        {
            command.current_dir(dir);
        }

        // Apply env
        if let Some(envs) = env_map {
            for (k, v) in envs {
                command.env(k, v);
            }
        }

        // Apply stdin
        if stdin_data.is_some() {
            command.stdin(Stdio::piped());
        }

        // Run
        let output = match command.spawn().and_then(|mut child| {
            if let Some(input) = stdin_data {
                if let Some(mut stdin) = child.stdin.take() {
                    use std::io::Write;
                    let _ = stdin.write_all(input.as_bytes());
                }
            }
            child.wait_with_output()
        }) {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                let code = out.status.code().unwrap_or(-1);
                serde_json::json!({
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": code
                })
            }
            Err(e) => {
                serde_json::json!({
                    "stdout": "",
                    "stderr": format!("failed to run command: {}", e),
                    "exit_code": -1
                })
            }
        };

        // Return
        Part::Text(output.to_string())
    });

    BuiltinTool::new(desc, f)
}
