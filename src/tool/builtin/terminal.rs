use super::super::function::FunctionTool;
use crate::value::Value;

pub fn create_terminal_tool(_config: Value) -> anyhow::Result<FunctionTool> {
    #[cfg(feature = "wasm")]
    {
        return Err(anyhow::anyhow!(
            "Builtin tool \"terminal\" is not supported on web browser environment."
        ));
    }

    #[cfg(not(feature = "wasm"))]
    {
        use std::{
            collections::HashMap,
            process::{Command, Stdio},
        };

        use crate::{
            to_value,
            tool::ToolFunc,
            value::{ToolDescBuilder, Value},
        };

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
        let desc = ToolDescBuilder::new("terminal")
        .description(format!(
            "Executes a command on the current system using the default shell. Current shell: {}.",
            current_shell
        ))
        .parameters(to_value!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to execute."},
                "cwd": {"type": "string", "description": "Optional working directory."},
                "env": {"type": "object", "description": "Optional environment variables as key-value pairs."},
                "stdin": {"type": "string", "description": "Optional string to send to STDIN."},
            },
            "required": ["command"]
        }))
        .returns(to_value!({
            "type": "object",
            "properties": {
                "stdout": {"type": "string", "description": "stdout of the executed command."},
                "stderr": {"type": "strsing", "description": "stderr of the executed command."},
                "exit_code": {"type": "number", "description": "Exit code of the executed command."}
            }
        }))
        .build();

        let f: Box<ToolFunc> = Box::new(move |args: Value| {
            Box::pin(async move {
                let args = match args.as_object() {
                    Some(a) => a,
                    None => {
                        return Ok(to_value!({
                            "stdout": "",
                            "stderr": "Invalid arguments: expected object",
                            "exit_code": -1 as i64
                        }));
                    }
                };

                let cmd_str = match args.get("command").and_then(|v| v.as_str()) {
                    Some(s) => s,
                    None => {
                        return Ok(to_value!({
                            "stdout": "",
                            "stderr": "Missing required 'command' string",
                            "exit_code": -1 as i64
                        }));
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
                let value = match command.spawn().and_then(|mut child| {
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
                        to_value!({
                            "stdout": stdout,
                            "stderr": stderr,
                            "exit_code": code as i64
                        })
                    }
                    Err(e) => {
                        to_value!({
                            "stdout": "",
                            "stderr": format!("failed to run command: {}", e),
                            "exit_code": -1 as i64
                        })
                    }
                };

                // Return
                Ok(value)
            })
        });

        Ok(FunctionTool::new(desc, std::sync::Arc::new(f)))
    }
}
