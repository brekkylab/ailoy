#![cfg(feature = "sandbox-microvm")]

use std::sync::Arc;

use futures::future::BoxFuture;

use crate::{
    datatype::Value,
    message::ToolDesc,
    sandbox::{ExecResult, Sandbox},
    tool::{Tool, ToolFunc},
};

const DEFAULT_TIMEOUT_SECS: u64 = 60;

pub(crate) struct PythonSourceToolConfig {
    pub(crate) source: String,
    pub(crate) timeout_secs: u64,
}

impl PythonSourceToolConfig {
    pub(crate) fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            timeout_secs: DEFAULT_TIMEOUT_SECS,
        }
    }

    pub(crate) fn timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }
}

pub(crate) async fn build_python_source_tool(
    desc: ToolDesc,
    config: PythonSourceToolConfig,
    sandbox: Arc<Sandbox>,
) -> anyhow::Result<Tool> {
    let source = Arc::new(config.source);
    let timeout_secs = config.timeout_secs;

    let f: ToolFunc = ToolFunc::new(move |args: Value| {
        let sandbox = sandbox.clone();
        let source = source.clone();

        Box::pin(async move {
            // Serialize args to JSON and write into the sandbox
            let args_json = match serde_json::to_string(&args) {
                Ok(j) => j,
                Err(e) => return python_source_error_value(
                    &args, "initialization",
                    &format!("failed to serialize args: {e}"),
                    None, None, None, None,
                ),
            };

            if let Err(e) = sandbox.write_file("/workspace/__ailoy_args.json", args_json.as_bytes()).await {
                return python_source_error_value(
                    &args, "initialization",
                    &format!("failed to write args to sandbox: {e}"),
                    None, None, None, None,
                );
            }

            if let Err(e) = sandbox.write_file("/workspace/__ailoy_src.py", source.as_bytes()).await {
                return python_source_error_value(
                    &args, "initialization",
                    &format!("failed to write source to sandbox: {e}"),
                    None, None, None, None,
                );
            }

            let result = sandbox
                .shell("AILOY_ARGS_JSON_PATH=/workspace/__ailoy_args.json python3 /workspace/__ailoy_src.py")
                .await;

            let result = match result {
                Ok(r) => r,
                Err(e) => return python_source_error_value(
                    &args, "execution",
                    &format!("execution error: {e}"),
                    None, None, None, None,
                ),
            };

            if result.exit_code != 0 {
                return python_source_error_value(
                    &args,
                    "execution",
                    &python_execution_error_message(&result, timeout_secs),
                    Some(result.stdout.as_str()),
                    Some(result.stderr.as_str()),
                    Some(result.exit_code as i64),
                    Some(result.timed_out),
                );
            }

            parse_python_source_json(&args, &result, &result.stdout)
        }) as BoxFuture<'static, Value>
    });

    Ok(Tool::new(desc, Arc::new(f)))
}

fn python_execution_error_message(result: &ExecResult, timeout_secs: u64) -> String {
    let stderr = result.stderr.trim();
    let stdout = result.stdout.trim();

    if !stderr.is_empty() {
        format!("python source execution failed: {stderr}")
    } else if !stdout.is_empty() {
        format!("python source execution failed: {stdout}")
    } else if result.timed_out {
        format!("python source timed out after {timeout_secs}s")
    } else {
        format!("python source exited with code {}", result.exit_code)
    }
}

fn parse_python_source_json(
    processed_args: &Value,
    result: &ExecResult,
    payload: &str,
) -> Value {
    match serde_json::from_str::<Value>(payload) {
        Ok(value) => value,
        Err(err) => python_source_parsing_error(
            processed_args,
            result,
            &format!("failed to parse python result as JSON: {err}"),
        ),
    }
}

fn python_source_parsing_error(
    processed_args: &Value,
    result: &ExecResult,
    error: &str,
) -> Value {
    python_source_error_value(
        processed_args,
        "parsing",
        error,
        Some(result.stdout.as_str()),
        Some(result.stderr.as_str()),
        Some(result.exit_code as i64),
        Some(result.timed_out),
    )
}

fn python_source_error_value(
    base_args: &Value,
    phase: &str,
    error: &str,
    stdout: Option<&str>,
    stderr: Option<&str>,
    exit_code: Option<i64>,
    timed_out: Option<bool>,
) -> Value {
    let mut value = match base_args.clone() {
        Value::Object(map) => Value::Object(map),
        other => Value::object([("input", other)]),
    };

    let object = value
        .as_object_mut()
        .expect("python source error payload must be object");
    object.insert("error".to_string(), Value::string(error));
    object.insert("phase".to_string(), Value::string(phase));

    if let Some(stdout) = stdout.filter(|stdout| !stdout.is_empty()) {
        object.insert("stdout".to_string(), Value::string(stdout));
    }
    if let Some(stderr) = stderr.filter(|stderr| !stderr.is_empty()) {
        object.insert("stderr".to_string(), Value::string(stderr));
    }
    if let Some(exit_code) = exit_code {
        object.insert("exit_code".to_string(), Value::integer(exit_code));
    }
    if let Some(timed_out) = timed_out {
        object.insert("timed_out".to_string(), Value::bool(timed_out));
    }

    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        message::ToolDescBuilder,
        to_value,
    };

    #[test]
    fn test_python_source_tool_config_defaults() {
        let config = PythonSourceToolConfig::new("print('ok')");
        assert_eq!(config.timeout_secs, DEFAULT_TIMEOUT_SECS);
    }
}
