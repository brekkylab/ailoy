use std::sync::Arc;

use futures::future::BoxFuture;

use super::{PythonReplConfig, install_python_packages, prepare_python_env};
use crate::{
    agent::rt::tool::{ToolAsyncFunc, ToolKind, ToolRuntime},
    datatype::Value,
    message::ToolDesc,
};

const DEFAULT_TIMEOUT_SECS: u64 = 60;

pub(crate) struct PythonSourceToolConfig {
    pub(crate) source: String,
    pub(crate) python: PythonReplConfig,
    pub(crate) timeout_secs: u64,
}

impl PythonSourceToolConfig {
    pub(crate) fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            python: PythonReplConfig::default(),
            timeout_secs: DEFAULT_TIMEOUT_SECS,
        }
    }

    pub(crate) fn python(mut self, python: PythonReplConfig) -> Self {
        self.python = python;
        self
    }

    pub(crate) fn timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }
}

pub(crate) async fn build_python_source_tool(
    desc: ToolDesc,
    config: PythonSourceToolConfig,
) -> anyhow::Result<ToolRuntime> {
    let env = Arc::new(prepare_python_env(&config.python).await?);
    install_python_packages(env.as_ref(), &config.python.packages).await?;

    let source = std::sync::Arc::new(config.source);
    let timeout_secs = config.timeout_secs;

    let f: Arc<ToolAsyncFunc> = Arc::new(move |args: Value| {
        let env = env.clone();
        let source = source.clone();

        Box::pin(async move {
            let args_dir = match tempfile::tempdir() {
                Ok(dir) => dir,
                Err(err) => {
                    return python_source_error_value(
                        &args,
                        "initialization",
                        &format!("failed to create temp args directory: {err}"),
                        None,
                        None,
                        None,
                        None,
                    );
                }
            };
            let args_path = args_dir.path().join("args.json");
            let args_json = match serde_json::to_string(&args) {
                Ok(json) => json,
                Err(err) => {
                    return python_source_error_value(
                        &args,
                        "initialization",
                        &format!("failed to serialize args to JSON: {err}"),
                        None,
                        None,
                        None,
                        None,
                    );
                }
            };
            if let Err(err) = std::fs::write(&args_path, args_json) {
                return python_source_error_value(
                    &args,
                    "initialization",
                    &format!("failed to write args JSON file: {err}"),
                    None,
                    None,
                    None,
                    None,
                );
            }
            let args_path_str = args_path.to_string_lossy().to_string();
            let result = env
                .run_code_with_env(
                    &source,
                    timeout_secs,
                    &[("AILOY_ARGS_JSON_PATH", args_path_str.as_str())],
                )
                .await;

            let result = match result {
                Ok(result) => result,
                Err(err) => {
                    return python_source_error_value(
                        &args,
                        "execution",
                        &format!("execution error: {err}"),
                        None,
                        None,
                        None,
                        None,
                    );
                }
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

    Ok(ToolRuntime::new_async_with_kind(desc, f, ToolKind::Builtin))
}

fn python_execution_error_message(result: &super::env::ExecResult, timeout_secs: u64) -> String {
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
    result: &super::env::ExecResult,
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
    result: &super::env::ExecResult,
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
        message::{Part, ToolDescBuilder},
        to_value,
    };

    fn test_tool_desc(name: &str) -> ToolDesc {
        ToolDescBuilder::new(name).build()
    }

    #[test]
    fn test_python_source_tool_config_defaults() {
        let config = PythonSourceToolConfig::new("print('ok')");
        assert_eq!(config.timeout_secs, DEFAULT_TIMEOUT_SECS);
    }

    #[test_with::executable(uv)]
    #[tokio::test]
    async fn test_build_python_source_tool_descriptor_reflects_config() {
        let mut desc = ToolDescBuilder::new("echo_json")
            .parameters(to_value!({
                "type": "object",
                "properties": { "value": { "type": "integer" } },
                "required": ["value"]
            }))
            .build();
        desc.description = Some("Echo args".to_string());
        desc.returns = Some(to_value!({
            "type": "object",
            "properties": { "value": { "type": "integer" } },
            "required": ["value"]
        }));

        let tool = build_python_source_tool(
            desc,
            PythonSourceToolConfig::new(
                "import json, os\nwith open(os.environ['AILOY_ARGS_JSON_PATH'], encoding='utf-8') as f:\n    args = json.load(f)\nprint(json.dumps(args), end='')",
            ),
        )
        .await
        .expect("failed to build source tool");

        let desc = tool.desc();
        assert_eq!(desc.name, "echo_json");
        assert_eq!(desc.description.as_deref(), Some("Echo args"));
        assert_eq!(
            desc.parameters
                .pointer("/required/0")
                .and_then(|value| value.as_str()),
            Some("value")
        );
        assert_eq!(
            desc.returns
                .as_ref()
                .and_then(|value| value.pointer("/required/0"))
                .and_then(|value| value.as_str()),
            Some("value")
        );
    }

    #[test_with::executable(uv)]
    #[tokio::test]
    async fn test_stdout_json_transport_parses_result() {
        let tool = build_python_source_tool(
            test_tool_desc("echo_json"),
            PythonSourceToolConfig::new(
                "import json, os\nwith open(os.environ['AILOY_ARGS_JSON_PATH'], encoding='utf-8') as f:\n    args = json.load(f)\nprint(json.dumps({'value': args['value']}), end='')",
            ),
        )
        .await
        .expect("failed to build source tool");

        let call = Part::function_with_id("call-1", "echo_json", to_value!({ "value": 7 }));
        let msg = tool.run(call).await.expect("tool execution failed");
        let value = msg.contents[0].as_value().unwrap();

        assert_eq!(
            value.pointer("/value").and_then(|value| value.as_integer()),
            Some(7),
            "unexpected value: {value:?}"
        );
    }

    #[test_with::executable(uv)]
    #[tokio::test]
    async fn test_source_receives_args_via_env_json_path() {
        let tool = build_python_source_tool(
            test_tool_desc("args_from_env"),
            PythonSourceToolConfig::new(
                "import json, os\nwith open(os.environ['AILOY_ARGS_JSON_PATH'], encoding='utf-8') as f:\n    args = json.load(f)\nprint(json.dumps({'value': args.get('value')}), end='')",
            ),
        )
        .await
        .expect("failed to build source tool");

        let call = Part::function_with_id("call-1", "args_from_env", to_value!({ "value": 11 }));
        let msg = tool.run(call).await.expect("tool execution failed");
        let value = msg.contents[0].as_value().unwrap();

        assert_eq!(
            value.pointer("/value").and_then(|value| value.as_integer()),
            Some(11),
            "unexpected value: {value:?}"
        );
    }

    #[test_with::executable(uv)]
    #[tokio::test]
    async fn test_large_json_payload_is_truncated_and_returns_parsing_error() {
        let tool = build_python_source_tool(
            test_tool_desc("big_payload"),
            PythonSourceToolConfig::new(
                "import json\nprint(json.dumps({'payload': 'x' * 9001}), end='')",
            ),
        )
        .await
        .expect("failed to build source tool");

        let call = Part::function_with_id("call-1", "big_payload", to_value!({}));
        let msg = tool.run(call).await.expect("tool execution failed");
        let value = msg.contents[0].as_value().unwrap();
        assert_eq!(
            value.pointer("/phase").and_then(|value| value.as_str()),
            Some("parsing"),
            "unexpected value: {value:?}"
        );
    }

    #[test_with::executable(uv)]
    #[tokio::test]
    async fn test_python_exception_returns_execution_error() {
        let tool = build_python_source_tool(
            ToolDescBuilder::new("raise_error")
                .parameters(to_value!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } }
                }))
                .build(),
            PythonSourceToolConfig::new("raise RuntimeError('boom')"),
        )
        .await
        .expect("failed to build source tool");

        let call = Part::function_with_id("call-1", "raise_error", to_value!({ "value": "x" }));
        let msg = tool.run(call).await.expect("tool execution failed");
        let value = msg.contents[0].as_value().unwrap();

        assert_eq!(
            value.pointer("/phase").and_then(|value| value.as_str()),
            Some("execution"),
            "unexpected value: {value:?}"
        );
        assert!(
            value
                .pointer("/error")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .contains("boom"),
            "unexpected value: {value:?}"
        );
    }

    #[test_with::executable(uv)]
    #[tokio::test]
    async fn test_invalid_json_stdout_returns_parsing_error() {
        let tool = build_python_source_tool(
            test_tool_desc("invalid_stdout"),
            PythonSourceToolConfig::new("print('oops', end='')"),
        )
        .await
        .expect("failed to build source tool");

        let call = Part::function_with_id("call-1", "invalid_stdout", to_value!({}));
        let msg = tool.run(call).await.expect("tool execution failed");
        let value = msg.contents[0].as_value().unwrap();

        assert_eq!(
            value.pointer("/phase").and_then(|value| value.as_str()),
            Some("parsing"),
            "unexpected value: {value:?}"
        );
    }
}
