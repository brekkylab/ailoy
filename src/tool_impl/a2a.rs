use serde::{Deserialize, Serialize};
use url::Url;

use crate::{
    datatype::Value,
    message::{Message, Part, Role, ToolDescBuilder},
    tool::{ToolContext, ToolFactory, ToolFunc},
};

// ── Tool constructor ──────────────────────────────────────────────────────────

/// Build a [`Tool`] that delegates to a remote A2A agent.
///
/// Eagerly fetches the agent card from `{url}/.well-known/agent-card.json` so
/// that the tool name and description are known at startup.  Each call sends a
/// JSON-RPC `message/send` request and returns the agent's text response.
pub(crate) async fn make_a2a_tool(url: Url) -> anyhow::Result<ToolFactory> {
    let base_url = url.to_string();
    let card = discover(&base_url).await?;

    let description = if card.skills.is_empty() {
        card.description.clone()
    } else {
        let skills = card
            .skills
            .iter()
            .map(|s| format!("* {}: {}", s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n");
        format!("{}\n\n# Skills\n\n{}", card.description, skills)
    };

    let desc = ToolDescBuilder::new(&card.name)
        .description(description)
        .parameters(crate::to_value!({"type": "string"}))
        .build();

    let f = ToolFunc::new(move |args: Value, ctx: ToolContext| {
        let url = base_url.clone();
        async move {
            let id = ctx.id;
            let task = match args.as_str() {
                Some(v) => v.to_string(),
                None => {
                    return Message::new(Role::Tool)
                        .with_contents([Part::text("Error: expected string argument")])
                        .with_id(id);
                }
            };
            match message_send(&url, &task).await {
                Ok(text) => Message::new(Role::Tool)
                    .with_contents([Part::text(text)])
                    .with_id(id),
                Err(e) => Message::new(Role::Tool)
                    .with_contents([Part::text(format!("Error: {e}"))])
                    .with_id(id),
            }
        }
    });

    Ok(ToolFactory::simple(desc, f))
}

// ── Agent Discovery ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AgentCard {
    pub(crate) name: String,
    pub(crate) description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) version: Option<String>,
    #[serde(default)]
    pub(crate) skills: Vec<AgentSkill>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) capabilities: Option<AgentCapabilities>,
    #[serde(default = "default_input_modes")]
    pub(crate) default_input_modes: Vec<String>,
    #[serde(default = "default_output_modes")]
    pub(crate) default_output_modes: Vec<String>,
}

fn default_input_modes() -> Vec<String> {
    vec!["text/plain".into()]
}

fn default_output_modes() -> Vec<String> {
    vec!["text/plain".into()]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AgentSkill {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) examples: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AgentCapabilities {
    pub(crate) streaming: bool,
}

// ── JSON-RPC 2.0 ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct JsonRpcRequest {
    pub(crate) jsonrpc: String,
    pub(crate) id: serde_json::Value,
    pub(crate) method: String,
    pub(crate) params: MessageSendParams,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MessageSendParams {
    pub(crate) message: A2AMessage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct A2AMessage {
    #[serde(default = "message_kind")]
    pub(crate) kind: String,
    pub(crate) role: A2ARole,
    pub(crate) parts: Vec<A2APart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) message_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) context_id: Option<String>,
}

fn message_kind() -> String {
    "message".into()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum A2ARole {
    User,
    Agent,
}

#[derive(Clone, Debug)]
pub(crate) enum A2APart {
    Text { text: String },
}

#[derive(Serialize)]
struct TextPartSer<'a> {
    kind: &'static str,
    text: &'a str,
}

impl Serialize for A2APart {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            A2APart::Text { text } => TextPartSer { kind: "text", text }.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for A2APart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let map: serde_json::Map<String, serde_json::Value> =
            serde_json::Map::deserialize(deserializer)?;
        if let Some(text) = map.get("text").and_then(|v| v.as_str()) {
            Ok(A2APart::Text {
                text: text.to_owned(),
            })
        } else {
            Err(serde::de::Error::missing_field("text"))
        }
    }
}

// ── Task ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Task {
    #[serde(default = "task_kind")]
    pub(crate) kind: String,
    pub(crate) id: String,
    pub(crate) context_id: String,
    pub(crate) status: TaskStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) history: Vec<A2AMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) artifacts: Vec<serde_json::Value>,
}

fn task_kind() -> String {
    "task".into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct TaskStatus {
    pub(crate) state: TaskState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) timestamp: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum TaskState {
    Submitted,
    Working,
    Completed,
    Failed,
}

// ── JSON-RPC Response ─────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct JsonRpcResponse {
    pub(crate) jsonrpc: String,
    pub(crate) id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) result: Option<Task>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    #[cfg(test)]
    pub(crate) fn success(id: serde_json::Value, task: Task) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(task),
            error: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn error(id: serde_json::Value, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
            }),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct JsonRpcError {
    pub(crate) code: i32,
    pub(crate) message: String,
}

// ── Client functions ──────────────────────────────────────────────────────────

/// Fetch an agent's card from `{base_url}/.well-known/agent-card.json`.
pub(crate) async fn discover(base_url: &str) -> anyhow::Result<AgentCard> {
    let url = format!(
        "{}/.well-known/agent-card.json",
        base_url.trim_end_matches('/')
    );
    let card = reqwest::get(&url)
        .await?
        .error_for_status()?
        .json::<AgentCard>()
        .await?;
    Ok(card)
}

/// Send a task to an A2A agent via JSON-RPC `message/send` and return the text response.
pub(crate) async fn message_send(base_url: &str, task: &str) -> anyhow::Result<String> {
    let url = base_url.trim_end_matches('/').to_string();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: serde_json::Value::Number(1.into()),
        method: "message/send".into(),
        params: MessageSendParams {
            message: A2AMessage {
                kind: "message".into(),
                role: A2ARole::User,
                parts: vec![A2APart::Text {
                    text: task.to_string(),
                }],
                message_id: Some(uuid::Uuid::new_v4().to_string()),
                task_id: None,
                context_id: None,
            },
        },
    };

    let resp = reqwest::Client::new()
        .post(&url)
        .json(&req)
        .send()
        .await?
        .error_for_status()?
        .json::<JsonRpcResponse>()
        .await?;

    if let Some(err) = resp.error {
        return Err(anyhow::anyhow!("A2A error {}: {}", err.code, err.message));
    }

    let task_result = resp
        .result
        .ok_or_else(|| anyhow::anyhow!("A2A response missing result"))?;
    let text = task_result
        .history
        .iter()
        .find(|m| m.role == A2ARole::Agent)
        .and_then(|m| {
            m.parts.iter().find_map(|p| match p {
                A2APart::Text { text } => Some(text.clone()),
            })
        })
        .ok_or_else(|| anyhow::anyhow!("No text in A2A agent response"))?;

    Ok(text)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use axum::{
        Json, Router,
        routing::{get, post},
    };

    use super::*;

    fn test_card() -> AgentCard {
        AgentCard {
            name: "test-agent".into(),
            description: "A test agent".into(),
            url: None,
            version: Some("1.0.0".into()),
            skills: vec![AgentSkill {
                id: "test".into(),
                name: "Test Skill".into(),
                description: "Does something".into(),
                tags: vec![],
                examples: vec![],
            }],
            capabilities: None,
            default_input_modes: vec!["text/plain".into()],
            default_output_modes: vec!["text/plain".into()],
        }
    }

    #[tokio::test]
    async fn test_discover_parses_card() -> anyhow::Result<()> {
        let card = test_card();
        let card_clone = card.clone();
        let app = Router::new().route(
            "/.well-known/agent-card.json",
            get(move || async move { Json(card_clone.clone()) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        tokio::spawn(async move { axum::serve(listener, app).await.ok() });

        let base_url = format!("http://{}", addr);
        let fetched = discover(&base_url).await?;
        assert_eq!(fetched.name, "test-agent");
        assert_eq!(fetched.description, "A test agent");
        assert_eq!(fetched.skills[0].id, "test");
        Ok(())
    }

    #[tokio::test]
    async fn test_message_send_parses_response() -> anyhow::Result<()> {
        let app = Router::new().route(
            "/",
            post(|Json(req): Json<JsonRpcRequest>| async move {
                let agent_msg = A2AMessage {
                    kind: "message".into(),
                    role: A2ARole::Agent,
                    parts: vec![A2APart::Text {
                        text: "Hello from agent".into(),
                    }],
                    message_id: Some("msg-1".into()),
                    task_id: Some("task-1".into()),
                    context_id: Some("ctx-1".into()),
                };
                let task = Task {
                    kind: "task".into(),
                    id: "task-1".into(),
                    context_id: "ctx-1".into(),
                    status: TaskStatus {
                        state: TaskState::Completed,
                        timestamp: None,
                    },
                    history: vec![agent_msg],
                    artifacts: vec![],
                };
                Json(JsonRpcResponse::success(req.id, task))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        tokio::spawn(async move { axum::serve(listener, app).await.ok() });

        let base_url = format!("http://{}", addr);
        let text = message_send(&base_url, "Say hello").await?;
        assert_eq!(text, "Hello from agent");
        Ok(())
    }

    #[tokio::test]
    async fn test_message_send_handles_error() -> anyhow::Result<()> {
        let app = Router::new().route(
            "/",
            post(|Json(req): Json<JsonRpcRequest>| async move {
                Json(JsonRpcResponse::error(
                    req.id,
                    -32000,
                    "Something went wrong",
                ))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        tokio::spawn(async move { axum::serve(listener, app).await.ok() });

        let base_url = format!("http://{}", addr);
        let result = message_send(&base_url, "do something").await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Something went wrong")
        );
        Ok(())
    }
}
