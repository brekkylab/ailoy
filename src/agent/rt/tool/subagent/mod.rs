pub(crate) mod a2a;

use std::sync::Arc;

use tokio::sync::Mutex as TokioMutex;

use super::{ToolFunc, ToolKind, ToolRuntime};
use crate::{
    agent::AgentRuntime,
    message::{Message, Part, Role, ToolDescBuilder},
};

/// Build a [`ToolRuntime`] that delegates to an in-process [`AgentRuntime`].
///
/// The generated tool accepts a single `task` string parameter and forwards it to the
/// wrapped agent, returning the agent's first text response as the tool result.
pub(crate) fn build_in_memory_subagent_tool(
    name: &str,
    description: &str,
    agent: Arc<TokioMutex<AgentRuntime>>,
) -> ToolRuntime {
    let desc = ToolDescBuilder::new(name)
        .description(description)
        .parameters(serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task or question to send to this agent"
                }
            },
            "required": ["task"]
        }))
        .build();

    let f: Arc<ToolFunc> = Arc::new(move |args| {
        let agent = agent.clone();
        Box::pin(async move {
            let task = args
                .pointer("/task")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let msg = Message::new(Role::User).with_contents(vec![Part::text(task)]);
            let mut guard = agent.lock().await;
            // Reset history to system message only before each invocation so that subagent
            // calls remain stateless and history does not accumulate across tool calls.
            let initial_history: Vec<Message> = guard
                .get_history()
                .into_iter()
                .filter(|m| m.role == Role::System)
                .collect();
            guard.set_history(initial_history);
            match guard.run(msg).await {
                Ok(result) => {
                    let text = result
                        .contents
                        .iter()
                        .find_map(|p: &Part| p.as_text())
                        .unwrap_or_default()
                        .to_string();
                    text.into()
                }
                Err(e) => format!("Error: {}", e).into(),
            }
        })
    });

    ToolRuntime::new_with_kind(desc, f, ToolKind::Subagent)
}

/// Build a [`ToolRuntime`] that delegates to a remote A2A agent.
///
/// The generated tool accepts a single `task` string parameter, sends it to the remote
/// agent via JSON-RPC `message/send`, and returns the response text as the tool result.
pub(crate) fn build_a2a_subagent_tool(card: &a2a::AgentCard, url: String) -> ToolRuntime {
    let desc = ToolDescBuilder::new(&card.name)
        .description(&card.description)
        .parameters(serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task or question to send to this agent"
                }
            },
            "required": ["task"]
        }))
        .build();

    let f: Arc<ToolFunc> = Arc::new(move |args| {
        let url = url.clone();
        Box::pin(async move {
            let task = args
                .pointer("/task")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            match a2a::message_send(&url, &task).await {
                Ok(result) => result.into(),
                Err(e) => format!("Error: {}", e).into(),
            }
        })
    });

    ToolRuntime::new_with_kind(desc, f, ToolKind::Subagent)
}
