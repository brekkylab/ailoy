pub(crate) mod a2a;

use std::sync::Arc;

use futures::StreamExt as _;
use tokio::sync::Mutex as TokioMutex;

use super::{ToolAsyncFunc, ToolKind, ToolRuntime, ToolStreamingFunc};
use crate::{
    agent::{AgentRuntime, TurnEvent},
    message::{Part, Role, StreamingToolOutput, ToolDescBuilder, ToolResultDelta},
};

/// Build a [`ToolRuntime`] that delegates to an in-process [`AgentRuntime`].
///
/// The generated tool accepts a single `task` string parameter and forwards it
/// to the wrapped agent. Intermediate assistant messages from the subagent are
/// yielded as [`ToolResultDelta`] items so callers can display progress. The
/// subagent's final assistant response becomes the tool result.
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

    let tool_name = name.to_string();
    let f: Arc<ToolStreamingFunc> = Arc::new(move |args| {
        let agent = agent.clone();
        let tool_name = tool_name.clone();
        Box::pin(async move {
            let task = args
                .pointer("/task")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let msg =
                crate::message::Message::new(Role::User).with_contents(vec![Part::text(task)]);

            // Stream the subagent's turn:
            // - Intermediate AssistantMessages → StreamingToolOutput::Delta (UI progress)
            // - Final AssistantMessage → StreamingToolOutput::Result (definitive output)
            let strm: futures::stream::BoxStream<'static, StreamingToolOutput> =
                Box::pin(async_stream::stream! {
                    let mut guard = agent.lock().await;
                    let mut inner_strm = guard.stream_turn(msg);
                    let mut final_text: Option<String> = None;

                    while let Some(event) = inner_strm.next().await {
                        match event {
                            Ok(TurnEvent::AssistantMessage(output)) => {
                                let text = output.message.contents.iter()
                                    .find_map(|p: &Part| p.as_text())
                                    .unwrap_or_default()
                                    .to_string();

                                if !text.is_empty() {
                                    // Every intermediate message is a Delta
                                    yield StreamingToolOutput::Delta(ToolResultDelta {
                                        tool_call_id: None,
                                        tool_name: tool_name.clone(),
                                        content: Part::text(text.clone()),
                                    });
                                    final_text = Some(text);
                                }
                            }
                            Err(e) => {
                                yield StreamingToolOutput::Result(
                                    format!("Error: {}", e).into()
                                );
                                return;
                            }
                            _ => {}
                        }
                    }

                    // Emit the definitive result using the last assistant text
                    let result_value = final_text
                        .unwrap_or_default()
                        .into();
                    yield StreamingToolOutput::Result(result_value);
                });

            strm
        })
    });

    ToolRuntime::new_streaming_with_kind(desc, f, ToolKind::Subagent)
}

/// Build a [`ToolRuntime`] that delegates to a remote A2A agent.
///
/// The generated tool accepts a single `task` string parameter, sends it to the
/// remote agent via JSON-RPC `message/send`, and returns the response text as
/// the tool result. Registered as an `Async` tool (single result, potentially
/// long-running).
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

    let f: Arc<ToolAsyncFunc> = Arc::new(move |args| {
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

    ToolRuntime::new_async_with_kind(desc, f, ToolKind::Subagent)
}
