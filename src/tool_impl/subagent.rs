use std::sync::Arc;

use futures::StreamExt as _;
use tokio::sync::Mutex;

use crate::{
    agent::AgentRuntime,
    datatype::Value,
    message::{AgentCard, FinishReason, Message, MessageOutput, Part, Role, ToolDescBuilder},
    tool::{ToolFunc, ToolRuntime},
};

/// Creates a [`ToolRuntime`] that wraps `agent` as a callable sub-agent tool.
///
/// The tool accepts a single `string` argument (the task for the sub-agent) and:
/// 1. Streams all [`MessageOutput`] items produced during the sub-agent's turn as
///    intermediate outputs. The outer agent's [`stream_turn`] assigns these `depth + 1`.
/// 2. Emits a final `Role::Tool` [`MessageOutput`] whose text content is the sub-agent's
///    last assistant answer. The outer agent assigns this `depth 0` and pushes it to history.
pub fn make_subagent_tool(card: AgentCard, agent: Arc<Mutex<AgentRuntime>>) -> ToolRuntime {
    let description = if card.skills.is_empty() {
        card.description
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

    let f = ToolFunc::Stream(Box::new(move |id: String, args: Value| {
        let agent = agent.clone();
        Box::pin(async_stream::stream! {
            let task = match args.as_str() {
                Some(v) => v.to_string(),
                None => {
                    yield MessageOutput {
                        depth: None,
                        message: Message::new(Role::Tool)
                            .with_contents([Part::text("Error: expected string argument")])
                            .with_id(id),
                        finish_reason: FinishReason::Stop {},
                    };
                    return;
                }
            };

            let query = Message::new(Role::User).with_contents([Part::text(task)]);
            let mut last_answer = String::new();

            {
                let mut agent_guard = agent.lock().await;
                let mut strm = agent_guard.stream_turn(query);
                while let Some(result) = strm.next().await {
                    match result {
                        Ok(output) => {
                            // Track the final assistant text to use as the tool result.
                            if output.message.role == Role::Assistant
                                && matches!(output.finish_reason, FinishReason::Stop {})
                            {
                                last_answer = output
                                    .message
                                    .contents
                                    .iter()
                                    .filter_map(|p| p.as_text())
                                    .collect::<Vec<_>>()
                                    .join("");
                            }
                            yield output;
                        }
                        Err(e) => {
                            yield MessageOutput {
                                depth: None,
                                message: Message::new(Role::Tool)
                                    .with_contents([Part::text(format!("Error: {e}"))])
                                    .with_id(id),
                                finish_reason: FinishReason::Stop {},
                            };
                            return;
                        }
                    }
                }
            } // drop strm and agent_guard

            // Emit the final tool response — the outer agent assigns this depth 0.
            yield MessageOutput {
                depth: None,
                message: Message::new(Role::Tool)
                    .with_contents([Part::text(last_answer)])
                    .with_id(id),
                finish_reason: FinishReason::Stop {},
            };
        }) as futures::stream::BoxStream<'static, MessageOutput>
    }));

    ToolRuntime::new(desc, Arc::new(f))
}
