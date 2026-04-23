use std::sync::Arc;

use futures::StreamExt as _;
use tokio::sync::Mutex;

use crate::{
    agent::{Agent, AgentCard},
    datatype::Value,
    message::{FinishReason, Message, Part, Role, ToolDescBuilder},
    tool::{Tool, ToolContext, ToolFunc},
};

/// Creates a [`Tool`] that wraps `agent` as a callable sub-agent tool.
///
/// The tool accepts a single `string` argument (the task for the sub-agent) and:
/// 1. Streams all [`MessageOutput`] items produced during the sub-agent's turn as
///    intermediate outputs. The outer agent's [`stream_turn`] assigns these `depth + 1`.
/// 2. Emits a final `Role::Tool` [`MessageOutput`] whose text content is the sub-agent's
///    last assistant answer. The outer agent assigns this `depth 0` and pushes it to history.
pub fn make_subagent_tool(card: AgentCard, agent: Arc<Mutex<Agent>>) -> Tool {
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
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task description to send to the sub-agent"
                }
            },
            "required": ["task"]
        }))
        .build();

    let f = ToolFunc::new(Box::new(move |args: Value, ctx: ToolContext| {
        let id = ctx.id;
        let agent = agent.clone();
        Box::pin(async_stream::stream! {
            let task = match args
                .as_object()
                .and_then(|o| o.get("task"))
                .and_then(|v| v.as_str())
            {
                Some(v) => v.to_string(),
                None => {
                    yield Message::new(Role::Tool)
                        .with_contents([Part::value(Value::string("Error: expected 'task' string field in arguments"))])
                        .with_id(id);
                    return;
                }
            };

            let query = Message::new(Role::User).with_contents([Part::text(task)]);
            let mut last_answer = String::new();

            {
                let mut agent_guard = agent.lock().await;
                let mut strm = agent_guard.run(query);
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
                            yield output.message;
                        }
                        Err(e) => {
                            yield Message::new(Role::Tool)
                                .with_contents([Part::value(Value::string(format!("Error: {e}")))])
                                .with_id(id);
                            return;
                        }
                    }
                }
            } // drop strm and agent_guard

            // Emit the final tool response — the outer agent assigns this depth 0.
            // Use Part::value so provider codecs marshal it as a plain string output.
            yield Message::new(Role::Tool)
                .with_contents([Part::value(Value::string(last_answer))])
                .with_id(id);
        }) as futures::stream::BoxStream<'static, Message>
    }));

    Tool::new(desc, Arc::new(f))
}
