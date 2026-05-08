use std::sync::Arc;

use futures::StreamExt as _;
use tokio::sync::Mutex;

use crate::{
    agent::{Agent, AgentCard},
    datatype::Value,
    message::{FinishReason, Message, Part, Role},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

/// Build the [`ToolDesc`] for a sub-agent from its agent card.
pub fn get_subagent_tool_desc(card: &AgentCard) -> ToolDesc {
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

    ToolDescBuilder::new(&card.name)
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
        .build()
}

/// Build a [`ToolFunc`] that runs a sub-agent turn for the supplied agent.
///
/// The returned function:
/// 1. Streams every [`MessageOutput`](crate::message::MessageOutput) produced by
///    the sub-agent during its turn.
/// 2. Emits a final `Role::Tool` message whose value content is the sub-agent's
///    last assistant answer.
pub fn get_subagent_tool_func(agent: Arc<Mutex<Agent>>) -> ToolFunc {
    tool_func!(stream |args: Value, id: String| -> Message {
        let agent = agent.clone();
        let id = id.clone();
        async_stream::stream! {
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
            }

            yield Message::new(Role::Tool)
                .with_contents([Part::value(Value::string(last_answer))])
                .with_id(id);
        }
    })
}
