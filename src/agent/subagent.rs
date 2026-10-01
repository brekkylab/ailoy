use std::sync::Arc;

use futures::StreamExt as _;
use tokio::sync::Mutex;
use virtx::console::ConsoleClient;

use crate::{
    agent::{Agent, AgentCard, AgentSpec, AgentState},
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
};

/// Prefix on every subagent tool name, so subagent calls are identifiable without extra
/// metadata. The card name (and so `source_agent`) is unprefixed.
pub const SUBAGENT_TOOL_PREFIX: &str = "subagent_";

/// The prefixed tool name for a subagent card.
pub fn subagent_tool_name(card: &AgentCard) -> String {
    format!("{}{}", SUBAGENT_TOOL_PREFIX, card.name)
}

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

    ToolDescBuilder::new(subagent_tool_name(card))
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

/// A [`ToolFunc`] that builds a fresh sub-agent from `spec` per call and runs it for one turn,
/// streaming every [`MessageOutput`], then a `Role::Tool` message with its last answer, all
/// tagged with [`AgentCard::name`] as `source_agent`.
///
/// `provider` is re-resolved from [`get_agent_providers`](crate::agent::get_agent_providers)
/// per call, so it must stay registered for the parent agent's lifetime.
pub fn get_subagent_tool_func(
    spec: AgentSpec,
    provider: String,
    console: Arc<Mutex<Option<ConsoleClient>>>,
) -> ToolFunc {
    let card_name = spec.card.as_ref().map(|c| c.name.clone());

    ToolFunc::new(move |args: Value, id: String| {
        let spec = spec.clone();
        let provider = provider.clone();
        let console = console.clone();
        let card_name = card_name.clone();

        async_stream::stream! {
            let task = match args
                .as_object()
                .and_then(|o| o.get("task"))
                .and_then(|v| v.as_str())
            {
                Some(v) => v.to_string(),
                None => {
                    yield MessageOutput {
                        message: Message::new(Role::Tool)
                            .with_contents([Part::value(Value::string("Error: expected 'task' string field in arguments"))])
                            .with_id(id),
                        finish_reason: FinishReason::Stop {},
                        usage: None,
                        depth: None,
                        source_agent: card_name,
                    };
                    return;
                }
            };

            // Shares the parent's console slot so both see the same filesystem.
            let state = AgentState::new().with_console_slot(console);
            let mut agent = match Agent::try_with_provider_and_state(spec, &provider, state).await {
                Ok(a) => a,
                Err(e) => {
                    yield MessageOutput {
                        message: Message::new(Role::Tool)
                            .with_contents([Part::value(Value::string(format!("Error: {e}")))])
                            .with_id(id),
                        finish_reason: FinishReason::Stop {},
                        usage: None,
                        depth: None,
                        source_agent: card_name,
                    };
                    return;
                }
            };

            let query = Message::new(Role::User).with_contents([Part::text(task)]);
            let mut last_answer = String::new();

            {
                let mut strm = agent.run(query);
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
                            // Forwarded whole to keep the source_agent the sub-agent stamped.
                            yield output;
                        }
                        Err(e) => {
                            yield MessageOutput {
                                message: Message::new(Role::Tool)
                                    .with_contents([Part::value(Value::string(format!("Error: {e}")))])
                                    .with_id(id.clone()),
                                finish_reason: FinishReason::Stop {},
                                usage: None,
                                depth: None,
                                source_agent: card_name.clone(),
                            };
                            return;
                        }
                    }
                }
            }

            yield MessageOutput {
                message: Message::new(Role::Tool)
                    .with_contents([Part::value(Value::string(last_answer))])
                    .with_id(id),
                finish_reason: FinishReason::Stop {},
                usage: None,
                depth: None,
                source_agent: card_name,
            };
        }
        .boxed()
    })
}
