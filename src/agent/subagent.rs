use std::sync::Arc;

use futures::StreamExt as _;

use crate::{
    agent::{Agent, AgentCard, AgentSpec, AgentState},
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role},
    runenv::Console,
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
};

/// Prefix applied to every subagent tool's descriptor name so callers can
/// identify subagent tool calls without additional metadata.
/// The card name (and therefore `source_agent`) is left unchanged.
pub const SUBAGENT_TOOL_PREFIX: &str = "subagent_";

/// Returns the tool-descriptor name for a subagent card (prefixed form).
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

/// Build a one-shot [`ToolFunc`] that materialises a fresh sub-agent from the
/// supplied [`AgentSpec`] every time the tool is invoked, runs it for one turn,
/// then drops it.
///
/// The closure captures only owned, cheaply-cloneable values: the spec, the
/// agent-provider name (a [`String`]), the shared machine, and the card name.
/// The provider name is re-resolved against [`get_agent_providers`] on every
/// invocation via [`Agent::try_with_provider_and_state`], so the registry
/// entry must stay live for the parent agent's lifetime.
///
/// The returned function:
/// 1. Streams every [`MessageOutput`](crate::message::MessageOutput) produced by
///    the sub-agent during its turn, with `source_agent` already set to the
///    sub-agent's [`AgentCard::name`].
/// 2. Emits a final `Role::Tool` message whose value content is the sub-agent's
///    last assistant answer, also tagged with `source_agent`.
pub fn get_subagent_tool_func(
    spec: AgentSpec,
    provider: String,
    machine: Arc<dyn Console>,
) -> ToolFunc {
    // Capture the card name once; it's needed on every synthesised MessageOutput.
    let card_name = spec.card.as_ref().map(|c| c.name.clone());

    ToolFunc::new(move |args: Value, id: String| {
        let spec = spec.clone();
        let provider = provider.clone();
        let machine = machine.clone();
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

            // Build a fresh Agent for this invocation; the parent's machine is
            // shared so filesystem state stays consistent across the call.
            let mut state = AgentState::new();
            state.runenv = machine;
            let mut agent = match Agent::try_with_provider_and_state(spec, &provider, state) {
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
                            // Yield the full MessageOutput so source_agent (stamped by the
                            // sub-agent's own Agent::run) is preserved through to the parent.
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
