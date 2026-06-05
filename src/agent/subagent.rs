use futures::StreamExt as _;

use crate::{
    agent::{Agent, AgentCard, AgentProvider, AgentSpec, rt::AgentParts},
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role},
    runenv::SharedMachine,
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
/// Resolution against the [`AgentProvider`] happens **once at outer call time**:
/// the model is captured as a cheap [`LangModelFactory`], the sub-agent's tools
/// and any nested sub-sub-agent tool funcs are pre-resolved, and the initial
/// system message is pre-rendered.  The returned closure captures only owned,
/// `Clone`-able values — no provider clone per invocation.
///
/// The returned function:
/// 1. Streams every [`MessageOutput`](crate::message::MessageOutput) produced by
///    the sub-agent during its turn, with `source_agent` already set to the
///    sub-agent's [`AgentCard::name`].
/// 2. Emits a final `Role::Tool` message whose value content is the sub-agent's
///    last assistant answer, also tagged with `source_agent`.
pub fn get_subagent_tool_func(
    spec: AgentSpec,
    provider: &AgentProvider,
    machine: SharedMachine,
) -> anyhow::Result<ToolFunc> {
    // Capture the card name once; it's needed on every synthesised MessageOutput.
    let card_name = spec.card.as_ref().map(|c| c.name.clone());

    // Pre-resolve everything that depends on the provider so the closure can
    // capture cheap, owned values only — no AgentProvider in the capture set.
    let parts = AgentParts::from_spec(&spec, provider, &machine)?;

    Ok(ToolFunc::new(move |args: Value, id: String| {
        let parts = parts.clone();
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

            // Build a fresh Agent for this invocation directly from the
            // pre-resolved parts; no provider lookup required.
            let mut agent = Agent::from_resolved_parts(parts, machine);

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
    }))
}
