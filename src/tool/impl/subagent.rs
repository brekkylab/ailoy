use std::sync::Arc;

use futures::StreamExt as _;

use crate::{
    agent::{Agent, AgentCard, AgentProvider, AgentSpec},
    datatype::Value,
    message::{FinishReason, Message, Part, Role},
    runenv::RunEnv,
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

/// Build a one-shot [`ToolFunc`] that materialises a fresh sub-agent from the
/// supplied [`AgentSpec`] every time the tool is invoked, runs it for one turn,
/// then drops it.
///
/// The closure captures `spec`, `provider`, and `runenv` by clone, so the
/// produced [`ToolFunc`] is independent of any parent [`Agent`]: the parent
/// keeps it in its `tools` map alongside ordinary tools, with no special
/// dispatch needed.  The sub-spec is taken as-is — its
/// [`AgentSpec::skills`] and [`AgentSpec::skill_root`] are respected
/// verbatim, so a sub-spec is portable across parents.
///
/// The returned function:
/// 1. Streams every [`MessageOutput`](crate::message::MessageOutput) produced by
///    the sub-agent during its turn.
/// 2. Emits a final `Role::Tool` message whose value content is the sub-agent's
///    last assistant answer.
pub fn get_subagent_tool_func(
    spec: AgentSpec,
    provider: AgentProvider,
    runenv: Arc<dyn RunEnv>,
) -> ToolFunc {
    tool_func!(stream |args: Value, id: String| -> Message {
        let spec = spec.clone();
        let provider = provider.clone();
        let runenv = runenv.clone();
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

            let mut agent = match Agent::try_with_provider_and_runenv(spec, &provider, runenv) {
                Ok(a) => a,
                Err(e) => {
                    yield Message::new(Role::Tool)
                        .with_contents([Part::value(Value::string(format!("Error building subagent: {e}")))])
                        .with_id(id);
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
