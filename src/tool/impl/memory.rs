//! `memory` over a [`Memory`]: an agent's own notes, as a tool.
//!
//! Not a built-in: a registry entry resolves from the [`AgentSpec`](crate::agent::AgentSpec)
//! alone, but this also needs *which* file, one a caller handed to one agent. So the file is
//! captured in the closure rather than taken as an argument: the model never names a memory
//! file and cannot name another. Only an agent given a memory file gets this tool — see
//! [`Agent::try_with_state`](crate::agent::Agent::try_with_state).

use crate::{
    datatype::Value,
    memory::{EditError, Memory, MemoryEdit},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
};

pub fn get_memory_tool_desc() -> ToolDesc {
    let action = crate::to_value!({
        "type": "string",
        "enum": ["add", "replace", "remove"],
        "description": "'add' a new fact as its own entry; 'replace' the entry containing old_text with content, when its fact changed; 'remove' the entry containing old_text."
    });
    let content = crate::to_value!({
        "type": "string",
        "description": "One fact: the entry to add, or the whole new text of the replaced entry."
    });
    let old_text = crate::to_value!({
        "type": "string",
        "description": "A short piece of the entry to replace or remove, long enough that only that entry contains it."
    });
    ToolDescBuilder::new("memory")
        .description(concat!(
            "Keep short notes that are put in your system prompt at the start of every future ",
            "conversation: who the user is (name, work, lasting preferences, how they like to be ",
            "helped) and your own notes (agreements, what to avoid, lessons learned). ",
            "Only what will matter in every conversation: leave out details, one-off tasks and ",
            "progress. Each entry is one fact: a new fact is a new entry, even about the same ",
            "user. When a fact changes, replace that fact's entry instead of adding another, and ",
            "never rewrite an entry to fit another fact into it. End a fact whose date matters ",
            "with the date, as (YYYY-MM-DD). ",
            "The memory has a size limit; when it is full, make room in the same call: shorten ",
            "entries first, and remove one only when shortening is not enough, then tell the user ",
            "what you removed. For several changes at once, send them as operations: all are made or none. ",
            "Your system prompt keeps the memory as it was when this conversation started; ",
            "changes show up from the next one.",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "action": action.clone(),
                "content": content.clone(),
                "old_text": old_text.clone(),
                "operations": {
                    "type": "array",
                    "description": "Several changes, made in order, all or none; instead of the fields above.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": action,
                            "content": content,
                            "old_text": old_text
                        },
                        "required": ["action"]
                    }
                }
            }
        }))
        .build()
}

/// The `memory` tool bound to `memory`.
pub fn get_memory_tool_func(memory: Memory) -> ToolFunc {
    crate::tool_func!(async |args: Value| -> Value with [memory = memory.clone()] {
        let edits = match args.pointer("/operations") {
            Some(Value::Array(ops)) => ops.iter().map(edit).collect::<Result<Vec<_>, _>>(),
            Some(_) => Err("operations must be an array of objects".to_string()),
            None => edit(&args).map(|edit| vec![edit]),
        };
        let edits = match edits {
            Ok(edits) if !edits.is_empty() => edits,
            Ok(_) => return validation("operations is empty".into()),
            Err(e) => return validation(e),
        };

        let limit = memory.limit();
        let written = tokio::task::spawn_blocking(move || memory.edit(&edits)).await;
        match written {
            Ok(Ok(entries)) => crate::to_value!({
                "result": "saved",
                "memory": shown(&entries, limit),
                "note": "Your system prompt still shows the memory from the start of this conversation; this shows up from the next one.",
            }),
            Ok(Err(EditError::Refused(refusal))) => crate::to_value!({
                "error": format!("{refusal}. Nothing was changed."),
                "phase": "validation",
                "memory": shown(&refusal.entries, limit),
            }),
            Ok(Err(e)) => crate::to_value!({ "error": e.to_string(), "phase": "io" }),
            Err(e) => crate::to_value!({ "error": e.to_string(), "phase": "io" }),
        }
    })
}

/// One edit as the model spelled it.
fn edit(op: &Value) -> Result<MemoryEdit, String> {
    let field = |name: &str| op.pointer(&format!("/{name}")).and_then(|v| v.as_str());
    let required = |name: &'static str| {
        field(name)
            .map(str::to_string)
            .ok_or(format!("missing parameter: {name}"))
    };
    Ok(match field("action") {
        Some("add") => MemoryEdit::Add {
            content: required("content")?,
        },
        Some("replace") => MemoryEdit::Replace {
            old_text: required("old_text")?,
            content: required("content")?,
        },
        Some("remove") => MemoryEdit::Remove {
            old_text: required("old_text")?,
        },
        other => {
            return Err(format!(
                "action must be add, replace or remove, not {other:?}"
            ));
        }
    })
}

/// The memory's entries and how full the file is.
fn shown(entries: &[String], limit: usize) -> Value {
    crate::to_value!({
        "usage": format!("{}/{limit} chars", crate::memory::used(entries)),
        "entries": Value::array(entries.to_vec()),
    })
}

fn validation(error: String) -> Value {
    crate::to_value!({ "error": error, "phase": "validation" })
}

#[cfg(test)]
mod tests {
    use futures::StreamExt as _;

    use super::*;
    use crate::{datatype::Value, message::Message, to_value};

    fn value(msg: &Message) -> Value {
        msg.contents[0].as_value().unwrap().clone()
    }

    /// One call of a tool that needs no console.
    async fn call_pure(func: &ToolFunc, args: Value) -> Value {
        let output = func
            .call_pure(args, "1")
            .expect("a pure tool")
            .next()
            .await
            .unwrap();
        value(&output.message)
    }

    fn fresh() -> Memory {
        let dir = tempfile::tempdir().unwrap().keep();
        Memory::new(dir).with_limit(100)
    }

    #[tokio::test]
    async fn test_memory_tool_edits_the_file() {
        let memory = fresh();
        let func = get_memory_tool_func(memory.clone());

        let out = call_pure(
            &func,
            to_value!({ "action": "add", "content": "Name is Minsu" }),
        )
        .await;
        assert_eq!(
            out.pointer("/result").unwrap().as_str(),
            Some("saved"),
            "{out:?}"
        );
        assert_eq!(
            out.pointer("/memory/usage").unwrap().as_str(),
            Some("13/100 chars")
        );

        let out = call_pure(
            &func,
            to_value!({ "operations": [
                { "action": "replace", "old_text": "Minsu", "content": "Name is Minsu Kim" },
                { "action": "add", "content": "Answer briefly" }
            ] }),
        )
        .await;
        assert_eq!(
            out.pointer("/result").unwrap().as_str(),
            Some("saved"),
            "{out:?}"
        );
        assert_eq!(
            memory.read().unwrap(),
            ["Name is Minsu Kim", "Answer briefly"]
        );
    }

    /// A refusal changes nothing and shows the entries as they stand, so the model can retry
    /// from them.
    #[tokio::test]
    async fn test_memory_tool_refusal_shows_the_entries() {
        let memory = fresh();
        let func = get_memory_tool_func(memory.clone());
        call_pure(
            &func,
            to_value!({ "action": "add", "content": "Drinks tea" }),
        )
        .await;

        for args in [
            to_value!({ "action": "remove", "old_text": "coffee" }),
            to_value!({ "action": "add", "content": "x".repeat(100) }),
        ] {
            let out = call_pure(&func, args).await;
            assert_eq!(
                out.pointer("/phase").unwrap().as_str(),
                Some("validation"),
                "{out:?}"
            );
            assert_eq!(
                out.pointer("/memory/entries/0").unwrap().as_str(),
                Some("Drinks tea")
            );
        }
        assert_eq!(memory.read().unwrap(), ["Drinks tea"]);

        for args in [
            to_value!({}),
            to_value!({ "action": "add" }),
            to_value!({ "action": "append", "content": "x" }),
            to_value!({ "operations": [] }),
        ] {
            let out = call_pure(&func, args.clone()).await;
            assert_eq!(
                out.pointer("/phase").unwrap().as_str(),
                Some("validation"),
                "{args:?}"
            );
        }
    }
}
