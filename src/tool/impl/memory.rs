//! `mem_search` and `mem_insert` over a [`Memory`], and `memory` over a [`MemoryFile`]: an
//! agent's own memory, as tools.
//!
//! Not built-ins: a registry entry resolves from the [`AgentSpec`](crate::agent::AgentSpec)
//! alone, but these also need *which* store, one a caller handed to one agent. So the store is
//! captured in the closure rather than taken as an argument: the model never names a memory
//! file and cannot name another. Only an agent given a memory gets these tools — see
//! [`Agent::try_with_state`](crate::agent::Agent::try_with_state).

use crate::{
    datatype::Value,
    memory::{EditError, Memory, MemoryEdit, MemoryFile},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
};

/// Matches `mem`'s own default; spelled here because it goes in the model-facing
/// description, which must agree with the command.
const DEFAULT_LIMIT: i64 = 10;

pub fn get_mem_search_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("mem_search")
        .description(concat!(
            "Recall what you have written down about this user and your past work with them. ",
            "Answers with the stored memories nearest the query, nearest first — this is a ",
            "similarity search over remembered statements, not a substring match, so ask it ",
            "in words rather than in a pattern. ",
            "No memory near the query is an empty list: the store was read and holds nothing ",
            "about this. Which store is searched is fixed; there is no path to give.",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to recall, as a question or a phrase (e.g. 'what does the user drink?')"
                },
                "limit": {
                    "type": "integer",
                    "description": "How many memories to answer with (default 10).",
                    "minimum": 1,
                    "default": DEFAULT_LIMIT
                }
            },
            "required": ["query"]
        }))
        .build()
}

/// The `mem_search` tool bound to `memory`.
pub fn get_mem_search_tool_func(memory: Memory) -> ToolFunc {
    crate::tool_func!(async |args: Value, console: &mut ConsoleClient| -> Value
        with [memory = memory.clone()]
    {
        let Some(query) = args.pointer("/query").and_then(|v| v.as_str()) else {
            return crate::to_value!({
                "error": "missing required parameter: query",
                "phase": "validation",
            });
        };
        let limit = args
            .pointer("/limit")
            .and_then(|v| v.as_integer())
            .map(|n| n.max(1))
            .unwrap_or(DEFAULT_LIMIT) as usize;

        let found = match memory.search(console, query).await {
            Ok(found) => found,
            Err(e) => {
                return crate::to_value!({
                    "error": format!("{e}"),
                    "phase": "io",
                });
            }
        };

        // Truncated here rather than via `mem -n`: results are nearest-first, so the
        // front of the answer is what a smaller `-n` would have returned.
        let count = found.len().min(limit);
        crate::to_value!({
            "memories": crate::datatype::Value::array(found.into_iter().take(limit)),
            "count": count as i64,
        })
    })
}

pub fn get_mem_insert_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("mem_insert")
        .description(concat!(
            "Write down something worth remembering after this conversation ends — a ",
            "preference, a decision, a fact about the user or their project. ",
            "Each memory is stored word for word, so write one that will still make sense ",
            "read on its own months from now: a whole statement rather than a fragment, ",
            "with the subject named instead of 'they' or 'it'. ",
            "Nothing is rewritten or summarized on the way in, and nothing is ",
            "de-duplicated — deciding what is worth keeping is yours. ",
            "Which store is written is fixed; there is no path to give.",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "description": "The memories to store, one statement per item (e.g. 'The user prefers oat milk in coffee').",
                    "items": { "type": "string" }
                }
            },
            "required": ["memories"]
        }))
        .build()
}

/// The `mem_insert` tool for `memory`.
pub fn get_mem_insert_tool_func(memory: Memory) -> ToolFunc {
    crate::tool_func!(async |args: Value, console: &mut ConsoleClient| -> Value
        with [memory = memory.clone()]
    {
        // Arrays only: a model that packed two memories into one string would have them
        // stored verbatim as one. The refusal names the expected shape instead.
        let Some(memories) = args.pointer("/memories").and_then(|v| v.as_array()) else {
            return crate::to_value!({
                "error": "missing required parameter: memories (an array of strings)",
                "phase": "validation",
            });
        };

        let mut texts: Vec<String> = Vec::with_capacity(memories.len());
        for memory in memories {
            let Some(text) = memory.as_str() else {
                return crate::to_value!({
                    "error": "every entry in memories must be a string",
                    "phase": "validation",
                });
            };
            texts.push(text.to_string());
        }

        let written = match memory.insert(console, &texts).await {
            Ok(written) => written,
            Err(e) => {
                return crate::to_value!({
                    "error": format!("{e}"),
                    "phase": "io",
                });
            }
        };

        crate::to_value!({
            "stored": crate::datatype::Value::array(written.clone()),
            "count": written.len() as i64,
        })
    })
}

pub fn get_memory_tool_desc() -> ToolDesc {
    let action = crate::to_value!({
        "type": "string",
        "enum": ["add", "replace", "remove"],
        "description": "'add' a new entry; 'replace' the entry containing old_text with content; 'remove' the entry containing old_text."
    });
    let content = crate::to_value!({
        "type": "string",
        "description": "The entry to add, or the whole new text of the replaced entry."
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
            "progress. When a fact changes, replace its entry instead of adding another. End a ",
            "fact whose date matters with the date, as (YYYY-MM-DD). ",
            "The memory has a size limit; when it is full, remove or shorten entries in the same ",
            "call. For several changes at once, send them as operations: all are made or none. ",
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
pub fn get_memory_tool_func(memory: MemoryFile) -> ToolFunc {
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
    use crate::{
        console::ConsoleClient, datatype::Value, message::Message, test_console, to_value,
    };

    /// A store with memories in it, made and filled by `mem` itself.
    async fn filled(console: &mut ConsoleClient, memories: &[&str]) -> Memory {
        let dir = tempfile::tempdir().unwrap();
        let path = dir
            .keep()
            .join("notes.sqlite")
            .to_string_lossy()
            .to_string();

        let init = console.exec(["mem", "init", &path], None).await.unwrap();
        assert_eq!(init.code, 0, "{}", String::from_utf8_lossy(&init.stderr));

        if !memories.is_empty() {
            let mut argv = vec!["mem".to_string(), "insert".to_string(), path.clone()];
            argv.extend(memories.iter().map(|m| m.to_string()));
            let insert = console.exec(&argv, None).await.unwrap();
            assert_eq!(
                insert.code,
                0,
                "{}",
                String::from_utf8_lossy(&insert.stderr)
            );
        }

        Memory::new(path)
    }

    /// One call of a tool, as the agent makes it.
    async fn call(func: &ToolFunc, args: Value, console: &mut ConsoleClient) -> Message {
        func.call(args, "1", console).next().await.unwrap().message
    }

    fn value(msg: &Message) -> Value {
        msg.contents[0].as_value().unwrap().clone()
    }

    fn strings(value: &Value, at: &str) -> Vec<String> {
        value
            .pointer(at)
            .and_then(|v| v.as_array())
            .expect("an array")
            .iter()
            .map(|v| v.as_str().expect("a string").to_string())
            .collect()
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_mem_search_tool_answers_with_the_nearest_memory() {
        let mut console = test_console().await;
        let memory = filled(
            &mut console,
            &["User drinks tea", "User switched to oat milk"],
        )
        .await;
        let func = get_mem_search_tool_func(memory);

        let out = value(&call(&func, to_value!({ "query": "oat milk" }), &mut console).await);
        assert_eq!(strings(&out, "/memories"), ["User switched to oat milk"]);
        assert_eq!(out.pointer("/count").unwrap().as_integer().unwrap(), 1);
    }

    /// The bound the model asked for is the bound it gets.
    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_mem_search_tool_honors_limit() {
        let mut console = test_console().await;
        let memory = filled(&mut console, &["User drinks tea", "User drinks coffee"]).await;
        let func = get_mem_search_tool_func(memory);

        let out = value(
            &call(
                &func,
                to_value!({ "query": "what does the user drink", "limit": 1 }),
                &mut console,
            )
            .await,
        );
        assert_eq!(strings(&out, "/memories").len(), 1);
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_mem_search_tool_with_nothing_near_is_an_empty_list() {
        let mut console = test_console().await;
        let memory = filled(&mut console, &["User drinks tea"]).await;
        let func = get_mem_search_tool_func(memory);

        let out = value(&call(&func, to_value!({ "query": "almond" }), &mut console).await);
        assert!(strings(&out, "/memories").is_empty(), "{out:?}");
        assert_eq!(out.pointer("/count").unwrap().as_integer().unwrap(), 0);
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_mem_search_tool_without_a_query_is_a_validation_error() {
        let mut console = test_console().await;
        let memory = filled(&mut console, &[]).await;
        let func = get_mem_search_tool_func(memory);

        let out = value(&call(&func, to_value!({}), &mut console).await);
        assert_eq!(
            out.pointer("/phase").unwrap().as_str().unwrap(),
            "validation"
        );
    }

    /// Written by the tool, and read back by the other one: the pair works on one store.
    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_mem_insert_tool_writes_what_it_was_given() {
        let mut console = test_console().await;
        let memory = filled(&mut console, &[]).await;
        let insert = get_mem_insert_tool_func(memory.clone());
        let search = get_mem_search_tool_func(memory);

        let out = value(
            &call(
                &insert,
                to_value!({ "memories": ["User switched to oat milk", "User drinks tea"] }),
                &mut console,
            )
            .await,
        );
        assert_eq!(
            strings(&out, "/stored"),
            ["User switched to oat milk", "User drinks tea"]
        );

        let found = value(&call(&search, to_value!({ "query": "oat milk" }), &mut console).await);
        assert_eq!(strings(&found, "/memories"), ["User switched to oat milk"]);
    }

    /// A single string is not one memory here — the shape is named in the refusal rather
    /// than guessed at, because a guess would store two statements as one.
    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_mem_insert_tool_refuses_a_bare_string() {
        let mut console = test_console().await;
        let memory = filled(&mut console, &[]).await;
        let func = get_mem_insert_tool_func(memory);

        let out = value(
            &call(
                &func,
                to_value!({ "memories": "User drinks tea" }),
                &mut console,
            )
            .await,
        );
        assert_eq!(
            out.pointer("/phase").unwrap().as_str().unwrap(),
            "validation"
        );
    }

    /// A store that is not there is `mem`'s sentence, arriving as an `io` failure the
    /// model can read rather than as a tool that returned nothing.
    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_mem_insert_tool_reports_a_missing_store() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir
            .path()
            .join("missing.sqlite")
            .to_string_lossy()
            .to_string();

        let mut console = test_console().await;
        let func = get_mem_insert_tool_func(Memory::new(&path));

        let out = value(
            &call(
                &func,
                to_value!({ "memories": ["User drinks tea"] }),
                &mut console,
            )
            .await,
        );
        assert_eq!(out.pointer("/phase").unwrap().as_str().unwrap(), "io");
        assert!(
            out.pointer("/error")
                .unwrap()
                .as_str()
                .unwrap()
                .contains(&path),
            "{out:?}"
        );
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

    fn memory_file() -> MemoryFile {
        let dir = tempfile::tempdir().unwrap().keep();
        MemoryFile::new(dir.join("memory.md")).with_limit(100)
    }

    #[tokio::test]
    async fn test_memory_tool_edits_the_file() {
        let memory = memory_file();
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
        let memory = memory_file();
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
