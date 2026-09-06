//! `mem_search` and `mem_insert` — an agent's own memory, as two tools.
//!
//! # Why these are not built-ins
//!
//! Every tool in [`builtins`](super::builtins) is registered by name in a
//! [`ToolProvider`](crate::tool::ToolProvider) and resolved from an
//! [`AgentSpec`](crate::agent::AgentSpec), which works because the spec carries
//! everything the tool needs. These two need one thing more: *which* store, and that is
//! not a name in a registry — it is a [`Memory`] a caller handed to one agent.
//!
//! So the store is captured in the closure rather than asked for in the arguments. The
//! model never names a memory file, cannot name a different one, and does not have to be
//! told in the prompt which one is its own. An agent that was given a memory gets these
//! two tools for it; one that was not, does not — see
//! [`Agent::try_with_provider_and_state`](crate::agent::Agent::try_with_provider_and_state).

use crate::{
    memory::Memory,
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
};

/// How many memories a search answers with when the caller does not say.
///
/// `mem`'s own default is the same number. Spelled here because it goes in the
/// description the model reads, and a description that disagreed with the command would
/// be worse than none.
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

/// The `mem_search` tool for `memory`.
///
/// One store per tool, captured here — see the module docs for why the model is not asked
/// for a path.
pub fn get_mem_search_tool_func(memory: Memory) -> ToolFunc {
    crate::tool_func!(async |args: Value, console: &mut Console| -> Value
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

        // Bounded here rather than by `mem -n`: the limit is the model's to raise and the
        // store answers in nearest-first order either way, so taking the front of the
        // answer is the same list a smaller `-n` would have returned.
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
    crate::tool_func!(async |args: Value, console: &mut Console| -> Value
        with [memory = memory.clone()]
    {
        // An array, and only an array. A bare string would be one memory and is tempting
        // to accept, but a model that meant two and wrote them into one string would have
        // that stored as a single memory, word for word — which is exactly what this tool
        // promises and exactly the wrong result. The refusal names the shape instead.
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

#[cfg(test)]
mod tests {
    use futures::StreamExt as _;

    use super::*;
    use crate::{console::Console, datatype::Value, message::Message, test_console, to_value};

    /// A store with memories in it, made and filled by `mem` itself.
    async fn filled(console: &mut Console, memories: &[&str]) -> Memory {
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
    async fn call(func: &ToolFunc, args: Value, console: &mut Console) -> Message {
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
}
