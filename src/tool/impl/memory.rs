//! `mem_search` and `memory`: an agent's own memory, as tools.
//!
//! Built per agent rather than registered, since each is bound to the [`Memory`] that agent was
//! given: the model never names a folder and cannot name another.

use crate::{
    datatype::Value,
    memory::{Memory, MemoryError, Profile, ProfileEdit, ProfileTarget, Record, Speaker},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
};

/// `mem`'s own default, spelled here for the model-facing description.
const DEFAULT_LIMIT: i64 = 10;

/// Lines are up to 1,500 characters, so 15 with the best match's context stay under the
/// agent's cap on a tool result, past which the middle of the answer is cut.
const MAX_LIMIT: i64 = 15;

/// Lines shown on each side of the best match, and of an `around` id.
const NEIGHBOURS: usize = 2;

pub fn get_mem_search_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("mem_search")
        .description(concat!(
            "Search everything said in past conversations with this user: their messages and ",
            "your replies, word for word, each with the date it was said. ",
            "Matching is by words, not meaning: a line matches if it has any of the query's ",
            "words, and lines with more of them, and rarer ones, come first. So use a few ",
            "specific words (names, things, numbers) rather than a whole question, and try ",
            "other words before concluding something never came up. ",
            "The best match comes with the lines said just before and after it; to read ",
            "around another result, call again with `around` set to its id. ",
            "No match is an empty list. Which memory is searched is fixed; there is no path to give.",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A few words to find (e.g. 'oat milk latte')."
                },
                "limit": {
                    "type": "integer",
                    "description": "How many lines to answer with (default 10, at most 15).",
                    "minimum": 1,
                    "maximum": MAX_LIMIT,
                    "default": DEFAULT_LIMIT
                },
                "sort": {
                    "type": "string",
                    "enum": ["relevance", "newest"],
                    "description": "'relevance' (default): best match first. 'newest': the same matches, most recent first, for what is true now."
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant"],
                    "description": "Only lines the user said, or only your replies. Default: both."
                },
                "around": {
                    "type": "string",
                    "description": "Instead of searching, read the lines around this result id."
                }
            }
        }))
        .build()
}

pub fn get_mem_search_tool_func(memory: Memory) -> ToolFunc {
    crate::tool_func!(async |args: Value, console: &mut ConsoleClient| -> Value
        with [memory = memory.clone()]
    {
        if let Some(id) = args.pointer("/around").and_then(|v| v.as_str()) {
            return match memory.around(console, id, NEIGHBOURS).await {
                Ok(lines) => match lines.iter().find(|r| r.id == id) {
                    Some(center) => crate::to_value!({
                        "count": 1i64,
                        "results": Value::array([with_context(center, &lines)]),
                    }),
                    None => crate::to_value!({ "count": 0i64, "results": [] }),
                },
                Err(e) => failure(e),
            };
        }

        let Some(query) = args.pointer("/query").and_then(|v| v.as_str()) else {
            return validation("missing parameter: query (or around, to read around a result id)".into());
        };
        let limit = args
            .pointer("/limit")
            .and_then(|v| v.as_integer())
            .unwrap_or(DEFAULT_LIMIT)
            .clamp(1, MAX_LIMIT) as usize;
        let role = match args.pointer("/role").and_then(|v| v.as_str()) {
            None => None,
            Some(role) => match Speaker::parse(role) {
                Some(role) => Some(role),
                None => return validation(format!("role must be 'user' or 'assistant', not {role:?}")),
            },
        };

        let mut note = None;
        let found = match memory.search(console, query, limit, role).await {
            Err(MemoryError::Unsupported(_)) if role.is_some() => {
                note = Some("This memory does not record who said a line, so role was ignored.");
                memory.search(console, query, limit, None).await
            }
            found => found,
        };
        let mut found = match found {
            Ok(found) => found,
            Err(e) => return failure(e),
        };
        let best = found.first().map(|r| r.id.clone());
        if args.pointer("/sort").and_then(|v| v.as_str()) == Some("newest") {
            // Fixed-width UTC, so text order is time order; ties keep relevance order.
            found.sort_by(|a, b| b.written_at.cmp(&a.written_at));
        }

        // The best match gets its context; a `mem` without `around` answers without it.
        let context = match &best {
            Some(best) => memory.around(console, best, NEIGHBOURS).await.ok(),
            None => None,
        };
        let results = found.iter().map(|r| match &context {
            Some(lines) if Some(&r.id) == best.as_ref() => with_context(r, lines),
            _ => line(r),
        });
        let mut out = crate::to_value!({
            "count": found.len() as i64,
            "results": Value::array(results),
        });
        if let (Some(note), Some(out)) = (note, out.as_object_mut()) {
            out.insert("note".into(), note.into());
        }
        out
    })
}

pub fn get_memory_tool_desc() -> ToolDesc {
    let action = crate::to_value!({
        "type": "string",
        "enum": ["add", "replace", "remove"],
        "description": "'add' a new entry; 'replace' the entry containing old_text with content; 'remove' the entry containing old_text."
    });
    let target = crate::to_value!({
        "type": "string",
        "enum": ["user", "memory"],
        "description": "'user': who the user is. 'memory': your own notes."
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
            "Keep short notes about this user that are put in your system prompt at the start ",
            "of every future conversation. Two targets: 'user' for who the user is (name, ",
            "work, lasting preferences, how they like to be helped) and 'memory' for your own ",
            "notes (agreements, what to avoid, lessons learned). ",
            "Only what will matter in every conversation: everything said is already saved ",
            "word for word and found with mem_search, so leave out details, one-off tasks and ",
            "progress. When a fact changes, replace its entry instead of adding another. End a ",
            "fact whose date matters with the date, as (YYYY-MM-DD). ",
            "Each target has a size limit; when full, remove or shorten entries in the same ",
            "call. For several changes at once, send them as operations: all are made or none. ",
            "Your system prompt keeps the profile as it was when this conversation started; ",
            "changes show up from the next one.",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "action": action.clone(),
                "target": target.clone(),
                "content": content.clone(),
                "old_text": old_text.clone(),
                "operations": {
                    "type": "array",
                    "description": "Several changes, made in order, all or none; instead of the fields above.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": action,
                            "target": target,
                            "content": content,
                            "old_text": old_text
                        },
                        "required": ["action", "target"]
                    }
                }
            }
        }))
        .build()
}

pub fn get_memory_tool_func(memory: Memory) -> ToolFunc {
    crate::tool_func!(async |args: Value, console: &mut ConsoleClient| -> Value
        with [memory = memory.clone()]
    {
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

        // Edits the files as they are now, not as this conversation's prompt shows them.
        let current = match memory.read_profile(console).await {
            Ok(current) => current,
            Err(e) => return failure(e),
        };
        let edited = match current.edited(&edits, |target| memory.limit(target)) {
            Ok(edited) => edited,
            Err(refusal) => {
                return crate::to_value!({
                    "error": format!("{refusal}. Nothing was changed."),
                    "phase": "validation",
                    "profile": shown(&memory, &current, &[refusal.target]),
                });
            }
        };
        if let Err(e) = memory.write_profile(console, &current, &edited).await {
            return failure(e);
        }
        let changed: Vec<ProfileTarget> = ProfileTarget::ALL
            .into_iter()
            .filter(|&t| edits.iter().any(|e| e.target() == t))
            .collect();
        crate::to_value!({
            "result": "saved",
            "profile": shown(&memory, &edited, &changed),
            "note": "Your system prompt still shows the profile from the start of this conversation; this shows up from the next one.",
        })
    })
}

/// One edit as the model spelled it.
fn edit(op: &Value) -> Result<ProfileEdit, String> {
    let field = |name: &str| op.pointer(&format!("/{name}")).and_then(|v| v.as_str());
    let required = |name: &'static str| {
        field(name)
            .map(str::to_string)
            .ok_or(format!("missing parameter: {name}"))
    };
    let target = required("target")?;
    let target = ProfileTarget::parse(&target)
        .ok_or_else(|| format!("target must be 'user' or 'memory', not {target:?}"))?;
    Ok(match field("action") {
        Some("add") => ProfileEdit::Add {
            target,
            content: required("content")?,
        },
        Some("replace") => ProfileEdit::Replace {
            target,
            old_text: required("old_text")?,
            content: required("content")?,
        },
        Some("remove") => ProfileEdit::Remove {
            target,
            old_text: required("old_text")?,
        },
        other => {
            return Err(format!(
                "action must be add, replace or remove, not {other:?}"
            ));
        }
    })
}

/// `targets` of `profile`: their entries and how full each file is.
fn shown(memory: &Memory, profile: &Profile, targets: &[ProfileTarget]) -> Value {
    Value::object(targets.iter().map(|&target| {
        (
            target.as_str(),
            crate::to_value!({
                "usage": format!("{}/{} chars", profile.used(target), memory.limit(target)),
                "entries": Value::array(profile.entries(target).to_vec()),
            }),
        )
    }))
}

/// A history line as a result: when, who, what.
fn line(record: &Record) -> Value {
    let mut out = crate::to_value!({ "id": record.id.clone(), "date": date(&record.written_at) });
    if let Some(map) = out.as_object_mut() {
        if let Some(role) = record.role {
            map.insert("role".into(), role.as_str().into());
        }
        map.insert("text".into(), record.text.clone().into());
    }
    out
}

/// `record` as a result, with the lines before and after it in `lines`.
fn with_context(record: &Record, lines: &[Record]) -> Value {
    let mut out = line(record);
    if let (Some(at), Some(map)) = (
        lines.iter().position(|r| r.id == record.id),
        out.as_object_mut(),
    ) {
        map.insert("before".into(), Value::array(lines[..at].iter().map(line)));
        map.insert(
            "after".into(),
            Value::array(lines[at + 1..].iter().map(line)),
        );
    }
    out
}

/// `2026-10-21T07:28:19Z` as `2026-10-21 07:28 UTC`; anything else as it is.
fn date(written_at: &str) -> String {
    match (written_at.get(..10), written_at.get(11..16)) {
        (Some(day), Some(time)) if written_at.ends_with('Z') => format!("{day} {time} UTC"),
        _ => written_at.to_string(),
    }
}

fn validation(error: String) -> Value {
    crate::to_value!({ "error": error, "phase": "validation" })
}

fn failure(e: MemoryError) -> Value {
    crate::to_value!({ "error": e.to_string(), "phase": "io" })
}

#[cfg(test)]
mod tests {
    use futures::StreamExt as _;

    use super::*;
    use crate::{console::ConsoleClient, message::Message, test_console, to_value};

    /// One call of a tool, as the agent makes it.
    async fn call(func: &ToolFunc, args: Value, console: &mut ConsoleClient) -> Message {
        func.call(args, "1", console).next().await.unwrap().message
    }

    fn value(msg: &Message) -> Value {
        msg.contents[0].as_value().unwrap().clone()
    }

    fn phase(out: &Value) -> &str {
        out.pointer("/phase").and_then(|v| v.as_str()).unwrap_or("")
    }

    /// A memory folder with a history written by `mem` itself, without roles, so these tests
    /// run on any `mem`.
    async fn filled(console: &mut ConsoleClient, lines: &[&str]) -> Memory {
        let memory = Memory::new(
            tempfile::tempdir()
                .unwrap()
                .keep()
                .to_string_lossy()
                .into_owned(),
        );
        let init = console
            .exec(["mem", "init", &memory.store()], None)
            .await
            .unwrap();
        assert_eq!(init.code, 0, "{}", String::from_utf8_lossy(&init.stderr));
        if !lines.is_empty() {
            let mut argv = vec!["mem".to_string(), "insert".into(), memory.store()];
            argv.extend(lines.iter().map(|l| l.to_string()));
            let insert = console.exec(&argv, None).await.unwrap();
            assert_eq!(
                insert.code,
                0,
                "{}",
                String::from_utf8_lossy(&insert.stderr)
            );
        }
        memory
    }

    /// Whether this console's `mem` has `command` (`["around"]`, `["insert", "--role"]`).
    async fn knows(console: &mut ConsoleClient, command: &[&str]) -> bool {
        let (sub, flag) = (command[0], command.get(1));
        let help = console.exec(["mem", sub, "--help"], None).await.unwrap();
        help.code == 0 && flag.is_none_or(|f| String::from_utf8_lossy(&help.stdout).contains(f))
    }

    fn texts(out: &Value) -> Vec<String> {
        out.pointer("/results")
            .and_then(|v| v.as_array())
            .expect("results")
            .iter()
            .map(|r| r.pointer("/text").unwrap().as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn test_date_reads_as_a_date() {
        assert_eq!(date("2026-10-21T07:28:19Z"), "2026-10-21 07:28 UTC");
        assert_eq!(date("yesterday"), "yesterday");
    }

    #[test]
    fn test_edit_names_what_is_missing() {
        assert!(
            edit(&to_value!({"action": "add", "target": "user"}))
                .unwrap_err()
                .contains("content")
        );
        assert!(
            edit(&to_value!({"action": "remove", "target": "memory"}))
                .unwrap_err()
                .contains("old_text")
        );
        assert!(
            edit(&to_value!({"action": "add", "target": "me", "content": "x"}))
                .unwrap_err()
                .contains("target")
        );
        assert!(
            edit(&to_value!({"action": "append", "target": "user", "content": "x"}))
                .unwrap_err()
                .contains("action")
        );
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_mem_search_answers_with_dated_lines_and_context() {
        let mut console = test_console().await;
        let memory = filled(
            &mut console,
            &[
                "I drink tea",
                "I switched to oat milk",
                "Noted, oat milk",
                "Thanks",
            ],
        )
        .await;
        let func = get_mem_search_tool_func(memory);

        let out = value(&call(&func, to_value!({ "query": "switched" }), &mut console).await);
        assert_eq!(texts(&out), ["I switched to oat milk"]);
        let best = out.pointer("/results/0").unwrap();
        assert!(
            best.pointer("/date")
                .unwrap()
                .as_str()
                .unwrap()
                .ends_with(" UTC")
        );
        assert!(best.pointer("/id").is_some());
        // Context only where `mem` can read around a line.
        if knows(&mut console, &["around"]).await {
            let before = best.pointer("/before").expect("context before the match");
            assert_eq!(before.as_array().unwrap().len(), 1);
            assert_eq!(best.pointer("/after").unwrap().as_array().unwrap().len(), 2);
        } else {
            assert!(best.pointer("/before").is_none(), "{best:?}");
        }
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_mem_search_honors_limit_and_sort() {
        let mut console = test_console().await;
        let memory = filled(&mut console, &["tea, tea and more tea"]).await;
        // A second apart, since `mem` stamps lines to the second.
        tokio::time::sleep(std::time::Duration::from_millis(1_100)).await;
        let later = console
            .exec(["mem", "insert", &memory.store(), "I drink tea"], None)
            .await
            .unwrap();
        assert_eq!(later.code, 0, "{}", String::from_utf8_lossy(&later.stderr));
        let func = get_mem_search_tool_func(memory);

        let out = value(
            &call(
                &func,
                to_value!({ "query": "tea", "limit": 1 }),
                &mut console,
            )
            .await,
        );
        assert_eq!(texts(&out), ["tea, tea and more tea"]);

        let out = value(
            &call(
                &func,
                to_value!({ "query": "tea", "sort": "newest" }),
                &mut console,
            )
            .await,
        );
        assert_eq!(texts(&out), ["I drink tea", "tea, tea and more tea"]);
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_mem_search_refuses_what_it_cannot_run() {
        let mut console = test_console().await;
        let memory = filled(&mut console, &[]).await;
        let func = get_mem_search_tool_func(memory);

        let out = value(&call(&func, to_value!({}), &mut console).await);
        assert_eq!(phase(&out), "validation");
        let out = value(
            &call(
                &func,
                to_value!({ "query": "x", "role": "tool" }),
                &mut console,
            )
            .await,
        );
        assert_eq!(phase(&out), "validation");

        let missing = Memory::new("/nowhere/at/all");
        let func = get_mem_search_tool_func(missing);
        let out = value(&call(&func, to_value!({ "query": "x" }), &mut console).await);
        assert_eq!(phase(&out), "io");
        assert!(
            out.pointer("/error")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("/nowhere/at/all"),
            "{out:?}"
        );
    }

    /// A role filter on a `mem` without one still answers, and says so.
    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_mem_search_by_role_answers_on_any_mem() {
        let mut console = test_console().await;
        let memory = filled(&mut console, &[]).await;
        let roles = knows(&mut console, &["insert", "--role"]).await;
        if roles {
            memory
                .append(
                    &mut console,
                    [
                        (Speaker::User, "I drink tea"),
                        (Speaker::Assistant, "Tea it is"),
                    ],
                )
                .await
                .unwrap();
        } else {
            let insert = console
                .exec(["mem", "insert", &memory.store(), "I drink tea"], None)
                .await
                .unwrap();
            assert_eq!(
                insert.code,
                0,
                "{}",
                String::from_utf8_lossy(&insert.stderr)
            );
        }
        let func = get_mem_search_tool_func(memory);

        let out = value(
            &call(
                &func,
                to_value!({ "query": "tea", "role": "user" }),
                &mut console,
            )
            .await,
        );
        assert_eq!(texts(&out), ["I drink tea"], "{out:?}");
        assert_eq!(out.pointer("/note").is_some(), !roles, "{out:?}");
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_memory_tool_edits_the_profile() {
        let mut console = test_console().await;
        let memory = filled(&mut console, &[]).await;
        let func = get_memory_tool_func(memory.clone());

        let out = value(
            &call(
                &func,
                to_value!({ "action": "add", "target": "user", "content": "Name is Minsu" }),
                &mut console,
            )
            .await,
        );
        assert_eq!(
            out.pointer("/result").unwrap().as_str().unwrap(),
            "saved",
            "{out:?}"
        );
        assert_eq!(
            out.pointer("/profile/user/usage")
                .unwrap()
                .as_str()
                .unwrap(),
            "13/2000 chars"
        );

        let out = value(
            &call(
                &func,
                to_value!({ "operations": [
                    { "action": "replace", "target": "user", "old_text": "Minsu", "content": "Name is Minsu Kim" },
                    { "action": "add", "target": "memory", "content": "Answer briefly" }
                ] }),
                &mut console,
            )
            .await,
        );
        assert_eq!(
            out.pointer("/result").unwrap().as_str().unwrap(),
            "saved",
            "{out:?}"
        );
        let profile = memory.read_profile(&mut console).await.unwrap();
        assert_eq!(profile.user, ["Name is Minsu Kim"]);
        assert_eq!(profile.memory, ["Answer briefly"]);
    }

    /// A refusal shows the entries as they stand, so the model can retry from them.
    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_memory_tool_refusal_shows_the_entries() {
        let mut console = test_console().await;
        let memory = filled(&mut console, &[]).await;
        let func = get_memory_tool_func(memory);

        call(
            &func,
            to_value!({ "action": "add", "target": "user", "content": "Drinks tea" }),
            &mut console,
        )
        .await;
        let out = value(
            &call(
                &func,
                to_value!({ "action": "remove", "target": "user", "old_text": "coffee" }),
                &mut console,
            )
            .await,
        );
        assert_eq!(phase(&out), "validation");
        assert_eq!(
            out.pointer("/profile/user/entries/0")
                .unwrap()
                .as_str()
                .unwrap(),
            "Drinks tea"
        );

        let out = value(
            &call(
                &func,
                to_value!({ "action": "add", "target": "user", "content": "x".repeat(2_000) }),
                &mut console,
            )
            .await,
        );
        assert_eq!(phase(&out), "validation");
        let error = out.pointer("/error").unwrap().as_str().unwrap();
        assert!(error.contains("over its limit"), "{error}");
    }
}
