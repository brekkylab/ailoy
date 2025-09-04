use std::{any::Any, collections::BTreeMap};

use async_stream::try_stream;
use futures::stream::Stream;

use crate::{
    cache::{Cache, CacheClaim, CacheContents, CacheEntry, CacheProgress, TryFromCache},
    dyn_maybe_send,
    model::{
        LanguageModel,
        local::{ChatTemplate, LanguageModelInferencer, Tokenizer},
    },
    utils::{BoxFuture, BoxStream},
    value::{FinishReason, Message, MessageOutput, Part, Role, ToolDesc},
};

#[derive(Debug)]
pub struct LocalLanguageModel {
    chat_template: ChatTemplate,

    tokenizer: Tokenizer,

    // The inferencer performs mutable operations such as KV cache updates, so the mutex is applied.
    inferencer: LanguageModelInferencer,
}

impl LocalLanguageModel {
    pub async fn try_new(
        model_name: impl Into<String>,
    ) -> impl Stream<Item = Result<CacheProgress<Self>, String>> + 'static {
        let model_name = model_name.into();
        Cache::new().try_create::<LocalLanguageModel>(model_name)
    }

    pub fn enable_reasoning(&self) {
        self.chat_template.enable_reasoning();
    }

    pub fn disable_reasoning(&self) {
        self.chat_template.disable_reasoning();
    }
}

impl LanguageModel for LocalLanguageModel {
    fn run<'a>(
        self: &'a mut Self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> BoxStream<'a, Result<MessageOutput, String>> {
        let strm = try_stream! {
            let prompt = self.chat_template.apply(msgs, tools, true)?;
            let input_tokens = self.tokenizer.encode(&prompt, true)?;
            #[cfg(target_family = "wasm")]
            self.inferencer.prefill(&input_tokens).await;
            #[cfg(not(target_family = "wasm"))]
            self.inferencer.prefill(&input_tokens);
            let mut last_token = *input_tokens.last().unwrap();
            let mut agg_tokens = Vec::<u32>::new();
            let mut count = 0;
            let mut mode = "content".to_owned();
            let mut finish_reason = FinishReason::Stop;

            yield MessageOutput::new().with_delta(Message::with_role(Role::Assistant));

            // @jhlee: TODO remove hard-coded token names
            loop {
                count += 1;
                if count > 16384 {
                    Err("Too long assistant message. It may be infinite loop".to_owned())?;
                }
                #[cfg(target_family = "wasm")]
                let new_token = self.inferencer.decode(last_token).await;
                #[cfg(not(target_family = "wasm"))]
                let new_token = self.inferencer.decode(last_token);
                agg_tokens.push(new_token);
                last_token = new_token;
                let s = self.tokenizer.decode(agg_tokens.as_slice(), false)?;
                if s.ends_with("�") {
                    continue;
                }
                agg_tokens.clear();

                if s == "<|im_end|>" {
                    yield MessageOutput::new().with_finish_reason(finish_reason);
                    break;
                } else if s == "<tool_call>" {
                    mode = "tool_call".to_owned();
                    continue;
                } else if s == "</tool_call>" {
                    mode = "content".to_owned();
                    finish_reason = FinishReason::ToolCalls;
                    continue;
                } else if s == "<think>" {
                    mode = "reasoning".to_owned();
                    continue;
                } else if s == "</think>" {
                    mode = "content".to_owned();
                    continue;
                } else {
                    let delta = if mode == "content" {
                        Message::new().with_contents(vec![Part::Text(s)])
                    } else if mode == "reasoning" {
                        Message::new().with_reasoning(s)
                    } else if mode == "tool_call" {
                        Message::new().with_tool_calls(vec![Part::FunctionString(s)])
                    } else {
                        unreachable!();
                    };
                    yield MessageOutput::new().with_delta(delta);
                }
            }
            return;
        };
        Box::pin(strm)
    }
}

impl TryFromCache for LocalLanguageModel {
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> BoxFuture<'static, Result<CacheClaim, String>> {
        let key = key.as_ref().to_owned();
        Box::pin(async move {
            let mut chat_template = ChatTemplate::claim_files(cache.clone(), &key).await?;
            let mut tokenizer = Tokenizer::claim_files(cache.clone(), &key).await?;
            let mut inferncer = LanguageModelInferencer::claim_files(cache.clone(), &key).await?;
            let ctx: Box<dyn_maybe_send!(Any)> = Box::new([
                chat_template
                    .entries
                    .iter()
                    .map(|v| v.clone())
                    .collect::<Vec<_>>(),
                tokenizer
                    .entries
                    .iter()
                    .map(|v| v.clone())
                    .collect::<Vec<_>>(),
                inferncer
                    .entries
                    .iter()
                    .map(|v| v.clone())
                    .collect::<Vec<_>>(),
            ]);
            let mut rv = Vec::new();
            rv.append(&mut chat_template.entries);
            rv.append(&mut tokenizer.entries);
            rv.append(&mut inferncer.entries);
            Ok(CacheClaim::with_ctx(rv, ctx))
        })
    }

    fn try_from_contents(mut contents: CacheContents) -> BoxFuture<'static, Result<Self, String>>
    where
        Self: Sized,
    {
        Box::pin(async move {
            let (ct_entries, tok_entries, inf_entries) = match contents.ctx.take() {
                Some(ctx_any) => match ctx_any.downcast::<[Vec<CacheEntry>; 3]>() {
                    Ok(boxed) => {
                        let [a, b, c] = *boxed;
                        (a, b, c)
                    }
                    Err(_) => return Err("contents.ctx is not [Vec<CacheEntry>; 3]".into()),
                },
                None => return Err("contents.ctx is None".into()),
            };

            let chat_template = {
                let mut files = BTreeMap::new();
                for k in ct_entries {
                    let v = contents.entries.remove(&k).unwrap();
                    files.insert(k, v);
                }
                let contents = CacheContents {
                    root: contents.root.clone(),
                    entries: files,
                    ctx: None,
                };
                ChatTemplate::try_from_contents(contents).await?
            };
            let tokenizer = {
                let mut files = BTreeMap::new();
                for k in tok_entries {
                    let v = contents.entries.remove(&k).unwrap();
                    files.insert(k, v);
                }
                let contents = CacheContents {
                    root: contents.root.clone(),
                    entries: files,
                    ctx: None,
                };
                Tokenizer::try_from_contents(contents).await?
            };
            let inferencer = {
                let mut files = BTreeMap::new();
                for k in inf_entries {
                    let v = contents.entries.remove(&k).unwrap();
                    files.insert(k, v);
                }
                let contents = CacheContents {
                    root: contents.root.clone(),
                    entries: files,
                    ctx: None,
                };
                LanguageModelInferencer::try_from_contents(contents).await?
            };

            Ok(LocalLanguageModel {
                chat_template,
                tokenizer,
                inferencer,
            })
        })
    }
}

#[cfg(any(target_family = "unix", target_family = "windows"))]
#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn infer_simple_chat() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{MessageAggregator, Role};

        let cache = crate::cache::Cache::new();
        let key = "Qwen/Qwen3-0.6B";

        let mut model_strm = Box::pin(cache.try_create::<LocalLanguageModel>(key));
        let mut model: Option<LocalLanguageModel> = None;
        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            println!(
                "{} ({} / {})",
                progress.comment, progress.current_task, progress.total_task
            );
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        let mut model = model.unwrap();
        model.enable_reasoning();
        let msgs = vec![
            Message::with_role(Role::System)
                .with_contents(vec![Part::Text("You are an assistant.".to_owned())]),
            Message::with_role(Role::User)
                .with_contents(vec![Part::Text("Hi what's your name?".to_owned())]),
            // Message::with_role(Role::Assistant)
            //     .with_reasoning("\nOkay, the user asked, \"Hi what's your name?\" So I need to respond appropriately.\n\nFirst, I should acknowledge their question. Since I'm an AI assistant, I don't have a name, but I can say something like, \"Hi! I'm an AI assistant. How can I assist you today?\" That shows I'm here to help. I should keep it friendly and open. Let me make sure the response is polite and professional.\n")
            //     .with_contents(vec![Part::Text(
            //         "Hi! I'm an AI assistant. How can I assist you today? 😊".to_owned(),
            //     )]),
            // Message::with_role(Role::User)
            //     .with_contents(vec![Part::Text("Who made you?".to_owned())]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(msgs, Vec::new());
        while let Some(out) = strm.next().await {
            let out = out.unwrap();
            println!("{:?}", out);
            if let Some(msg) = agg.update(out) {
                println!("{:?}", msg);
            }
        }
    }

    #[tokio::test]
    async fn infer_tool_call() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{MessageAggregator, Role, ToolDesc, ToolDescArg};

        let cache = crate::cache::Cache::new();
        let key = "Qwen/Qwen3-0.6B";
        let mut model_strm = Box::pin(cache.try_create::<LocalLanguageModel>(key));
        let mut model: Option<LocalLanguageModel> = None;
        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            println!(
                "{} ({} / {})",
                progress.comment, progress.current_task, progress.total_task
            );
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        let mut model = model.unwrap();
        model.disable_reasoning();
        let tools = vec![ToolDesc::new(
            "temperature",
            "Get current temperature",
            ToolDescArg::new_object().with_properties(
                [
                    (
                        "location",
                        ToolDescArg::new_string().with_desc("The city name"),
                    ),
                    (
                        "unit",
                        ToolDescArg::new_string().with_enum(["Celcius", "Fernheit"]),
                    ),
                ],
                ["location", "unit"],
            ),
            Some(
                ToolDescArg::new_number().with_desc("Null if the given city name is unavailable."),
            ),
        )];
        let msgs = vec![
            Message::with_role(Role::User).with_contents(vec![Part::Text(
                "How much hot currently in Dubai?".to_owned(),
            )]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(msgs, tools);
        let mut assistant_msg: Option<Message> = None;
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            println!("{:?}", delta);
            if let Some(msg) = agg.update(delta) {
                assistant_msg = Some(msg);
            }
        }
        let assistant_msg = assistant_msg.unwrap();
        println!("Assistant message: {:?}", assistant_msg);
        let tc = assistant_msg.tool_calls.get(0).unwrap();
        println!("Tool call: {:?}", tc);
    }

    #[tokio::test]
    async fn infer_result_from_tool_call() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{MessageAggregator, Role, ToolDesc, ToolDescArg};

        let cache = crate::cache::Cache::new();
        let key = "Qwen/Qwen3-0.6B";
        let mut model_strm = Box::pin(cache.try_create::<LocalLanguageModel>(key));
        let mut model: Option<LocalLanguageModel> = None;
        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            println!(
                "{} ({} / {})",
                progress.comment, progress.current_task, progress.total_task
            );
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        let mut model = model.unwrap();
        model.disable_reasoning();
        let tools = vec![ToolDesc::new(
            "temperature",
            "Get current temperature",
            ToolDescArg::new_object().with_properties(
                [
                    (
                        "location",
                        ToolDescArg::new_string().with_desc("The city name"),
                    ),
                    (
                        "unit",
                        ToolDescArg::new_string().with_enum(["Celcius", "Fernheit"]),
                    ),
                ],
                ["location", "unit"],
            ),
            Some(
                ToolDescArg::new_number().with_desc("Null if the given city name is unavailable."),
            ),
        )];
        let msgs = vec![
            Message::with_role(Role::User)
                .with_contents([Part::new_text("How much hot currently in Dubai?".to_owned())]),
            Message::with_role(Role::Assistant)
                .with_contents([Part::new_text("\n\n")])
                .with_tool_calls([Part::new_function_string("\n{\"name\": \"temperature\", \"arguments\": {\"location\": \"Dubai\", \"unit\": \"Celcius\"}}\n")]),
            Message::with_role(Role::Tool("temperature".into(), None)).with_contents([Part::new_text("40")])
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(msgs, tools);
        let mut assistant_msg: Option<Message> = None;
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                assistant_msg = Some(msg);
            }
        }
        let assistant_msg = assistant_msg.unwrap();
        println!("Assistant message: {:?}", assistant_msg);
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
#[cfg(test)]
mod tests {
    use futures::StreamExt as _;
    use wasm_bindgen_test::*;

    use super::*;
    use crate::value::MessageAggregator;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn infer_simple_chat() {
        let cache = crate::cache::Cache::new();
        let key = "Qwen/Qwen3-0.6B";
        let mut model_strm = Box::pin(cache.try_create::<LocalLanguageModel>(key));
        let mut model: Option<LocalLanguageModel> = None;

        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            web_sys::console::log_1(
                &format!(
                    "{} ({} / {})",
                    progress.comment, progress.current_task, progress.total_task
                )
                .into(),
            );
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        let mut model = model.unwrap();
        model.enable_reasoning();
        let msgs = vec![
            Message::with_role(Role::System)
                .with_contents(vec![Part::Text("You are an assistant.".to_owned())]),
            Message::with_role(Role::User)
                .with_contents(vec![Part::Text("Hi what's your name?".to_owned())]),
            // Message::with_role(Role::Assistant)
            //     .with_reasoning("\nOkay, the user asked, \"Hi what's your name?\" So I need to respond appropriately.\n\nFirst, I should acknowledge their question. Since I'm an AI assistant, I don't have a name, but I can say something like, \"Hi! I'm an AI assistant. How can I assist you today?\" That shows I'm here to help. I should keep it friendly and open. Let me make sure the response is polite and professional.\n")
            //     .with_contents(vec![Part::Text(
            //         "Hi! I'm an AI assistant. How can I assist you today? 😊".to_owned(),
            //     )]),
            // Message::with_role(Role::User)
            //     .with_contents(vec![Part::Text("Who made you?".to_owned())]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(msgs, Vec::new());
        while let Some(out) = strm.next().await {
            let out = out.unwrap();
            web_sys::console::log_1(&format!("{:?}", out).into());
            if let Some(msg) = agg.update(out) {
                web_sys::console::log_1(&format!("{:?}", msg).into());
            }
        }
    }
}
