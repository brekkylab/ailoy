use std::{any::Any, collections::BTreeMap, sync::Arc};

use async_stream::try_stream;
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::{
    cache::{Cache, CacheClaim, CacheContents, CacheEntry, CacheProgress, TryFromCache},
    dyn_maybe_send,
    model::{
        ChatTemplate, InferenceConfig, LangModelInference, LanguageModelInferencer, ThinkEffort,
        Tokenizer,
    },
    utils::{BoxFuture, BoxStream},
    value::{
        FinishReason, Message, MessageDelta, MessageOutput, PartDelta, PartDeltaFunction, ToolDesc,
    },
};

struct Request {
    msgs: Vec<Message>,
    tools: Vec<ToolDesc>,
    config: InferenceConfig,
    tx_resp: mpsc::UnboundedSender<Result<MessageOutput, String>>,
}

#[derive(Clone, Debug)]
pub struct LocalLangModel {
    tx: Arc<mpsc::Sender<Request>>,
}

impl LocalLangModel {
    pub async fn try_new(model: impl Into<String>) -> Result<Self, String> {
        let cache = Cache::new();
        let mut strm = Box::pin(cache.try_create::<LocalLangModel>(model));
        while let Some(v) = strm.next().await {
            if let Some(result) = v?.result {
                return Ok(result);
            }
        }
        unreachable!()
    }

    pub fn try_new_stream<'a>(
        model: impl Into<String>,
    ) -> BoxStream<'a, Result<CacheProgress<Self>, String>> {
        let cache = Cache::new();
        Box::pin(cache.try_create::<Self>(model))
    }
}

impl TryFromCache for LocalLangModel {
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> BoxFuture<'static, Result<CacheClaim, String>> {
        LocalLangModelImpl::claim_files(cache, key)
    }

    fn try_from_contents(contents: CacheContents) -> BoxFuture<'static, Result<Self, String>> {
        Box::pin(async move {
            let mut body = LocalLangModelImpl::try_from_contents(contents).await?;
            let (tx, mut rx) = mpsc::channel(1);

            let fut = async move {
                while let Some(req) = rx.recv().await {
                    let Request {
                        msgs,
                        tools,
                        config,
                        tx_resp,
                    } = req;
                    let mut strm = body.infer(msgs, tools, config);
                    while let Some(resp) = strm.next().await {
                        if tx_resp.send(resp).is_err() {
                            break;
                        }
                    }
                }
            };
            #[cfg(not(target_arch = "wasm32"))]
            tokio::spawn(fut);
            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_futures::spawn_local(fut);

            Ok(Self { tx: Arc::new(tx) })
        })
    }
}

impl LangModelInference for LocalLangModel {
    fn infer<'a>(
        &'a mut self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: InferenceConfig,
    ) -> crate::utils::BoxStream<'a, Result<MessageOutput, String>> {
        let (tx_resp, mut rx_resp) = tokio::sync::mpsc::unbounded_channel();
        let req = Request {
            msgs,
            tools,
            config,
            tx_resp,
        };
        let tx = self.tx.clone();
        let strm = async_stream::stream! {
            tx.send(req).await.unwrap();
            while let Some(resp) = rx_resp.recv().await {
                yield resp;
            }
        };
        Box::pin(strm)
    }
}

#[derive(Debug)]
struct LocalLangModelImpl {
    chat_template: ChatTemplate,

    tokenizer: Tokenizer,

    inferencer: LanguageModelInferencer,
}

impl LocalLangModelImpl {
    pub fn infer<'a>(
        &'a mut self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: InferenceConfig,
    ) -> BoxStream<'a, Result<MessageOutput, String>> {
        let strm = try_stream! {
            match config.think_effort {
                ThinkEffort::Disable => {
                    self.chat_template.disable_reasoning();
                },
                ThinkEffort::Enable | ThinkEffort::Low | ThinkEffort::Medium | ThinkEffort::High => {
                    self.chat_template.enable_reasoning();
                },
            }
            let prompt = self.chat_template.apply(msgs, tools, true)?;
            let input_tokens = self.tokenizer.encode(&prompt, true)?;

            {
                #[cfg(target_family = "wasm")]
                self.inferencer.prefill(&input_tokens).await;
                #[cfg(not(target_family = "wasm"))]
                self.inferencer.prefill(&input_tokens);
            }

            let mut last_token = *input_tokens.last().unwrap();
            let mut agg_tokens = Vec::<u32>::new();
            let mut count = 0;
            let mut mode = "content".to_owned();
            let mut finish_reason = FinishReason::Stop();

            // @jhlee: TODO remove hard-coded token names
            loop {
                count += 1;
                if count > config.max_tokens.unwrap_or(16384) {
                    yield MessageOutput{delta: MessageDelta::new(), finish_reason: Some(FinishReason::Length())};
                    break;
                }

                {
                    #[cfg(target_family = "wasm")]
                    let new_token = self.inferencer.decode(last_token).await;
                    #[cfg(not(target_family = "wasm"))]
                    let new_token = self.inferencer.decode(last_token);

                    agg_tokens.push(new_token);
                    last_token = new_token;
                }

                let s = self.tokenizer.decode(agg_tokens.as_slice(), false)?;
                if s.ends_with("�") {
                    continue;
                }
                agg_tokens.clear();

                if s == "<|im_end|>" {
                    yield MessageOutput{delta: MessageDelta::new(), finish_reason: Some(finish_reason)};
                    break;
                } else if s == "<tool_call>" {
                    mode = "tool_call".to_owned();
                    continue;
                } else if s == "</tool_call>" {
                    mode = "content".to_owned();
                    finish_reason = FinishReason::ToolCall();
                    continue;
                } else if s == "<think>" {
                    mode = "reasoning".to_owned();
                    continue;
                } else if s == "</think>" {
                    mode = "content".to_owned();
                    continue;
                } else {
                    let delta = if mode == "content" {
                        MessageDelta::new().with_contents([PartDelta::Text{text: s}])
                    } else if mode == "reasoning" {
                        MessageDelta::new().with_thinking(s)
                    } else if mode == "tool_call" {
                        MessageDelta::new().with_tool_calls([PartDelta::Function{id: None, f: PartDeltaFunction::Verbatim(s)}])
                    } else {
                        unreachable!();
                    };
                    yield MessageOutput{delta, finish_reason: None};
                }
            }
            return;
        };
        Box::pin(strm)
    }
}

impl TryFromCache for LocalLangModelImpl {
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

            Ok(Self {
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
    use crate::value::{Delta, Part};

    #[tokio::test]
    async fn infer_simple_chat() {
        use futures::StreamExt;

        use super::*;
        use crate::value::Role;

        let cache = crate::cache::Cache::new();
        let key = "Qwen/Qwen3-0.6B";

        let mut model_strm = Box::pin(cache.try_create::<LocalLangModel>(key));
        let mut model: Option<LocalLangModel> = None;
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
        let msgs = vec![
            Message::new(Role::System).with_contents([Part::Text {
                text: "You are an assistant.".to_owned(),
            }]),
            Message::new(Role::User).with_contents([Part::Text {
                text: "Hi what's your name?".to_owned(),
            }]),
            // Message::with_role(Role::Assistant)
            //     .with_reasoning("\nOkay, the user asked, \"Hi what's your name?\" So I need to respond appropriately.\n\nFirst, I should acknowledge their question. Since I'm an AI assistant, I don't have a name, but I can say something like, \"Hi! I'm an AI assistant. How can I assist you today?\" That shows I'm here to help. I should keep it friendly and open. Let me make sure the response is polite and professional.\n")
            //     .with_contents(vec![Part::Text(
            //         "Hi! I'm an AI assistant. How can I assist you today? 😊".to_owned(),
            //     )]),
            // Message::with_role(Role::User)
            //     .with_contents(vec![Part::Text("Who made you?".to_owned())]),
        ];
        let mut delta = MessageDelta::new();
        let mut strm = model.infer(msgs, Vec::new(), InferenceConfig::default());
        while let Some(out) = strm.next().await {
            let out = out.unwrap();
            crate::utils::log::debug(format!("{:?}", out));
            delta = delta.aggregate(out.delta).unwrap();
        }
        crate::utils::log::info(format!("{:?}", delta.finish().unwrap()));
    }

    // #[tokio::test]
    // async fn infer_tool_call() {
    //     use futures::StreamExt;

    //     use crate::value::{MessageAggregator, Role, ToolDesc};
    //     use local_language_model::*;

    //     let cache = crate::cache::Cache::new();
    //     let key = "Qwen/Qwen3-0.6B";
    //     let mut model_strm = Box::pin(cache.try_create::<LocalLangModel>(key));
    //     let mut model: Option<LocalLangModel> = None;
    //     while let Some(progress) = model_strm.next().await {
    //         let mut progress = progress.unwrap();
    //         println!(
    //             "{} ({} / {})",
    //             progress.comment, progress.current_task, progress.total_task
    //         );
    //         if progress.current_task == progress.total_task {
    //             model = progress.result.take();
    //         }
    //     }
    //     let mut model = model.unwrap();
    //     model.disable_reasoning();
    //     let tools = vec![
    //         ToolDesc::new(
    //             "temperature".into(),
    //             "Get current temperature".into(),
    //             json!({
    //                 "type": "object",
    //                 "properties": {
    //                     "location": {
    //                         "type": "string",
    //                         "description": "The city name"
    //                     },
    //                     "unit": {
    //                         "type": "string",
    //                         "enum": ["Celsius", "Fahrenheit"]
    //                     }
    //                 },
    //                 "required": ["location", "unit"]
    //             }),
    //             Some(json!({
    //                 "type": "number",
    //                 "description": "Null if the given city name is unavailable.",
    //                 "nullable": true,
    //             })),
    //         )
    //         .unwrap(),
    //     ];
    //     let msgs = vec![
    //         Message::new()
    //             .with_role(Role::User)
    //             .with_contents(vec![Part::Text(
    //                 "How much hot currently in Dubai?".to_owned(),
    //             )]),
    //     ];
    //     let mut agg = MessageAggregator::new();
    //     let mut strm = model.run(msgs, tools);
    //     let mut assistant_msg: Option<Message> = None;
    //     while let Some(delta_opt) = strm.next().await {
    //         let delta = delta_opt.unwrap();
    //         println!("{:?}", delta);
    //         if let Some(msg) = agg.update(delta) {
    //             assistant_msg = Some(msg);
    //         }
    //     }
    //     let assistant_msg = assistant_msg.unwrap();
    //     println!("Assistant message: {:?}", assistant_msg);
    //     let tc = assistant_msg.tool_calls.get(0).unwrap();
    //     println!("Tool call: {:?}", tc);
    // }

    // #[tokio::test]
    // async fn infer_result_from_tool_call() {
    //     use futures::StreamExt;

    //     use crate::value::{MessageAggregator, Role, ToolDesc};
    //     use local_language_model::*;

    //     let cache = crate::cache::Cache::new();
    //     let key = "Qwen/Qwen3-0.6B";
    //     let mut model_strm = Box::pin(cache.try_create::<LocalLangModel>(key));
    //     let mut model: Option<LocalLangModel> = None;
    //     while let Some(progress) = model_strm.next().await {
    //         let mut progress = progress.unwrap();
    //         println!(
    //             "{} ({} / {})",
    //             progress.comment, progress.current_task, progress.total_task
    //         );
    //         if progress.current_task == progress.total_task {
    //             model = progress.result.take();
    //         }
    //     }
    //     let mut model = model.unwrap();
    //     model.disable_reasoning();
    //     let tools = vec![
    //         ToolDesc::new(
    //             "temperature".into(),
    //             "Get current temperature".into(),
    //             json!({
    //                 "type": "object",
    //                 "properties": {
    //                     "location": {
    //                         "type": "string",
    //                         "description": "The city name"
    //                     },
    //                     "unit": {
    //                         "type": "string",
    //                         "description": "The unit of temperature",
    //                         "enum": ["Celsius", "Fahrenheit"]
    //                     }
    //                 },
    //                 "required": ["location", "unit"]
    //             }),
    //             Some(json!({
    //                 "type": "number",
    //                 "description": "Null if the given city name is unavailable.",
    //                 "nullable": true,
    //             })),
    //         )
    //         .unwrap(),
    //     ];
    //     let msgs = vec![
    //         Message::new().with_role(Role::User)
    //             .with_contents([Part::new_text("How much hot currently in Dubai?".to_owned())]),
    //         Message::new().with_role(Role::Assistant)
    //             .with_contents([Part::new_text("\n\n")])
    //             .with_tool_calls([Part::new_function_string("\n{\"name\": \"temperature\", \"arguments\": {\"location\": \"Dubai\", \"unit\": \"Celsius\"}}\n")]),
    //         Message::new().with_role(Role::Tool).with_tool_call_id("temperature").with_contents([Part::new_text("40")])
    //     ];
    //     let mut agg = MessageAggregator::new();
    //     let mut strm = model.run(msgs, tools);
    //     let mut assistant_msg: Option<Message> = None;
    //     while let Some(delta_opt) = strm.next().await {
    //         let delta = delta_opt.unwrap();
    //         if let Some(msg) = agg.update(delta) {
    //             assistant_msg = Some(msg);
    //         }
    //     }
    //     let assistant_msg = assistant_msg.unwrap();
    //     println!("Assistant message: {:?}", assistant_msg);
    // }
}

#[cfg(all(test, target_arch = "wasm32"))]
#[cfg(test)]
mod tests {
    use futures::StreamExt as _;
    use wasm_bindgen_test::*;

    use super::*;
    use crate::value::{Delta, Part, Role};

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn infer_simple_chat() {
        let cache = crate::cache::Cache::new();
        let key = "Qwen/Qwen3-0.6B";
        let mut model_strm = Box::pin(cache.try_create::<LocalLangModel>(key));
        let mut model: Option<LocalLangModel> = None;

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
        let msgs = vec![
            Message::new(Role::System).with_contents(vec![Part::text("You are an assistant.")]),
            Message::new(Role::User).with_contents(vec![Part::text("Hi what's your name?")]),
            // Message::with_role(Role::Assistant)
            //     .with_reasoning("\nOkay, the user asked, \"Hi what's your name?\" So I need to respond appropriately.\n\nFirst, I should acknowledge their question. Since I'm an AI assistant, I don't have a name, but I can say something like, \"Hi! I'm an AI assistant. How can I assist you today?\" That shows I'm here to help. I should keep it friendly and open. Let me make sure the response is polite and professional.\n")
            //     .with_contents(vec![Part::Text(
            //         "Hi! I'm an AI assistant. How can I assist you today? 😊".to_owned(),
            //     )]),
            // Message::with_role(Role::User)
            //     .with_contents(vec![Part::Text("Who made you?".to_owned())]),
        ];
        let mut delta = MessageDelta::new();
        let mut strm = model.infer(msgs, Vec::new(), InferenceConfig::default());
        while let Some(out) = strm.next().await {
            let out = out.unwrap();
            crate::utils::log::debug(format!("{:?}", out));
            delta = delta.aggregate(out.delta).unwrap();
        }
        crate::utils::log::info(format!("{:?}", delta.finish().unwrap()));
    }
}
