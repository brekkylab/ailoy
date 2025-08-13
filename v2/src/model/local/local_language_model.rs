use std::sync::Arc;

use async_stream::try_stream;
use futures::{Stream, future::BoxFuture, stream::BoxStream};
use tokio::sync::Mutex;

use crate::{
    cache::{Cache, CacheContents, CacheEntry, CacheProgress, TryFromCache},
    model::{
        LanguageModel,
        local::{ChatTemplate, Inferencer, Tokenizer},
    },
    value::{Message, MessageDelta, Part, ToolDescription},
};

#[derive(Debug)]
pub struct LocalLanguageModel {
    chat_template: ChatTemplate,

    tokenizer: Tokenizer,

    // The inferencer performs mutable operations such as KV cache updates, so the mutex is applied.
    inferencer: Mutex<Inferencer>,
}

impl LocalLanguageModel {
    pub async fn try_new(
        model_name: impl Into<String>,
    ) -> impl Stream<Item = Result<CacheProgress<Self>, String>> + 'static {
        let model_name = model_name.into();
        Cache::new().try_create::<LocalLanguageModel>(model_name)
    }
}

impl LanguageModel for LocalLanguageModel {
    fn run(
        self: Arc<Self>,
        tools: Vec<ToolDescription>,
        msgs: Vec<Message>,
    ) -> BoxStream<'static, Result<MessageDelta, String>> {
        let strm = try_stream! {
            let prompt = self.chat_template.apply_with_vec(&tools, &msgs, true)?;
            let input_tokens = self.tokenizer.encode(&prompt, true)?;
            self.inferencer.lock().await.prefill(&input_tokens);
            let mut last_token = *input_tokens.last().unwrap();
            let mut agg_tokens = Vec::<u32>::new();
            let mut count = 0;
            let mut mode = "content".to_owned();
            // @jhlee: TODO remove hard-coded token names
            loop {
                count += 1;
                if count > 16384 {
                    Err("Too long assistant message. It may be infinite loop".to_owned())?;
                }
                let new_token = self.inferencer.lock().await.decode(last_token);
                agg_tokens.push(new_token);
                last_token = new_token;
                let s = self.tokenizer.decode(agg_tokens.as_slice(), false)?;
                if s.ends_with("�") {
                    continue;
                }
                agg_tokens.clear();

                if s == "<|im_end|>" {
                    break;
                } else if s == "<tool_call>" {
                    mode = "tool_call".to_owned();
                    continue;
                } else if s == "</tool_call>" {
                    mode = "content".to_owned();
                    // It's a separator when successive tool call occurs
                     yield MessageDelta::new_assistant_content(Part::Text(String::new()));
                    continue;
                } else if s == "<think>" {
                    mode = "reasoning".to_owned();
                    continue;
                } else if s == "</think>" {
                    mode = "content".to_owned();
                    continue;
                } else {
                    if mode == "content" {
                        yield MessageDelta::new_assistant_content(Part::Text(s));
                    } else if mode == "reasoning" {
                        yield MessageDelta::new_assistant_reasoning(Part::Text(s));
                    } else if mode == "tool_call" {
                        yield MessageDelta::new_assistant_tool_call(Part::Function{id: None, function: s});
                    }
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
    ) -> BoxFuture<'static, Result<Vec<CacheEntry>, String>> {
        let key = key.as_ref().to_owned();
        Box::pin(async move {
            let mut rv = Vec::new();
            rv.append(&mut ChatTemplate::claim_files(cache.clone(), &key).await?);
            rv.append(&mut Tokenizer::claim_files(cache.clone(), &key).await?);
            rv.append(&mut Inferencer::claim_files(cache.clone(), &key).await?);
            Ok(rv)
        })
    }

    fn try_from_contents(contents: &mut CacheContents) -> Result<Self, String>
    where
        Self: Sized,
    {
        let chat_template = ChatTemplate::try_from_contents(contents)?;
        let tokenizer = Tokenizer::try_from_contents(contents)?;
        let inferencer = Inferencer::try_from_contents(contents)?;
        Ok(LocalLanguageModel {
            chat_template,
            tokenizer,
            inferencer: Mutex::new(inferencer),
        })
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(target_family = "unix", target_family = "windows"))]
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
            let progress = progress.unwrap();
            println!(
                "{} ({} / {})",
                progress.comment(),
                progress.current_task(),
                progress.total_task()
            );
            if progress.current_task() == progress.total_task() {
                model = progress.take();
            }
        }
        let model = Arc::new(model.unwrap());
        let msgs = vec![
            Message::with_content(Role::System, Part::new_text("You are an assistant.")),
            Message::with_content(Role::User, Part::new_text("Hi what's your name?")),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(Vec::new(), msgs);
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                println!("{:?}", msg);
            }
        }
        println!("{:?}", agg.finalize());
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn infer_tool_call() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{
            MessageAggregator, Role, ToolCall, ToolDescription, ToolDescriptionArgument,
        };

        let cache = crate::cache::Cache::new();
        let key = "Qwen/Qwen3-0.6B";
        let mut model_strm = Box::pin(cache.try_create::<LocalLanguageModel>(key));
        let mut model: Option<LocalLanguageModel> = None;
        while let Some(progress) = model_strm.next().await {
            let progress = progress.unwrap();
            println!(
                "{} ({} / {})",
                progress.comment(),
                progress.current_task(),
                progress.total_task()
            );
            if progress.current_task() == progress.total_task() {
                model = progress.take();
            }
        }
        let model = Arc::new(model.unwrap());
        let tools = vec![ToolDescription::new(
            "temperature",
            "Get current temperature",
            ToolDescriptionArgument::new_object().with_properties(
                [
                    (
                        "location",
                        ToolDescriptionArgument::new_string().with_desc("The city name"),
                    ),
                    (
                        "unit",
                        ToolDescriptionArgument::new_string().with_enum(["Celcius", "Fernheit"]),
                    ),
                ],
                ["location", "unit"],
            ),
            Some(
                ToolDescriptionArgument::new_number()
                    .with_desc("Null if the given city name is unavailable."),
            ),
        )];
        let msgs = vec![Message::with_content(
            Role::User,
            Part::new_text("How much hot currently in Dubai?"),
        )];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(tools, msgs);
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                println!("{:?}", msg);
            }
        }
        let resp = agg.finalize();
        println!("Resp: {:?}", resp);
        let tc = ToolCall::try_from_string(
            resp.unwrap()
                .tool_calls
                .get(0)
                .unwrap()
                .get_function_owned()
                .unwrap(),
        )
        .unwrap();
        println!("Tool call: {:?}", tc);
    }
}
