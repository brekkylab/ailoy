use std::{pin::Pin, sync::Arc};

use async_stream::try_stream;
use futures::Stream;
use tokio::sync::{Mutex, RwLock};

use crate::{
    cache::{Cache, CacheElement, TryFromCache},
    model::{
        LanguageModel,
        local::{ChatTemplate, Inferencer, Tokenizer},
    },
    value::{Message, MessageDelta, Part, ToolDescription},
};

#[derive(Debug, Clone)]
pub struct LocalLanguageModel {
    chat_template: Arc<RwLock<ChatTemplate>>,
    tokenizer: Arc<RwLock<Tokenizer>>,
    inferencer: Arc<Mutex<Inferencer>>,
}

impl LanguageModel for LocalLanguageModel {
    fn run(
        self,
        tools: Vec<ToolDescription>,
        msgs: Vec<Message>,
    ) -> Pin<Box<dyn Stream<Item = Result<MessageDelta, String>>>> {
        let strm = try_stream! {
            let prompt = self.chat_template.read().await.apply_with_vec(&tools, &msgs, true)?;
            let input_tokens = self.tokenizer.read().await.encode(&prompt, true)?;
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
                let s = self.tokenizer.read().await.decode(agg_tokens.as_slice(), false)?;
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
                    continue;
                } else if s == "<think>" {
                    mode = "reasoning".to_owned();
                    continue;
                } else if s == "</think>" {
                    mode = "content".to_owned();
                    continue;
                } else {
                    if mode == "content" {
                        yield MessageDelta::content(Part::Text(s));
                    } else if mode == "reasoning" {
                        yield MessageDelta::reasoning(Part::Text(s));
                    } else if mode == "tool_call" {
                        yield MessageDelta::tool_call(Part::Json(s));
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
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CacheElement>, String>>>> {
        let key = key.as_ref().to_owned();
        Box::pin(async move {
            let mut rv = Vec::new();
            rv.append(&mut ChatTemplate::claim_files(cache.clone(), &key).await?);
            rv.append(&mut Tokenizer::claim_files(cache.clone(), &key).await?);
            rv.append(&mut Inferencer::claim_files(cache.clone(), &key).await?);
            Ok(rv)
        })
    }

    fn try_from_files(cache: &Cache, files: Vec<(CacheElement, Vec<u8>)>) -> Result<Self, String>
    where
        Self: Sized,
    {
        let chat_template_files = vec![files.get(0).unwrap().clone()];
        let chat_template = ChatTemplate::try_from_files(cache, chat_template_files)?;
        let tokenizer_files = vec![files.get(1).unwrap().clone()];
        let tokenizer = Tokenizer::try_from_files(cache, tokenizer_files)?;
        let inferencer_files = files[2..].to_vec();
        let inferencer = Inferencer::try_from_files(cache, inferencer_files)?;
        Ok(LocalLanguageModel {
            chat_template: Arc::new(RwLock::new(chat_template)),
            tokenizer: Arc::new(RwLock::new(tokenizer)),
            inferencer: Arc::new(Mutex::new(inferencer)),
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
        let model = cache.try_create::<LocalLanguageModel>(key).await.unwrap();
        let msgs = vec![
            Message::with_content(Role::System, Part::new_text("You are an assistant.")),
            Message::with_content(Role::User, Part::new_text("Hi what's your name?")),
        ];
        let mut agg = MessageAggregator::new(Role::Assistant);
        let mut strm = model.run(Vec::new(), msgs);
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            agg.update(delta);
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
        let model = cache.try_create::<LocalLanguageModel>(key).await.unwrap();
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
        let mut agg = MessageAggregator::new(Role::Assistant);
        let mut strm = model.run(tools, msgs);
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            agg.update(delta);
        }
        let resp = agg.finalize();
        println!("Resp: {:?}", resp);
        let tc =
            ToolCall::try_from_string(resp.tool_calls().get(0).unwrap().get_json_owned().unwrap())
                .unwrap();
        println!("Tool call: {:?}", tc);
    }
}
