use std::{collections::HashMap, sync::Arc};

use async_stream::try_stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use super::{
    super::language_model::{LangModelInferConfig, LangModelInference},
    chat_template::ChatTemplate,
    inferencer::LanguageModelInferencer,
    kv_cache::KVCacheConfig,
    tokenizer::Tokenizer,
};
use crate::{
    boxed,
    cache::{Cache, CacheClaim, CacheContents, CacheProgress, TryFromCache},
    to_value,
    utils::{BoxFuture, BoxStream, generate_random_hex_string},
    value::{
        Document, FinishReason, Message, MessageDelta, MessageDeltaOutput, PartDelta,
        PartDeltaFunction, Role, ToolDesc, Value,
    },
};

struct Request {
    msgs: Vec<Message>,
    tools: Vec<ToolDesc>,
    docs: Vec<Document>,
    config: LangModelInferConfig,
    tx_resp: mpsc::UnboundedSender<anyhow::Result<MessageDeltaOutput>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LocalLangModelConfig {
    pub device_id: Option<i32>,
    pub validate_checksum: Option<bool>,
    pub kv_cache: Option<KVCacheConfig>,
}

impl LocalLangModelConfig {
    pub fn with_device_id(mut self, device_id: i32) -> Self {
        self.device_id = Some(device_id);
        self
    }

    pub fn with_validate_checksum(mut self, validate_checksum: bool) -> Self {
        self.validate_checksum = Some(validate_checksum);
        self
    }

    pub fn with_kv_cache(mut self, kv_cache: &KVCacheConfig) -> Self {
        self.kv_cache = Some(kv_cache.clone());
        self
    }
}

#[derive(Clone, Debug)]
pub struct LocalLangModel {
    tx: Arc<mpsc::Sender<Request>>,
}

impl LocalLangModel {
    pub async fn try_new(
        model: impl Into<String>,
        config: Option<LocalLangModelConfig>,
    ) -> anyhow::Result<Self> {
        let mut strm = Self::try_new_stream(model, config);
        while let Some(v) = strm.next().await {
            if let Some(result) = v?.result {
                return Ok(result);
            }
        }
        unreachable!()
    }

    pub fn try_new_stream<'a>(
        model: impl Into<String>,
        config: Option<LocalLangModelConfig>,
    ) -> BoxStream<'a, anyhow::Result<CacheProgress<Self>>> {
        let config = config.unwrap_or_default();
        let cache = Cache::new();
        let mut ctx = HashMap::new();
        if let Some(device_id) = config.device_id {
            ctx.insert("device_id".to_owned(), Value::integer(device_id.into()));
        };
        if let Some(kv_cache) = config.kv_cache {
            let kv_cache = serde_json::to_value(kv_cache).unwrap();
            ctx.insert("kv_cache".to_owned(), kv_cache.into());
        }
        let strm = cache.try_create::<Self>(model, Some(ctx), config.validate_checksum);
        boxed!(strm)
    }

    pub fn download<'a>(
        model: impl Into<String>,
    ) -> BoxStream<'a, anyhow::Result<CacheProgress<()>>> {
        let cache = Cache::new();
        let mut strm = cache.prepare_files::<Self>(model, Some(true));
        boxed!(try_stream! {
            while let Some(res) = strm.next().await {
                let (entry, current_task, total_task, _) = res?;
                yield CacheProgress::<()> {
                    comment: format!("{} downloaded", entry.filename()),
                    current_task,
                    total_task,
                    result: None,
                }
            }
        })
    }

    pub async fn remove(model: impl Into<String>) -> anyhow::Result<()> {
        let cache = Cache::new();
        let model = model.into();
        let claim = Self::claim_files(cache.clone(), &model, &mut HashMap::new())
            .await
            .expect(format!("Failed to get the entries for {}", model).as_str());

        for entry in claim.entries.iter() {
            cache.remove(entry).await.unwrap();
        }

        Ok(())
    }
}

impl<'this> TryFromCache<'this> for LocalLangModel {
    fn claim_files<'a: 'this>(
        cache: Cache,
        key: impl AsRef<str>,
        ctx: &'a mut std::collections::HashMap<String, Value>,
    ) -> BoxFuture<'a, anyhow::Result<CacheClaim>> {
        LocalLangModelImpl::claim_files(cache, key, ctx)
    }

    fn try_from_contents<'a: 'this>(
        contents: &'a mut CacheContents,
        ctx: &'a std::collections::HashMap<String, Value>,
    ) -> BoxFuture<'a, anyhow::Result<Self>> {
        Box::pin(async move {
            let mut body = LocalLangModelImpl::try_from_contents(contents, ctx).await?;
            let (tx, mut rx) = mpsc::channel(1);

            let fut = async move {
                while let Some(req) = rx.recv().await {
                    let Request {
                        msgs,
                        tools,
                        docs,
                        config,
                        tx_resp,
                    } = req;
                    let mut strm = body.infer_delta(msgs, tools, docs, config);
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
    fn infer_delta<'a>(
        &'a mut self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        docs: Vec<Document>,
        config: LangModelInferConfig,
    ) -> BoxStream<'a, anyhow::Result<MessageDeltaOutput>> {
        let (tx_resp, mut rx_resp) = tokio::sync::mpsc::unbounded_channel();
        let req = Request {
            msgs,
            tools,
            docs,
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
    pub fn infer_delta<'a>(
        &'a mut self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        docs: Vec<Document>,
        config: LangModelInferConfig,
    ) -> BoxStream<'a, anyhow::Result<MessageDeltaOutput>> {
        let strm = try_stream! {
            let prompt = if let Some(polyfill) = config.document_polyfill {
                let msgs = polyfill.polyfill(msgs, docs)?;
                self.chat_template.apply(msgs, tools, Vec::new(), config.think_effort.unwrap_or_default(), true)?
            } else {
                self.chat_template.apply(msgs, tools, docs, config.think_effort.unwrap_or_default(), true)?
            };
            let input_tokens = self.tokenizer.encode(&prompt, true)?;

            {
                #[cfg(not(target_family = "wasm"))]
                self.inferencer.prefill(&input_tokens).unwrap();
                #[cfg(target_family = "wasm")]
                self.inferencer.prefill(&input_tokens).await.unwrap();
            }

            let mut last_token = *input_tokens.last().unwrap();
            let mut agg_tokens = Vec::<u32>::new();
            let mut count = 0;
            let mut mode = "content".to_owned();
            let mut last_s = "".to_owned();

            yield MessageDeltaOutput{delta: MessageDelta::new().with_role(Role::Assistant), finish_reason: None};

            // @jhlee: TODO remove hard-coded token names
            loop {
                let delta = MessageDelta::new();
                count += 1;
                if count > config.max_tokens.unwrap_or(16384) {
                    yield MessageDeltaOutput{delta, finish_reason: Some(FinishReason::Length{})};
                    break;
                }

                let temperature = config.temperature.unwrap_or(0.6);
                let top_p = config.top_p.unwrap_or(0.9);

                #[cfg(not(target_family = "wasm"))]
                let new_token = {
                    let logits = self.inferencer.decode(last_token).unwrap();
                    let new_token = self.inferencer.sample(logits, temperature, top_p).unwrap();
                    new_token
                };
                #[cfg(target_family = "wasm")]
                let new_token = self.inferencer.decode(last_token, temperature, top_p).await.unwrap();

                agg_tokens.push(new_token);
                last_token = new_token;

                let s = self.tokenizer.decode(agg_tokens.as_slice(), false)?;
                if s.ends_with("�") {
                    continue;
                }
                agg_tokens.clear();
                if last_s == "</tool_call>" && s == "\n" {
                    continue;
                }
                last_s = s.clone();

                if s == "<|im_end|>" {
                    yield MessageDeltaOutput{delta, finish_reason: Some(FinishReason::Stop{})};
                    break;
                } else if s == "<tool_call>" {
                    mode = "tool_call".to_owned();
                    // Generate a random tool call id
                    let delta = delta.with_tool_calls([PartDelta::Function{
                        id: Some(format!("call-{}", generate_random_hex_string(8).unwrap())),
                        function: PartDeltaFunction::Verbatim{text: "".into()}
                    }]);
                    yield MessageDeltaOutput{delta, finish_reason: None};
                } else if s == "</tool_call>" {
                    // @jhlee: Currently, parallel tool calls are not supported.
                    // The process stops immediately when the "eotc" token appears.
                    // We’ll need to establish a policy for handling parallel tool calls in the future.
                    yield MessageDeltaOutput{delta, finish_reason: Some(FinishReason::ToolCall{})};
                    break;
                } else if s == "<think>" {
                    mode = "reasoning".to_owned();
                    continue;
                } else if s == "</think>" {
                    mode = "content".to_owned();
                    continue;
                } else {
                    let delta = if mode == "content" {
                        delta.with_contents([PartDelta::Text{text: s}])
                    } else if mode == "reasoning" {
                        delta.with_thinking(s)
                    } else if mode == "tool_call" {
                        delta.with_tool_calls([PartDelta::Function{
                            id: None,
                            function: PartDeltaFunction::Verbatim{text: s}
                        }])
                    } else {
                        unreachable!();
                    };
                    yield MessageDeltaOutput{delta, finish_reason: None};
                }
            }
            return;
        };
        Box::pin(strm)
    }
}

impl<'this> TryFromCache<'this> for LocalLangModelImpl {
    fn claim_files<'a: 'this>(
        cache: Cache,
        key: impl AsRef<str>,
        ctx: &'a mut std::collections::HashMap<String, Value>,
    ) -> BoxFuture<'a, anyhow::Result<CacheClaim>> {
        let key = key.as_ref().to_owned();
        Box::pin(async move {
            let mut chat_template_claim =
                ChatTemplate::claim_files(cache.clone(), &key, ctx).await?;
            let mut chat_template_entries = Vec::new();
            for entry in chat_template_claim.entries.iter() {
                chat_template_entries.push(to_value!({
                    dirname: entry.dirname().to_owned(),
                    filename: entry.filename().to_owned()
                }));
            }
            ctx.insert(
                "chat_template_entries".to_owned(),
                chat_template_entries.into(),
            );

            let mut tokenizer_claim = Tokenizer::claim_files(cache.clone(), &key, ctx).await?;
            let mut tokenizer_entries = Vec::new();
            for entry in tokenizer_claim.entries.iter() {
                tokenizer_entries.push(to_value!({
                    dirname: entry.dirname().to_owned(),
                    filename: entry.filename().to_owned()
                }));
            }
            ctx.insert("tokenizer_entries".to_owned(), tokenizer_entries.into());

            let mut inferencer_claim =
                LanguageModelInferencer::claim_files(cache.clone(), &key, ctx).await?;
            let mut inferencer_entries = Vec::new();
            for entry in inferencer_claim.entries.iter() {
                inferencer_entries.push(to_value!({
                    dirname: entry.dirname().to_owned(),
                    filename: entry.filename().to_owned()
                }));
            }
            ctx.insert("inferencer_entries".to_owned(), inferencer_entries.into());

            let mut rv = Vec::new();
            rv.append(&mut chat_template_claim.entries);
            rv.append(&mut tokenizer_claim.entries);
            rv.append(&mut inferencer_claim.entries);
            Ok(CacheClaim::new(rv))
        })
    }

    fn try_from_contents<'a: 'this>(
        contents: &'a mut CacheContents,
        ctx: &'a std::collections::HashMap<String, Value>,
    ) -> BoxFuture<'a, anyhow::Result<Self>>
    where
        Self: Sized,
    {
        Box::pin(async move {
            let chat_template = ChatTemplate::try_from_contents(contents, ctx).await?;
            let tokenizer = Tokenizer::try_from_contents(contents, ctx).await?;
            let inferencer = LanguageModelInferencer::try_from_contents(contents, ctx).await?;
            Ok(Self {
                chat_template,
                tokenizer,
                inferencer,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use ailoy_macros::multi_platform_test;
    use futures::StreamExt;

    use super::*;
    use crate::{
        debug, to_value,
        value::{Delta, Part, Role, ToolDescBuilder},
    };

    #[multi_platform_test]
    async fn local_infer_simple_chat() {
        let mut model = LocalLangModel::try_new(
            "Qwen/Qwen3-0.6B",
            Some(LocalLangModelConfig::default().with_validate_checksum(false)),
        )
        .await
        .unwrap();
        let msgs = vec![
            Message::new(Role::System).with_contents([Part::Text {
                text: "You are an assistant.".to_owned(),
            }]),
            Message::new(Role::User).with_contents([Part::Text {
                text: "Hi what's your name?".to_owned(),
            }]),
        ];
        let mut assistant_msg = MessageDelta::new();
        let mut strm = model.infer_delta(
            msgs,
            Vec::new(),
            Vec::new(),
            LangModelInferConfig::default(),
        );
        let mut finish_reason = None;
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.accumulate(output.delta).unwrap();
            finish_reason = output.finish_reason;
        }
        assert_eq!(finish_reason, Some(FinishReason::Stop {}));
        assert!(assistant_msg.finish().is_ok_and(|message| {
            debug!("{:?}", message.contents.first().and_then(|c| c.as_text()));
            message.contents.len() > 0
        }));
    }

    #[multi_platform_test]
    async fn local_infer_tool_call() {
        let mut model = LocalLangModel::try_new(
            "Qwen/Qwen3-0.6B",
            Some(LocalLangModelConfig::default().with_validate_checksum(false)),
        )
        .await
        .unwrap();

        let tools = vec![
            ToolDescBuilder::new("temperature")
                .description("Get current temperature")
                .parameters(to_value!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "description": "The unit of temperature", "enum": ["celsius", "fahrenheit"]}
                    }
                })).build(),
        ];
        let msgs = vec![
            Message::new(Role::User)
                .with_contents(vec![Part::text("How much hot currently in Dubai?")]),
        ];

        let mut strm = model.infer_delta(msgs, tools, Vec::new(), LangModelInferConfig::default());
        let mut assistant_msg = MessageDelta::default();
        let mut finish_reason = None;
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.accumulate(output.delta).unwrap();
            finish_reason = output.finish_reason;
        }
        assert_eq!(finish_reason, Some(FinishReason::ToolCall {}));
        assert!(assistant_msg.finish().is_ok_and(|message| {
            if let Some(tool_calls) = message.tool_calls {
                debug!("{:?}", tool_calls.first().and_then(|f| f.as_function()));
                tool_calls
                    .first()
                    .and_then(|f| f.as_function())
                    .map(|f| f.1 == "temperature")
                    .unwrap_or(false)
            } else {
                false
            }
        }));
    }

    #[multi_platform_test]
    async fn local_infer_tool_response() {
        let mut model = LocalLangModel::try_new(
            "Qwen/Qwen3-0.6B",
            Some(LocalLangModelConfig::default().with_validate_checksum(false)),
        )
        .await
        .unwrap();

        let tools = vec![
            ToolDescBuilder::new("temperature")
                .description("Get current temperature")
                .parameters(to_value!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "description": "The unit of temperature", "enum": ["celsius", "fahrenheit"]}
                    }
                })).build(),
        ];
        let msgs = vec![
            Message::new(Role::User)
                .with_contents([Part::text("How much hot currently in Dubai?".to_owned())]),
            Message::new(Role::Assistant).with_tool_calls([Part::function_with_id(
                "call-a560d635cdaedff2",
                "temperature",
                to_value!({"location": "Dubai", "unit": "fahrenheit"}),
            )]),
            Message::new(Role::Assistant).with_tool_calls([Part::function_with_id(
                "call-4bc4010916f4fa08",
                "temperature",
                to_value!({"location": "Dubai", "unit": "celsius"}),
            )]),
            Message::new(Role::Tool)
                .with_id("call-a560d635cdaedff2")
                .with_contents([Part::Value {
                    value: to_value!({"temperature": 86, "unit": "fahrenheit"}),
                }]),
            Message::new(Role::Tool)
                .with_id("call-4bc4010916f4fa08")
                .with_contents([Part::Value {
                    value: to_value!({"temperature": 30, "unit": "celsius"}),
                }]),
        ];
        let mut strm = model.infer_delta(msgs, tools, Vec::new(), LangModelInferConfig::default());
        let mut assistant_msg = MessageDelta::default();
        let mut finish_reason = None;
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.accumulate(output.delta).unwrap();
            finish_reason = output.finish_reason;
        }
        assert_eq!(finish_reason, Some(FinishReason::Stop {}));
        assert!(assistant_msg.finish().is_ok_and(|message| {
            debug!("{:?}", message.contents.first().and_then(|c| c.as_text()));
            message.contents.len() > 0
        }));
    }
}
