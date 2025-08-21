mod openai;

use futures::StreamExt;
use std::{fmt::Debug, sync::Arc};

use crate::{
    model::LanguageModel,
    value::{Message, MessageOutput, MessageStyle, OPENAI_FMT, StyledMessage, ToolDesc},
};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub struct APILanguageModel {
    model: String,
    style: MessageStyle,
    make_request: Arc<
        dyn Fn(
                Vec<StyledMessage>,
                Vec<ToolDesc>,
            )
                -> futures::future::BoxFuture<'static, Result<reqwest::Response, reqwest::Error>>
            + Send
            + Sync,
    >,
    handle_response: Arc<dyn Fn(&mut Vec<u8>) -> Result<Vec<MessageOutput>, String> + Send + Sync>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
pub struct APILanguageModel {
    model: String,
    style: MessageStyle,
    make_request: Arc<
        dyn Fn(
                Vec<StyledMessage>,
                Vec<ToolDesc>,
            ) -> futures::future::LocalBoxFuture<
                'static,
                Result<reqwest::Response, reqwest::Error>,
            > + Send
            + Sync,
    >,
    handle_response: Arc<dyn Fn(&mut Vec<u8>) -> Result<Vec<MessageOutput>, String>>,
}

impl APILanguageModel {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> APILanguageModel {
        let model = model.into();
        let api_key = api_key.into();
        let (style, make_request, handle_response) =
            if model.starts_with("gpt") || model.starts_with("o") {
                let style = OPENAI_FMT.clone();
                let model = model.clone();
                let api_key = api_key.clone();
                (
                    style,
                    Arc::new(move |msgs: Vec<StyledMessage>, tools: Vec<ToolDesc>| {
                        openai::make_request(&model, &api_key, msgs, tools)
                    }),
                    Arc::new(&openai::handle_next_response),
                )
            } else if model.starts_with("claude") {
                todo!()
            } else if model.starts_with("gemini") {
                todo!()
            } else if model.starts_with("grok") {
                todo!()
            } else {
                panic!()
            };

        APILanguageModel {
            model,
            style,
            make_request,
            handle_response,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl LanguageModel for APILanguageModel {
    fn run(
        self: Arc<Self>,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> futures::stream::BoxStream<'static, Result<MessageOutput, String>> {
        let msgs = msgs
            .iter()
            .map(|v| StyledMessage {
                data: v.clone(),
                style: self.style.clone(),
            })
            .collect::<Vec<_>>();
        let req = (self.make_request)(msgs, tools);
        let strm = async_stream::try_stream! {
            let mut buf: Vec<u8> = Vec::with_capacity(8192);
            let resp = req.await.map_err(|e| e.to_string())?;
            if resp.status().is_success() {
                let mut strm = resp.bytes_stream();
                while let Some(chunk_res) = strm.next().await {
                    let chunk = chunk_res.map_err(|e| e.to_string())?;
                    buf.extend_from_slice(&chunk);
                    let outs = (self.handle_response)(&mut buf)?;
                    for v in outs {
                        yield v;
                    }
                }
            } else {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                Err(format!("Request failed: {} - {}", status, text))?;
            }
        };
        Box::pin(strm)
    }
}

#[cfg(target_arch = "wasm32")]
impl LanguageModel for APILanguageModel {
    fn run(
        self: Arc<Self>,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> futures::stream::LocalBoxStream<'static, Result<MessageOutput, String>> {
        let msgs = msgs
            .iter()
            .map(|v| StyledMessage {
                data: v.clone(),
                style: self.style.clone(),
            })
            .collect::<Vec<_>>();
        let req = (self.make_request)(msgs, tools);
        let strm = async_stream::try_stream! {
            let mut buf: Vec<u8> = Vec::with_capacity(8192);
            let resp = req.await.map_err(|e| e.to_string())?;
            if resp.status().is_success() {
                let mut strm = resp.bytes_stream();
                while let Some(chunk_res) = strm.next().await {
                    let chunk = chunk_res.map_err(|e| e.to_string())?;
                    buf.extend_from_slice(&chunk);
                    let outs = (self.handle_response)(&mut buf)?;
                    for v in outs {
                        yield v;
                    }
                }
            } else {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                Err(format!("Request failed: {} - {}", status, text))?;
            }
        };
        Box::pin(strm)
    }
}

impl Debug for APILanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("APILanguageModel")
            .field("model", &self.model)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    const OPENAI_API_KEY: &str = "";

    #[tokio::test]
    async fn infer_simple_chat() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{MessageAggregator, Part, Role};

        let model = Arc::new(APILanguageModel::new("gpt-4.1", OPENAI_API_KEY));

        let msgs = vec![
            Message::with_role(Role::System)
                .with_contents(vec![Part::Text("You are an assistant.".to_owned())]),
            Message::with_role(Role::User)
                .with_contents(vec![Part::Text("Hi what's your name?".to_owned())]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            println!("{:?}", delta);
            if let Some(msg) = agg.update(delta) {
                println!("{:?}", msg);
            }
        }
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn infer_tool_call() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{MessageAggregator, Part, Role, ToolDesc, ToolDescArg};

        let model = Arc::new(APILanguageModel::new("gpt-4.1", OPENAI_API_KEY));
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
                .with_contents([Part::Text("How much hot currently in Dubai?".to_owned())]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(msgs, tools);
        let mut assistant_msg: Option<Message> = None;
        while let Some(delta_opt) = strm.next().await {
            println!("{:?}", delta_opt);
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                assistant_msg = Some(msg);
            }
        }
        let assistant_msg = assistant_msg.unwrap();
        let tc = assistant_msg.tool_calls.get(0).unwrap();
        println!("Tool call: {:?}", tc);
    }
}
