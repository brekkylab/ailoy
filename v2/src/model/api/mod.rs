mod openai;

use futures::{StreamExt as _, future::BoxFuture, stream::BoxStream};
use std::{fmt::Debug, sync::Arc};

use crate::{
    model::LanguageModel,
    value::{Message, MessageOutput, ToolDescription},
};

#[derive(Clone)]
pub struct APILanguageModel {
    model: String,
    make_request: Arc<
        dyn Fn(
                Vec<Message>,
                Vec<ToolDescription>,
            ) -> BoxFuture<'static, Result<reqwest::Response, reqwest::Error>>
            + Send
            + Sync,
    >,
    handle_response: Arc<dyn Fn(&mut Vec<u8>) -> Result<Vec<MessageOutput>, String> + Send + Sync>,
}

impl APILanguageModel {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> APILanguageModel {
        let model = model.into();
        let api_key = api_key.into();
        let (make_request, handle_response) = if model.starts_with("gpt") || model.starts_with("o")
        {
            let model = model.clone();
            let api_key = api_key.clone();
            (
                Arc::new(move |msgs: Vec<Message>, tools: Vec<ToolDescription>| {
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
            make_request,
            handle_response,
        }
    }
}

impl LanguageModel for APILanguageModel {
    fn run(
        self: Arc<Self>,
        msgs: Vec<Message>,
        tools: Vec<ToolDescription>,
    ) -> BoxStream<'static, Result<MessageOutput, String>> {
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
            Message::with_content(Role::System, Part::new_text("You are an assistant.")),
            Message::with_content(Role::User, Part::new_text("Hi what's your name?")),
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
}
