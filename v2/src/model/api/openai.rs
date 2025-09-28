use crate::{
    model::{APIModel, sse::ServerSentEvent},
    value::{
        FinishReason, LMConfig, Marshaled, Message, MessageDelta, MessageOutput, OpenAIMarshal,
        OpenAIUnmarshal, ToolDesc, Unmarshaled,
    },
};

pub fn make_request(
    api_model: &APIModel,
    msgs: Vec<Message>,
    tools: Vec<ToolDesc>,
    config: LMConfig,
) -> reqwest::RequestBuilder {
    let mut body = serde_json::json!(&Marshaled::<_, OpenAIMarshal>::new(&config));

    body["input"] = serde_json::json!(&Marshaled::<_, OpenAIMarshal>::new(&msgs));
    if !tools.is_empty() {
        body["tool_choice"] = serde_json::json!("auto");
        body["tools"] = serde_json::json!(
            tools
                .iter()
                .map(|v| Marshaled::<_, OpenAIMarshal>::new(v))
                .collect::<Vec<_>>()
        );
    }

    reqwest::Client::new()
        .request(reqwest::Method::POST, api_model.endpoint())
        .bearer_auth(api_model.api_key.clone())
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .body(body.to_string())
}

pub fn handle_event(evt: ServerSentEvent) -> MessageOutput {

    let Ok(val) = serde_json::from_str::<serde_json::Value>(&evt.data) else {
        return MessageOutput::default();
    };

    match evt.event.as_str() {
        "response.completed" => {
            // Valid termination of stream
            return MessageOutput {
                delta: MessageDelta::default(),
                finish_reason: Some(FinishReason::Stop),
            };
        }
        "response.refusal.done" => {
            // Refusal message retrieved
            let refusal_text = val
                .pointer("/refusal")
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| "reason: unknown");
            return MessageOutput {
                delta: MessageDelta::default(),
                finish_reason: Some(FinishReason::Refusal(refusal_text.to_owned())),
            };
        }
        "response.incomplete" => {
            // Incomplete termination of stream
            let reason = val
                .pointer("/response.incomplete/reason")
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| "unknown");
            let finish_reason = match reason {
                "max_output_tokens" => FinishReason::Length,
                "content_filter" => FinishReason::Refusal(
                    "Model output violated OpenAI's safety policy.".to_owned(),
                ),
                reason => FinishReason::Refusal(format!("reason: {}", reason)),
            };
            return MessageOutput {
                delta: MessageDelta::default(),
                finish_reason: Some(finish_reason),
            };
        }
        _ => {
            // Ongoing stream
            let Ok(decoded) = serde_json::from_value::<Unmarshaled<_, OpenAIUnmarshal>>(val) else {
                return MessageOutput::default();
            };
            MessageOutput {
                delta: decoded.get(),
                finish_reason: None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        model::{LanguageModel as _, sse::SSELanguageModel},
        value::{Delta, LMConfigBuilder},
    };

    const OPENAI_API_KEY: &str = "";

    #[tokio::test]
    async fn infer_simple_chat() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{Part, Role};

        let model = SSELanguageModel::new(APIModel::new(
            crate::model::APIProvider::OpenAI,
            "gpt-4.1",
            OPENAI_API_KEY,
        ));

        let msgs =
            vec![Message::new(Role::User).with_contents([Part::text("Hi what's your name?")])];
        let config = LMConfigBuilder::new()
            .system_message("You are a helpful assistant.")
            .stream(true)
            .build();
        let mut assistant_msg = MessageDelta::new();
        let mut strm = model.run(msgs, Vec::new(), config);
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
        }
        println!("{:?}", assistant_msg.finish());
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn infer_tool_call() {
        use futures::StreamExt;

        use super::*;
        use crate::{
            model::APIProvider,
            to_value,
            value::{Part, Role, ToolDescBuilder},
        };

        let model = SSELanguageModel::new(APIModel::new(
            APIProvider::OpenAI,
            "gpt-4.1",
            OPENAI_API_KEY,
        ));
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
                .with_contents([Part::text("How much hot currently in Dubai?")]),
        ];
        let config = LMConfigBuilder::new().stream(true).build();
        let mut strm = model.run(msgs, tools, config);
        let mut assistant_msg = MessageDelta::default();
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
        }
        println!("{:?}", assistant_msg.finish());
    }
}
