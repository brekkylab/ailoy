use crate::{
    model::sse::ServerSentEvent,
    value::{
        FinishReason, GeminiMarshal, GeminiUnmarshal, LMConfig, Marshaled, Message, MessageDelta,
        MessageOutput, ToolDesc, Unmarshaled,
    },
};

pub fn make_request(
    api_key: &str,
    msgs: Vec<Message>,
    tools: Vec<ToolDesc>,
    config: LMConfig,
) -> reqwest::RequestBuilder {
    let mut body = serde_json::json!(&Marshaled::<_, GeminiMarshal>::new(&config));

    body["contents"] = serde_json::json!(&Marshaled::<_, GeminiMarshal>::new(&msgs));
    if !tools.is_empty() {
        body["tools"] = serde_json::json!(
            {
                "functionDeclarations": tools
                    .iter()
                    .map(|v| Marshaled::<_, GeminiMarshal>::new(v))
                    .collect::<Vec<_>>()
            }
        );
    };

    // let model = model_name;
    let model = config.model.unwrap();
    let generate_method = if config.stream {
        "streamGenerateContent?alt=sse"
    } else {
        "generateContent"
    };
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:{}",
        model, generate_method
    );

    reqwest::Client::new()
        .request(reqwest::Method::POST, url)
        .header("x-goog-api-key", api_key)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .body(body.to_string())
}

pub fn handle_event(evt: ServerSentEvent) -> MessageOutput {
    let Ok(j) = serde_json::from_str::<serde_json::Value>(&evt.data) else {
        return MessageOutput::default();
    };

    let Some(candidate) = j.pointer("/candidates/0") else {
        return MessageOutput::default();
    };

    let finish_reason = candidate
        .pointer("/finishReason")
        .and_then(|v| v.as_str())
        .map(|reason| match reason {
            "STOP" => FinishReason::Stop(),
            "MAX_TOKENS" => FinishReason::Length(),
            reason => FinishReason::Refusal(reason.to_owned()),
        });

    let delta = match finish_reason {
        Some(FinishReason::Refusal(_)) => MessageDelta::default(),
        _ => serde_json::from_value::<Unmarshaled<_, GeminiUnmarshal>>(candidate.clone())
            .ok()
            .map(|decoded| decoded.get())
            .unwrap_or_default(),
    };

    MessageOutput {
        delta,
        finish_reason,
    }
}

#[cfg(test)]
mod tests {
    // use std::sync::LazyLock;

    use crate::{
        debug,
        model::{LanguageModel as _, sse::SSELanguageModel},
        value::{Delta, LMConfigBuilder},
    };

    // static GEMINI_API_KEY: LazyLock<&'static str> = LazyLock::new(|| {
    //     option_env!("GEMINI_API_KEY")
    //         .expect("Environment variable 'GEMINI_API_KEY' is required for the tests.")
    // });
    const GEMINI_API_KEY: &str = "";

    #[tokio::test]
    async fn infer_simple_chat() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{Part, Role};

        let mut model = SSELanguageModel::new("gemini-2.5-flash-lite", GEMINI_API_KEY);

        let msgs =
            vec![Message::new(Role::User).with_contents([Part::text("Hi what's your name?")])];
        let config = LMConfigBuilder::new()
            .stream(true)
            .system_message("You are a helpful assistant.")
            .build();
        let mut assistant_msg = MessageDelta::new();
        let mut strm = model.run(msgs, Vec::new(), config);
        let mut finish_reason = FinishReason::Refusal("Initial".to_owned());
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
            if let Some(reason) = output.finish_reason {
                finish_reason = reason;
            }
        }
        assert!(assistant_msg.finish().is_ok_and(|message| {
            debug!("{:?}", message.contents.first().and_then(|c| c.as_text()));
            message.contents.len() > 0
        }));
        assert_eq!(finish_reason, FinishReason::Stop());
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn infer_tool_call() {
        use futures::StreamExt;

        use super::*;
        use crate::{
            to_value,
            value::{Part, Role, ToolDescBuilder},
        };

        let mut model = SSELanguageModel::new("gemini-2.5-flash-lite", GEMINI_API_KEY);
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
        let config = LMConfigBuilder::new()
            .stream(true)
            .system_message("You are a helpful assistant.")
            .build();
        let msgs = vec![
            Message::new(Role::User)
                .with_contents([Part::text("How much hot currently in Dubai?")]),
        ];
        let mut strm = model.run(msgs, tools, config);
        let mut assistant_msg = MessageDelta::default();
        let mut finish_reason = FinishReason::Refusal("Initial".to_owned());
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
            if let Some(reason) = output.finish_reason {
                finish_reason = reason;
            }
        }
        assert!(assistant_msg.finish().is_ok_and(|message| {
            debug!(
                "{:?}",
                message.tool_calls.first().and_then(|f| f.as_function())
            );
            message.tool_calls.len() > 0
                && message.tool_calls[0].as_function().unwrap().1 == "temperature"
        }));
        assert_eq!(finish_reason, FinishReason::Stop());
    }
}
