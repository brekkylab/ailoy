use crate::{
    model::{APIModel, sse::ServerSentEvent},
    value::{
        ChatCompletionMarshal, ChatCompletionUnmarshal, FinishReason, LMConfig, Marshaled, Message,
        MessageDelta, MessageOutput, Part, Role, ToolDesc, Unmarshaled,
    },
};

pub fn make_request(
    api_model: &APIModel,
    msgs: Vec<Message>,
    tools: Vec<ToolDesc>,
    config: LMConfig,
) -> reqwest::RequestBuilder {
    let mut body = serde_json::json!(Marshaled::<_, ChatCompletionMarshal>::new(&config));
    let msgs = if let Some(system_message) = config.system_message {
        let mut new_msgs =
            vec![Message::new(Role::System).with_contents([Part::text(system_message)])];
        new_msgs.extend(msgs);
        new_msgs
    } else {
        msgs
    };
    body["messages"] = serde_json::json!(Marshaled::<_, ChatCompletionMarshal>::new(&msgs));

    if !tools.is_empty() {
        body["tool_choice"] = serde_json::json!("auto");
        body["tools"] = serde_json::json!(
            tools
                .iter()
                .map(|v| Marshaled::<_, ChatCompletionMarshal>::new(v))
                .collect::<Vec<_>>()
        );
    }

    reqwest::Client::new()
        .request(
            reqwest::Method::POST,
            api_model.endpoint(),
        )
        .bearer_auth(api_model.api_key.clone())
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .body(body.to_string())
}

pub fn handle_event(evt: ServerSentEvent) -> MessageOutput {
    let Ok(j) = serde_json::from_str::<serde_json::Value>(&evt.data) else {
        return MessageOutput::default();
    };
    let Some(choice) = j.pointer("/choices/0/delta") else {
        return MessageOutput::default();
    };
    let finish_reason = choice
        .pointer("/finish_reason")
        .and_then(|v| v.as_str())
        .map(|reason| match reason {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCall,
            "content_filter" => {
                FinishReason::Refusal("Model output violated XAI's safety policy.".to_owned())
            }
            reason => FinishReason::Refusal(format!("reason: {}", reason)),
        });

    let delta = match finish_reason {
        Some(FinishReason::Refusal(_)) => MessageDelta::default(),
        _ => serde_json::from_value::<Unmarshaled<_, ChatCompletionUnmarshal>>(choice.clone())
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
    use futures::StreamExt;

    use super::*;
    use crate::{
        model::{APIProvider, LanguageModel as _, sse::SSELanguageModel},
        value::{Delta, LMConfigBuilder, Part, Role},
    };

    const XAI_API_KEY: &str = "";

    #[tokio::test]
    async fn infer_simple_chat() {
        let model =
            SSELanguageModel::new(APIModel::new(APIProvider::XAI, "grok-4-0709", XAI_API_KEY));

        let msgs = vec![
            Message::new(Role::System).with_contents([Part::text("You are a helpful assistant.")]),
            Message::new(Role::User).with_contents([Part::text("Hi what's your name?")]),
        ];
        let config = LMConfigBuilder::new()
            .system_message("You are a helpful assistant.")
            .stream(true)
            .build();
        let mut agg = MessageDelta::new();
        let mut strm = model.run(msgs, Vec::new(), config);
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            agg = agg.aggregate(output.delta).unwrap();
        }
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn infer_tool_call() {
        use crate::{to_value, value::ToolDescBuilder};

        let model =
            SSELanguageModel::new(APIModel::new(APIProvider::XAI, "grok-4-0709", XAI_API_KEY));
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
            println!("{:?}", output);
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
        }
        println!("{:?}", assistant_msg.finish());
    }
}
