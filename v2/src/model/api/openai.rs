use crate::{
    model::sse::ServerSentEvent,
    value::{
        Config, FinishReason, Marshaled, Message, MessageDelta, MessageOutput, OpenAIMarshal,
        OpenAIUnmarshal, ToolDesc, Unmarshaled,
    },
};

pub fn make_request(
    // model_name: &str,
    api_key: &str,
    msgs: Vec<Message>,
    tools: Vec<ToolDesc>,
    config: Config,
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
        .request(reqwest::Method::POST, "https://api.openai.com/v1/responses")
        .bearer_auth(api_key)
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
    let Ok(decoded) = serde_json::from_value::<
        crate::value::Unmarshaled<_, crate::value::ChatCompletionUnmarshal>,
    >(choice.clone()) else {
        return MessageOutput::default();
    };
    let rv = decoded.get();
    MessageOutput {
        delta: rv,
        finish_reason: None,
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        model::{LanguageModel as _, sse::SSELanguageModel},
        value::{ConfigBuilder, Delta},
    };

    const OPENAI_API_KEY: &str = "";

    #[tokio::test]
    async fn infer_simple_chat() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{Part, Role};

        let mut model = SSELanguageModel::new("gpt-4.1", OPENAI_API_KEY);

        let msgs =
            vec![Message::new(Role::User).with_contents([Part::text("Hi what's your name?")])];
        let config = ConfigBuilder::new()
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
            to_value,
            value::{Part, Role, ToolDescBuilder},
        };

        let mut model = SSELanguageModel::new("gpt-4.1", OPENAI_API_KEY);
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
        let config = ConfigBuilder::new().stream(true).build();
        let mut strm = model.run(msgs, tools, config);
        let mut assistant_msg = MessageDelta::default();
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
        }
        println!("{:?}", assistant_msg.finish());
    }
}
