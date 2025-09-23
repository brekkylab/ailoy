use crate::{
    model::sse::ServerSentEvent,
    value::{ChatCompletionMarshal, Marshaled, Message, MessageDelta, ToolDesc},
};

pub fn make_request(
    model_name: &str,
    api_key: &str,
    msgs: Vec<Message>,
    tools: Vec<ToolDesc>,
) -> reqwest::RequestBuilder {
    let mut body = serde_json::json!({
        "model": model_name,
        "messages": msgs.iter().map(|v| Marshaled::<_, ChatCompletionMarshal>::new(v)).collect::<Vec<_>>(),
        "stream": true
    });
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
            "https://api.openai.com/v1/chat/completions",
        )
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .body(body.to_string())
}

pub fn handle_event(evt: ServerSentEvent) -> Vec<MessageDelta> {
    let Ok(j) = serde_json::from_str::<serde_json::Value>(&evt.data) else {
        return Vec::new();
    };
    let Some(choice) = j.pointer("/choices/0/delta") else {
        return Vec::new();
    };
    let Ok(decoded) = serde_json::from_value::<
        crate::value::Unmarshaled<_, crate::value::ChatCompletionUnmarshal>,
    >(choice.clone()) else {
        return Vec::new();
    };
    let rv = decoded.get();
    vec![rv]
}

#[cfg(test)]
mod tests {
    use crate::{
        model::{LanguageModel as _, sse::SSELanguageModel},
        value::Delta,
    };

    const OPENAI_API_KEY: &str = "";

    #[tokio::test]
    async fn infer_simple_chat() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{Part, Role};

        let mut model = SSELanguageModel::new("gpt-4.1", OPENAI_API_KEY);

        let msgs = vec![
            Message::with_parts(
                Role::System,
                [Part::text_content("You are a helpful assistant.")],
            ),
            Message::with_parts(Role::User, [Part::text_content("Hi what's your name?")]),
        ];
        let mut agg = MessageDelta::new();
        let mut strm = model.run(msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            println!("{:?}", delta);
            agg = agg.aggregate(delta).unwrap();
        }
        println!("{:?}", agg);
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
        let msgs = vec![Message::with_parts(
            Role::User,
            [Part::text_content("How much hot currently in Dubai?")],
        )];
        let mut strm = model.run(msgs, tools);
        let mut assistant_msg = MessageDelta::default();
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            println!("{:?}", delta);
            assistant_msg = assistant_msg.aggregate(delta).unwrap();
        }
        println!("{:?}", assistant_msg.finish());
    }
}
