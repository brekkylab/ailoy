use crate::{
    model::sse::ServerSentEvent,
    value::{Config, GeminiMarshal, Marshaled, Message, MessageDelta, ToolDesc},
};

pub fn make_request(
    // model_name: &str,
    api_key: &str,
    msgs: Vec<Message>,
    tools: Vec<ToolDesc>,
    config: Config,
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

    println!("{:?}", url);
    println!("{}", body.to_string());

    reqwest::Client::new()
        .request(reqwest::Method::POST, url)
        .header("x-goog-api-key", api_key)
        .header("Content-Type", "application/json")
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .body(body.to_string())
}

pub fn handle_event(evt: ServerSentEvent) -> Vec<MessageDelta> {
    let Ok(j) = serde_json::from_str::<serde_json::Value>(&evt.data) else {
        return Vec::new();
    };
    println!("j: {:?}", j);

    let Some(candidate) = j.pointer("/candidates/0") else {
        return Vec::new();
    };

    println!("candidate: {:?}", candidate);

    let Ok(decoded) = serde_json::from_value::<
        crate::value::Unmarshaled<_, crate::value::GeminiUnmarshal>,
    >(candidate.clone()) else {
        return Vec::new();
    };
    let rv = decoded.get();
    vec![rv]
}

#[cfg(test)]
mod tests {
    // use std::sync::LazyLock;

    use crate::{
        model::{LanguageModel as _, sse::SSELanguageModel},
        value::{ConfigBuilder, Delta},
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
        let config = ConfigBuilder::new()
            .stream(true)
            .system_message("You are a helpful assistant.")
            .build();
        let mut agg = MessageDelta::new();
        let mut strm = model.run(msgs, Vec::new(), config);
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            agg = agg.aggregate(delta).unwrap();
        }
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
        let config = ConfigBuilder::new()
            .stream(true)
            .system_message("You are a helpful assistant.")
            .build();
        let msgs = vec![
            Message::new(Role::User)
                .with_contents([Part::text("How much hot currently in Dubai?")]),
        ];
        let mut strm = model.run(msgs, tools, config);
        let mut assistant_msg = MessageDelta::default();
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            println!("{:?}", delta);
            assistant_msg = assistant_msg.aggregate(delta).unwrap();
        }
        println!("{:?}", assistant_msg.finish());
    }
}
