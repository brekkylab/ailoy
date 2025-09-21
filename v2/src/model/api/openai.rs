use crate::{
    model::sse::ServerSideEvent,
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
        body["tools"] = serde_json::to_value(tools).unwrap();
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

pub fn handle_event(evt: ServerSideEvent) -> Vec<MessageDelta> {
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

    // #[cfg(any(target_family = "unix", target_family = "windows"))]
    // #[tokio::test]
    // async fn infer_tool_call() {
    //     use futures::StreamExt;

    //     use super::*;
    //     use crate::value::{MessageAggregator, Part, Role, ToolDesc, ToolDescArg};

    //     let model = Arc::new(APILanguageModel::new("gpt-4.1", OPENAI_API_KEY));
    //     let tools = vec![ToolDesc::new(
    //         "temperature",
    //         "Get current temperature",
    //         ToolDescArg::new_object().with_properties(
    //             [
    //                 (
    //                     "location",
    //                     ToolDescArg::new_string().with_desc("The city name"),
    //                 ),
    //                 (
    //                     "unit",
    //                     ToolDescArg::new_string()
    //                         .with_enum(["Celcius", "Fernheit"])
    //                         .with_desc("The unit of temperature"),
    //                 ),
    //             ],
    //             ["location", "unit"],
    //         ),
    //         Some(
    //             ToolDescArg::new_number().with_desc("Null if the given city name is unavailable."),
    //         ),
    //     )];
    //     let msgs = vec![
    //         Message::with_role(Role::User)
    //             .with_contents([Part::Text("How much hot currently in Dubai?".to_owned())]),
    //     ];
    //     let mut agg = MessageAggregator::new();
    //     let mut strm = model.run(msgs, tools);
    //     let mut assistant_msg: Option<Message> = None;
    //     while let Some(delta_opt) = strm.next().await {
    //         println!("{:?}", delta_opt);
    //         let delta = delta_opt.unwrap();
    //         if let Some(msg) = agg.update(delta) {
    //             assistant_msg = Some(msg);
    //         }
    //     }
    //     let assistant_msg = assistant_msg.unwrap();
    //     let tc = assistant_msg.tool_calls.get(0).unwrap();
    //     println!("Tool call: {:?}", tc);
    // }
}
