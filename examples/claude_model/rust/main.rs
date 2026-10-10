use ailoy::{
    experimental::model::{ClaudeModel, InferLangModel, LangModelOptions},
    message::{Message, Part, Role},
    tool::ToolDescBuilder,
};
use futures::StreamExt as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = match std::env::args().nth(1) {
        Some(name) => ClaudeModel::new().with_model(name),
        None => ClaudeModel::new(),
    };
    let tools = [ToolDescBuilder::new("get_weather")
        .description("Get the current weather for a city.")
        .parameters(serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {"type": "integer", "description": "Forecast days, 1 for today only"}
            },
            "required": ["city"]
        }))
        .build()];

    let mut messages = vec![Message::new(Role::User).with_contents([Part::text(
        "How's the weather in Seoul and Tokyo today, in celsius?",
    )])];

    // Turn 1: streamed, should stop at the tool calls.
    print!("[turn 1] ");
    let mut stream = model.infer_stream(&messages, &tools, &LangModelOptions::default());
    let mut acc = ailoy::message::MessageDeltaOutput::new();
    while let Some(delta) = stream.next().await {
        let delta = delta?;
        for part in delta.delta.contents.clone() {
            if let Some(text) = part.to_text() {
                print!("{text}");
            }
        }
        acc = ailoy::message::Delta::accumulate(acc, delta)?;
    }
    let output = ailoy::message::Delta::finish(acc)?;
    println!();
    println!("[finish] {:?}", output.finish_reason);
    println!("[tool_calls] {:?}", output.message.tool_calls);
    println!("[usage] {:?}", output.usage);

    // Answer each call with a made-up result.
    let calls = output.message.tool_calls.clone().unwrap_or_default();
    messages.push(output.message);
    for call in calls {
        let Part::Function { id, function } = call else {
            continue;
        };
        let city = function
            .arguments
            .pointer("/city")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let result = format!("{city}: 18°C, light rain");
        messages.push(
            Message::new(Role::Tool)
                .with_id(id)
                .with_contents([Part::text(result)]),
        );
    }

    // Turn 2: non-streamed, should answer from the results.
    let output = model
        .infer(&messages, &tools, &LangModelOptions::default())
        .await?;
    println!("[turn 2] {:?}", output.message.contents);
    println!("[finish] {:?}", output.finish_reason);
    println!("[usage] {:?}", output.usage);

    Ok(())
}
