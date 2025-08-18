use std::sync::LazyLock;

use crate::value::{MessageFmt, PartFmt, ToolCallFmt};

pub static QWEN3_FMT: LazyLock<MessageFmt> = LazyLock::new(|| {
    let tool_call_fmt = ToolCallFmt::new();

    let mut part_fmt = PartFmt::new();
    part_fmt.tool_call_fmt = tool_call_fmt;

    let mut msg_fmt = MessageFmt::new();
    msg_fmt.part_fmt = part_fmt;
    msg_fmt.reasoning_field = String::from("reasoning_content");
    msg_fmt.contents_textonly = true; // It's text-only model

    msg_fmt
});

pub static OPENAI_FMT: LazyLock<MessageFmt> = LazyLock::new(|| {
    let mut tool_call_fmt = ToolCallFmt::new();
    tool_call_fmt.arguments_field = String::from("parameters");

    let mut part_fmt = PartFmt::new();
    part_fmt.tool_call_fmt = tool_call_fmt;
    part_fmt.image_url_field = String::from("image_url");
    part_fmt.audio_url_field = String::from("audio_url");

    let mut msg_fmt = MessageFmt::new();
    msg_fmt.part_fmt = part_fmt;

    msg_fmt
});

#[cfg(test)]
mod tests {
    use crate::value::{Message, MessageWithFmt, Part, Role, ToolCall, ToolCallArg};

    fn get_example_msg() -> Vec<Message> {
        vec![
            Message::with_role(Role::System)
                .with_contents([Part::new_text("You are an assistant.")]),
            Message::with_role(Role::User).with_contents([Part::new_text("Hi what's your name?")]),
            Message::with_role(Role::Assistant)
                .with_reasoning("Think something...")
                .with_tool_calls([Part::new_function(
                    None,
                    ToolCall::new(
                        "get_something",
                        ToolCallArg::new_object([("WhatToGet", ToolCallArg::new_string("name"))]),
                    ),
                )]),
            Message::with_role(Role::Tool).with_contents([Part::Text(String::from("Jaden"))]),
            Message::with_role(Role::Assistant)
                .with_contents([Part::Text(String::from("You can call me Jaden."))]),
        ]
    }

    #[test]
    fn formatting_to_qwen3() {
        use super::*;

        let msgs = get_example_msg();
        let formatter = QWEN3_FMT.clone();
        let msgs_in = msgs
            .iter()
            .map(|msg| MessageWithFmt::new(msg, formatter.clone()))
            .collect::<Vec<_>>();

        let out = serde_json::to_string(&msgs_in).unwrap();
        println!("{}", out);
    }

    #[test]
    fn formatting_to_openai() {
        use super::*;

        let msgs = get_example_msg();
        let formatter = OPENAI_FMT.clone();
        let msgs_in = msgs
            .iter()
            .map(|msg| MessageWithFmt::new(msg, formatter.clone()))
            .collect::<Vec<_>>();

        let out = serde_json::to_string(&msgs_in).unwrap();
        println!("{}", out);
    }
}
