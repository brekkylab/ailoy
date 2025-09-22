use std::sync::LazyLock;

use crate::value::{MessageStyle, PartStyle};

pub static QWEN3_FMT: LazyLock<MessageStyle> = LazyLock::new(|| {
    let part_style = PartStyle::new();

    let mut message_style = MessageStyle::new();
    message_style.part_style = part_style;
    message_style.reasoning_field = String::from("reasoning_content");
    message_style.contents_textonly = true; // It's text-only model

    message_style
});

pub static OPENAI_FMT: LazyLock<MessageStyle> = LazyLock::new(|| {
    let mut part_style = PartStyle::new();
    part_style.function_arguments_field = String::from("parameters");
    part_style.image_url_field = String::from("image_url");

    let mut message_style = MessageStyle::new();
    message_style.part_style = part_style;

    message_style
});

#[cfg(test)]
mod tests {
    use crate::value::{Part, Role, StyledMessage};

    fn get_example_msg() -> Vec<StyledMessage> {
        vec![
            StyledMessage::new()
                .with_role(Role::System)
                .with_contents([Part::Text("You are an assistant.".to_owned())]),
            StyledMessage::new()
                .with_role(Role::User)
                .with_contents([Part::new_text("Hi what's your name?")]),
            StyledMessage::new()
                .with_role(Role::Assistant)
                .with_reasoning("Think something...")
                .with_tool_calls([Part::new_function(
                    "",
                    "get_something",
                    "{\"WhatToGet\": \"name\"}",
                )]),
            StyledMessage::new()
                .with_role(Role::Tool)
                .with_contents([Part::Text(String::from("Jaden"))]),
            StyledMessage::new()
                .with_role(Role::Assistant)
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
            .map(|msg| StyledMessage {
                data: msg.data.clone(),
                style: formatter.clone(),
            })
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
            .map(|msg| StyledMessage {
                data: msg.data.clone(),
                style: formatter.clone(),
            })
            .collect::<Vec<_>>();

        let out = serde_json::to_string(&msgs_in).unwrap();
        println!("{}", out);
    }
}
