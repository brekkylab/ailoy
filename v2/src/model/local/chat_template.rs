use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use minijinja::{Environment, context};
use minijinja_contrib::{add_to_environment, pycompat::unknown_method_callback};

use crate::{
    cache::{Cache, CacheClaim, CacheContents, TryFromCache},
    utils::BoxFuture,
    value::{Message, MessageStyle, QWEN3_FMT, StyledMessage, ToolDesc},
};

/// Global Environment (initialized once)
static ENV: OnceLock<Mutex<Environment>> = OnceLock::new();

fn get_env<'a>() -> MutexGuard<'a, Environment<'static>> {
    ENV.get_or_init(|| {
        let mut e = Environment::new();
        add_to_environment(&mut e);
        e.set_unknown_method_callback(unknown_method_callback);
        Mutex::new(e)
    })
    .lock()
    .unwrap()
}

#[derive(Debug, Clone)]
pub struct ChatTemplate {
    key: String,
    style: MessageStyle,
    do_reasoning: Arc<Mutex<bool>>,
}

impl ChatTemplate {
    pub fn new(key: String, source: String) -> Self {
        let mut env = get_env();
        if env.get_template(&key).is_err() {
            env.add_template_owned(key.clone(), source).unwrap();
        }

        // @jhlee: TODO How to remove hard-coded format determination?
        let style = if key.to_lowercase().starts_with("qwen--qwen3") {
            QWEN3_FMT.clone()
        } else {
            MessageStyle::default()
        };

        Self {
            key,
            style,
            do_reasoning: Arc::new(Mutex::new(true)),
        }
    }

    /// Only affects to hybrid reasoning models
    pub fn enable_reasoning(&self) {
        let mut v = self.do_reasoning.lock().unwrap();
        *v = true;
    }

    /// Only affects to hybrid reasoning models
    pub fn disable_reasoning(&self) {
        let mut v = self.do_reasoning.lock().unwrap();
        *v = false;
    }

    pub fn apply(
        &self,
        messages: impl IntoIterator<Item = Message>,
        tools: impl IntoIterator<Item = ToolDesc>,
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        use serde_json::json;

        let messages = messages
            .into_iter()
            .map(|v| crate::value::StyledMessage {
                data: v.clone(),
                style: self.style.clone(),
            })
            .collect::<Vec<_>>();
        let tools = tools
            .into_iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": tool
                })
            })
            .collect::<Vec<_>>();
        let do_reasoning = *self.do_reasoning.lock().unwrap();

        let ctx = if tools.is_empty() {
            context!(messages=>messages, add_generation_prompt=>add_generation_prompt, enable_thinking=>do_reasoning)
        } else {
            context!(messages=>messages, tools=>tools, add_generation_prompt=>add_generation_prompt, enable_thinking=>do_reasoning)
        };
        get_env()
            .get_template(&self.key)
            .unwrap()
            .render(ctx)
            .map_err(|e| format!("minijinja::render failed: {}", e.to_string()))
    }

    pub fn get_styled(&self, message: Message) -> StyledMessage {
        StyledMessage {
            data: message,
            style: self.style.clone(),
        }
    }
}

impl TryFromCache for ChatTemplate {
    fn claim_files(
        _: Cache,
        key: impl AsRef<str>,
    ) -> BoxFuture<'static, Result<CacheClaim, String>> {
        let dirname = key.as_ref().replace("/", "--");
        Box::pin(async move { Ok(CacheClaim::new([(dirname.as_str(), "chat_template.j2")])) })
    }

    fn try_from_contents(mut contents: CacheContents) -> BoxFuture<'static, Result<Self, String>> {
        Box::pin(async move {
            let Some((entry, bytes)) = contents.remove_with_filename("chat_template.j2") else {
                return Err("chat_template.j2 not exists".to_owned());
            };
            let s =
                std::str::from_utf8(&bytes).map_err(|_| "Utf-8 conversion failed".to_owned())?;
            Ok(ChatTemplate::new(entry.path(), s.to_owned()))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::LazyLock, vec};

    use ailoy_macros::multi_platform_test;
    use serde_json::json;

    use super::*;
    use crate::value::{Part, Role, ToolDesc};

    static TOOLS: LazyLock<Vec<ToolDesc>> = LazyLock::new(|| {
        vec![
            ToolDesc::new(
                "get_current_temperature".into(),
                "Get the current temperature at a location.".into(),
                json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the temperature for, in the format \"City, Country\""
                        },
                        "unit": {
                            "type": "string",
                            "description": "The unit to return the temperature in.",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location", "unit"]
                }),
                Some(json!({
                    "type": "number",
                    "description": "The current temperature at the specified location in the specified units, as a float.",
                })),
            ).unwrap(),
            ToolDesc::new(
                "get_current_wind_speed".into(),
                "Get the current wind speed in km/h at a given location.".into(),
                json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the wind speed for, in the format \"City, Country\""
                        },
                    },
                    "required": ["location"]
                }),
                Some(json!({
                    "type": "number",
                    "description": "The current wind speed at the given location in km/h, as a float.",
                })),
            ).unwrap(),
        ]
    });

    fn setup_input(index: i32) -> ((Vec<Message>, Vec<ToolDesc>, bool), &'static str) {
        let msgs = match index {
            0 => ((vec![
                Message::with_role(Role::User).with_contents(vec![Part::Text(
                    "Introduce yourself in one sentence.".to_owned(),
                )]),
            ], vec![], false), "<|im_start|>user\nIntroduce yourself in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            1 => ((vec![
                Message::with_role(Role::System).with_contents(vec![Part::Text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                        .to_owned(),
                )]),
                Message::with_role(Role::User).with_contents(vec![Part::Text(
                    "Who are you? Answer it simply.".to_owned(),
                )]),
            ], vec![], false), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWho are you? Answer it simply.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            2 => ((vec![
                Message::with_role(Role::System).with_contents(vec![Part::Text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                        .to_owned(),
                )]),
                Message::with_role(Role::User).with_contents(vec![Part::Text(
                    "Which is bigger, a virus or a bacterium?".to_owned(),
                )]),
                Message::with_role(Role::Assistant)
                    .with_contents(vec![Part::Text("A bacterium.".to_owned())]),
            ], vec![], true), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhich is bigger, a virus or a bacterium?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nA bacterium.<|im_end|>\n<|im_start|>assistant\n"),
            3 => ((vec![
                Message::with_role(Role::System).with_contents(vec![Part::Text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                        .to_owned(),
                )]),
                Message::with_role(Role::User)
                    .with_contents(vec![Part::Text("Introduce yourself.".to_owned())]),
                Message::with_role(Role::Assistant)
                    .with_reasoning("\nOkay, the user wants me to introduce myself. Let me start by acknowledging their request. I should be friendly and open. I can say something like, \"Hi, I'm an AI assistant here. I'm here to help you with your questions!\" That's a good start. Now, I need to add some personal details or a brief introduction. Maybe mention my name or a trait, like being a language model. Let me check if that's needed. Oh, the user might want to know more about their role. So, include that. Make sure it's concise and positive. Alright, that should cover it.\n")
                    .with_contents(vec![Part::Text("Hi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊".to_owned())]),
            ], vec![], true), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nIntroduce yourself.<|im_end|>\n<|im_start|>assistant\n<think>\nOkay, the user wants me to introduce myself. Let me start by acknowledging their request. I should be friendly and open. I can say something like, \"Hi, I'm an AI assistant here. I'm here to help you with your questions!\" That's a good start. Now, I need to add some personal details or a brief introduction. Maybe mention my name or a trait, like being a language model. Let me check if that's needed. Oh, the user might want to know more about their role. So, include that. Make sure it's concise and positive. Alright, that should cover it.\n</think>\n\nHi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊<|im_end|>\n<|im_start|>assistant\n"),
            4 => ((vec![
                Message::with_role(Role::System).with_contents(vec![Part::Text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                        .to_owned(),
                )]),
                Message::with_role(Role::User)
                    .with_contents(vec![Part::Text("Introduce yourself.".to_owned())]),
                Message::with_role(Role::Assistant)
                    .with_contents(vec![Part::Text("<think>\nOkay, the user wants me to introduce myself. Let me start by acknowledging their request. I should be friendly and open. I can say something like, \"Hi, I'm an AI assistant here. I'm here to help you with your questions!\" That's a good start. Now, I need to add some personal details or a brief introduction. Maybe mention my name or a trait, like being a language model. Let me check if that's needed. Oh, the user might want to know more about their role. So, include that. Make sure it's concise and positive. Alright, that should cover it.\n</think>\n\nHi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊".to_owned())]),
                    Message::with_role(Role::User)
                    .with_contents(vec![Part::Text("I love you.".to_owned())]),                
            ], vec![], true), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nIntroduce yourself.<|im_end|>\n<|im_start|>assistant\nHi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊<|im_end|>\n<|im_start|>user\nI love you.<|im_end|>\n<|im_start|>assistant\n"),
            5 => ((vec![
                Message::with_role(Role::System).with_contents(vec![Part::Text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                        .to_owned(),
                )]),
                Message::with_role(Role::User)
                    .with_contents(vec![Part::Text("Who are you? Answer it simply.".to_owned())]),
                Message::with_role(Role::Assistant)
                    .with_contents(vec![Part::Text("I am Qwen, a helpful assistant created by Alibaba Cloud.".to_owned())]),
                    Message::with_role(Role::User)
                    .with_contents(vec![Part::Text("Repeat it.".to_owned())]),                
            ], vec![], false), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWho are you? Answer it simply.<|im_end|>\n<|im_start|>assistant\nI am Qwen, a helpful assistant created by Alibaba Cloud.<|im_end|>\n<|im_start|>user\nRepeat it.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            6 => ((vec![
                Message::with_role(Role::System).with_contents(vec![Part::Text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                        .to_owned(),
                )]),
                Message::with_role(Role::User)
                    .with_contents(vec![Part::Text("How is the current weather in Seoul?".to_owned())]),
            ], TOOLS.clone(), false), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_temperature\",\"description\":\"Get the current temperature at a location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the temperature for, in the format \\\"City, Country\\\"\"},\"unit\":{\"type\":\"string\",\"description\":\"The unit to return the temperature in.\",\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"location\",\"unit\"]}}}\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_wind_speed\",\"description\":\"Get the current wind speed in km/h at a given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the wind speed for, in the format \\\"City, Country\\\"\"}},\"required\":[\"location\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nHow is the current weather in Seoul?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            7 => ((vec![
                Message::with_role(Role::System).with_contents(vec![Part::Text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                        .to_owned(),
                )]),
                Message::with_role(Role::User)
                    .with_contents(vec![Part::Text("How is the current weather in Seoul?".to_owned())]),
                Message::with_role(Role::Assistant)
                    .with_tool_calls(vec![Part::Function { id: "".to_owned(), name: "get_current_temperature".to_owned(), arguments: r#"{"location": "Seoul, South Korea", "unit": "celsius"}"#.to_owned() }]),
                Message::with_role(Role::Tool).with_tool_call_id("get_current_temperature")
                    .with_contents(vec![Part::Text("20.5".to_owned())]),
            ], TOOLS.clone(), false), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_temperature\",\"description\":\"Get the current temperature at a location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the temperature for, in the format \\\"City, Country\\\"\"},\"unit\":{\"type\":\"string\",\"description\":\"The unit to return the temperature in.\",\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"location\",\"unit\"]}}}\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_wind_speed\",\"description\":\"Get the current wind speed in km/h at a given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the wind speed for, in the format \\\"City, Country\\\"\"}},\"required\":[\"location\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nHow is the current weather in Seoul?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_current_temperature\", \"arguments\": {\"location\": \"Seoul, South Korea\", \"unit\": \"celsius\"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n20.5\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            _ => ((vec![], vec![], false), ""),
        };
        msgs
    }

    #[multi_platform_test]
    async fn test_chat_template() {
        use futures::StreamExt;

        use crate::debug;

        let cache = crate::cache::Cache::new();
        let key = "Qwen/Qwen3-0.6B";

        let mut template_strm = Box::pin(cache.try_create::<ChatTemplate>(key));
        let mut template: Option<ChatTemplate> = None;
        while let Some(progress) = template_strm.next().await {
            let mut progress = progress.unwrap();
            println!(
                "{} ({} / {})",
                progress.comment, progress.current_task, progress.total_task
            );
            if progress.current_task == progress.total_task {
                template = progress.result.take();
            }
        }
        let template = template.unwrap();
        for i in 0..8 {
            let ((msgs, tools, do_reasoning), expected) = setup_input(i);
            if do_reasoning {
                template.enable_reasoning();
            } else {
                template.disable_reasoning();
            }
            let prompt = template.apply(msgs, tools, true);
            debug!("{}", prompt.as_ref().unwrap());
            assert_eq!(prompt.unwrap().as_str(), expected);
        }
    }
}
