use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use anyhow::{Context, bail};
use minijinja::{Environment, context};
use minijinja_contrib::{add_to_environment, pycompat::unknown_method_callback};

use crate::{
    cache::{Cache, CacheClaim, CacheContents, TryFromCache},
    utils::BoxFuture,
    value::{Message, ToolDesc},
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
    do_thinking: Arc<Mutex<bool>>,
    think_effort: Arc<Mutex<String>>,
}

impl ChatTemplate {
    pub fn new(key: String, source: String) -> Self {
        let mut env = get_env();
        if env.get_template(&key).is_err() {
            env.add_template_owned(key.clone(), source).unwrap();
        }

        Self {
            key,
            do_thinking: Arc::new(Mutex::new(true)),
            think_effort: Arc::new(Mutex::new("".to_owned())),
        }
    }

    pub fn set_think_effort(&self, think_effort: impl Into<String>) {
        let mut v = self.think_effort.lock().unwrap();
        *v = think_effort.into();
        self.enable_thinking();
    }

    /// Only affects to hybrid thinking models
    pub fn enable_thinking(&self) {
        let mut v = self.do_thinking.lock().unwrap();
        *v = true;
    }

    /// Only affects to hybrid thinking models
    pub fn disable_thinking(&self) {
        let mut v = self.do_thinking.lock().unwrap();
        *v = false;
    }

    pub fn apply(
        &self,
        messages: impl IntoIterator<Item = Message>,
        tools: impl IntoIterator<Item = ToolDesc>,
        add_generation_prompt: bool,
    ) -> anyhow::Result<String> {
        let messages = messages.into_iter().collect::<Vec<_>>();
        let tools = tools.into_iter().collect::<Vec<_>>();
        let do_thinking = *self.do_thinking.lock().unwrap();
        let think_effort = self.think_effort.lock().unwrap().clone();

        let ctx = if tools.is_empty() {
            context!(messages => messages, add_generation_prompt=>add_generation_prompt, enable_thinking=>do_thinking, reasoning_effort=>think_effort)
        } else {
            context!(messages => messages, tools => tools, add_generation_prompt=>add_generation_prompt, enable_thinking=>do_thinking, reasoning_effort=>think_effort)
        };
        get_env()
            .get_template(&self.key)
            .unwrap()
            .render(ctx)
            .context("minijinja::render failed")
    }
}

impl TryFromCache for ChatTemplate {
    fn claim_files(
        _: Cache,
        key: impl AsRef<str>,
    ) -> BoxFuture<'static, anyhow::Result<CacheClaim>> {
        let dirname = key.as_ref().replace("/", "--");
        Box::pin(async move { Ok(CacheClaim::new([(dirname.as_str(), "chat_template.j2")])) })
    }

    fn try_from_contents(mut contents: CacheContents) -> BoxFuture<'static, anyhow::Result<Self>> {
        Box::pin(async move {
            let Some((entry, bytes)) = contents.remove_with_filename("chat_template.j2") else {
                bail!("chat_template.j2 not exists")
            };
            let s = std::str::from_utf8(&bytes).context("Utf-8 conversion failed")?;
            Ok(ChatTemplate::new(entry.path(), s.to_owned()))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::LazyLock, vec};

    use ailoy_macros::multi_platform_test;

    use super::*;
    use crate::{
        to_value,
        value::{Part, Role, ToolDesc, ToolDescBuilder},
    };

    static TOOLS: LazyLock<Vec<ToolDesc>> = LazyLock::new(|| {
        vec![
            ToolDescBuilder::new("get_current_temperature").description("Get the current temperature at a location.").parameters(to_value!({
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
                })).returns(to_value!({
                    "type": "number",
                    "description": "The current temperature at the specified location in the specified units, as a float.",
                })).build(),
            ToolDescBuilder::new("get_current_wind_speed").description("Get the current wind speed in km/h at a given location.").parameters(to_value!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the wind speed for, in the format \"City, Country\""
                        },
                    },
                    "required": ["location"]
                })).returns(to_value!({
                    "type": "number",
                    "description": "The current wind speed at the given location in km/h, as a float.",
                })).build()
        ]
    });

    fn setup_input(index: i32) -> ((Vec<Message>, Vec<ToolDesc>, &'static str), &'static str) {
        let msgs = match index {
            0 => ((vec![
                Message::new(Role::User).with_contents(vec![Part::text(
                    "Introduce yourself in one sentence.",
                )]),
            ], vec![], ""), "<|im_start|>user\nIntroduce yourself in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            1 => ((vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User).with_contents(vec![Part::text(
                    "Who are you? Answer it simply.",
                )]),
            ], vec![], ""), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWho are you? Answer it simply.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            2 => ((vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                        ,
                )]),
                Message::new(Role::User).with_contents(vec![Part::text(
                    "Which is bigger, a virus or a bacterium?",
                )]),
                Message::new(Role::Assistant)
                    .with_contents(vec![Part::text("A bacterium.")]),
            ], vec![], "medium"), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhich is bigger, a virus or a bacterium?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nA bacterium.<|im_end|>\n<|im_start|>assistant\n"),
            3 => ((vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User)
                    .with_contents(vec![Part::text("Introduce yourself.")]),
                Message::new(Role::Assistant)
                    .with_thinking("\nOkay, the user wants me to introduce myself. Let me start by acknowledging their request. I should be friendly and open. I can say something like, \"Hi, I'm an AI assistant here. I'm here to help you with your questions!\" That's a good start. Now, I need to add some personal details or a brief introduction. Maybe mention my name or a trait, like being a language model. Let me check if that's needed. Oh, the user might want to know more about their role. So, include that. Make sure it's concise and positive. Alright, that should cover it.\n")
                    .with_contents(vec![Part::text("Hi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊")]),
            ], vec![], "medium"), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nIntroduce yourself.<|im_end|>\n<|im_start|>assistant\n<think>\nOkay, the user wants me to introduce myself. Let me start by acknowledging their request. I should be friendly and open. I can say something like, \"Hi, I'm an AI assistant here. I'm here to help you with your questions!\" That's a good start. Now, I need to add some personal details or a brief introduction. Maybe mention my name or a trait, like being a language model. Let me check if that's needed. Oh, the user might want to know more about their role. So, include that. Make sure it's concise and positive. Alright, that should cover it.\n</think>\n\nHi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊<|im_end|>\n<|im_start|>assistant\n"),
            4 => ((vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User)
                    .with_contents(vec![Part::text("Introduce yourself.")]),
                Message::new(Role::Assistant)
                    .with_contents(vec![Part::text("<think>\nOkay, the user wants me to introduce myself. Let me start by acknowledging their request. I should be friendly and open. I can say something like, \"Hi, I'm an AI assistant here. I'm here to help you with your questions!\" That's a good start. Now, I need to add some personal details or a brief introduction. Maybe mention my name or a trait, like being a language model. Let me check if that's needed. Oh, the user might want to know more about their role. So, include that. Make sure it's concise and positive. Alright, that should cover it.\n</think>\n\nHi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊")]),
                Message::new(Role::User)
                    .with_contents(vec![Part::text("I love you.")]),
            ], vec![], "medium"), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nIntroduce yourself.<|im_end|>\n<|im_start|>assistant\nHi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊<|im_end|>\n<|im_start|>user\nI love you.<|im_end|>\n<|im_start|>assistant\n"),
            5 => ((vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User)
                    .with_contents(vec![Part::text("Who are you? Answer it simply.")]),
                Message::new(Role::Assistant)
                    .with_contents(vec![Part::text("I am Qwen, a helpful assistant created by Alibaba Cloud.")]),
                    Message::new(Role::User)
                    .with_contents(vec![Part::text("Repeat it.")]),
            ], vec![], ""), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWho are you? Answer it simply.<|im_end|>\n<|im_start|>assistant\nI am Qwen, a helpful assistant created by Alibaba Cloud.<|im_end|>\n<|im_start|>user\nRepeat it.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            6 => ((vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User)
                    .with_contents(vec![Part::text("How is the current weather in Seoul?")]),
            ], TOOLS.clone(), ""), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_temperature\",\"description\":\"Get the current temperature at a location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the temperature for, in the format \\\"City, Country\\\"\"},\"unit\":{\"type\":\"string\",\"description\":\"The unit to return the temperature in.\",\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"location\",\"unit\"]},\"returns\":{\"type\":\"number\",\"description\":\"The current temperature at the specified location in the specified units, as a float.\"}}}\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_wind_speed\",\"description\":\"Get the current wind speed in km/h at a given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the wind speed for, in the format \\\"City, Country\\\"\"}},\"required\":[\"location\"]},\"returns\":{\"type\":\"number\",\"description\":\"The current wind speed at the given location in km/h, as a float.\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nHow is the current weather in Seoul?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            7 => ((vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User)
                    .with_contents(vec![Part::text("How is the current weather in Seoul?")]),
                Message::new(Role::Assistant)
                    .with_tool_calls(vec![Part::function("get_current_temperature", to_value!({"location": "Seoul, South Korea", "unit": "celsius"}))]),
                Message::new(Role::Tool).with_id("get_current_temperature")
                    .with_contents(vec![Part::text("20.5")]),
            ], TOOLS.clone(), ""), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_temperature\",\"description\":\"Get the current temperature at a location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the temperature for, in the format \\\"City, Country\\\"\"},\"unit\":{\"type\":\"string\",\"description\":\"The unit to return the temperature in.\",\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"location\",\"unit\"]},\"returns\":{\"type\":\"number\",\"description\":\"The current temperature at the specified location in the specified units, as a float.\"}}}\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_wind_speed\",\"description\":\"Get the current wind speed in km/h at a given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the wind speed for, in the format \\\"City, Country\\\"\"}},\"required\":[\"location\"]},\"returns\":{\"type\":\"number\",\"description\":\"The current wind speed at the given location in km/h, as a float.\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nHow is the current weather in Seoul?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_current_temperature\", \"arguments\": {\"location\":\"Seoul, South Korea\",\"unit\":\"celsius\"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n20.5\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            8 => ((vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User)
                    .with_contents(vec![Part::text("How is the current weather in Seoul?")]),
                Message::new(Role::Assistant)
                    .with_tool_calls(vec![Part::function("get_current_temperature", to_value!({"location": "Seoul, South Korea", "unit": "celsius"}))]),
                Message::new(Role::Assistant)
                    .with_tool_calls(vec![Part::function("get_current_wind_speed", to_value!({"location": "Seoul, South Korea"}))]),
                Message::new(Role::Tool).with_id("get_current_temperature")
                    .with_contents(vec![Part::text("20.5")]),
                Message::new(Role::Tool).with_id("get_current_wind_speed")
                    .with_contents(vec![Part::text("30")]),
            ], TOOLS.clone(), ""), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_temperature\",\"description\":\"Get the current temperature at a location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the temperature for, in the format \\\"City, Country\\\"\"},\"unit\":{\"type\":\"string\",\"description\":\"The unit to return the temperature in.\",\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"location\",\"unit\"]},\"returns\":{\"type\":\"number\",\"description\":\"The current temperature at the specified location in the specified units, as a float.\"}}}\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_wind_speed\",\"description\":\"Get the current wind speed in km/h at a given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the wind speed for, in the format \\\"City, Country\\\"\"}},\"required\":[\"location\"]},\"returns\":{\"type\":\"number\",\"description\":\"The current wind speed at the given location in km/h, as a float.\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nHow is the current weather in Seoul?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_current_temperature\", \"arguments\": {\"location\":\"Seoul, South Korea\",\"unit\":\"celsius\"}}\n</tool_call><|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_current_wind_speed\", \"arguments\": {\"location\":\"Seoul, South Korea\"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n20.5\n</tool_response>\n<tool_response>\n30\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            _ => ((vec![], vec![], ""), ""),
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
            // if TARGET_INDEX < 0 || i == TARGET_INDEX {
            let ((msgs, tools, think_effort), expected) = setup_input(i);
            if !think_effort.is_empty() {
                template.set_think_effort(think_effort);
                // template.enable_thinking();
            } else {
                template.disable_thinking();
            }
            let prompt = template.apply(msgs, tools, true);
            debug!("<{}>\n{}", i, prompt.as_ref().unwrap());
            assert_eq!(prompt.unwrap().as_str(), expected);
            // }
        }
    }
}
