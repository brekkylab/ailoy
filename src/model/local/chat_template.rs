use std::sync::{Mutex, MutexGuard, OnceLock};

use anyhow::{Context, bail};
use minijinja::{Environment, context};
use minijinja_contrib::{add_to_environment, pycompat::unknown_method_callback};

use crate::{
    cache::{Cache, CacheClaim, CacheContents, TryFromCache},
    model::ThinkEffort,
    utils::BoxFuture,
    value::{Document, Message, ToolDesc},
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
}

impl ChatTemplate {
    pub fn new(key: String, source: String) -> Self {
        let mut env = get_env();
        if env.get_template(&key).is_err() {
            env.add_template_owned(key.clone(), source).unwrap();
        }

        Self { key }
    }

    pub fn apply(
        &self,
        messages: impl IntoIterator<Item = Message>,
        tools: impl IntoIterator<Item = ToolDesc>,
        documents: impl IntoIterator<Item = Document>,
        think_effort: ThinkEffort,
        add_generation_prompt: bool,
    ) -> anyhow::Result<String> {
        let messages = messages.into_iter().collect::<Vec<_>>();
        let tools = tools.into_iter().collect::<Vec<_>>();
        let documents = documents.into_iter().collect::<Vec<_>>();
        let enable_thinking = match think_effort {
            ThinkEffort::Disable => false,
            _ => true,
        };
        let reasoning_effort = match think_effort {
            ThinkEffort::Low => "low",
            ThinkEffort::Medium => "medium",
            ThinkEffort::High => "high",
            _ => "",
        };
        let ctx = context!(
            messages => messages,
            tools => if !tools.is_empty() { Some(tools) } else { None::<_> },
            documents => if !documents.is_empty() { Some(documents) } else { None::<_> },
            add_generation_prompt => add_generation_prompt,
            enable_thinking => enable_thinking,
            reasoning_effort => if !reasoning_effort.is_empty() { Some(reasoning_effort) } else { None::<_> },
        );
        get_env()
            .get_template(&self.key)
            .unwrap()
            .render(ctx)
            .context("minijinja::render failed")
    }
}

impl<'this> TryFromCache<'this> for ChatTemplate {
    fn claim_files<'a: 'this>(
        _: Cache,
        key: impl AsRef<str>,
        _: &'a mut std::collections::HashMap<String, crate::value::Value>,
    ) -> BoxFuture<'a, anyhow::Result<CacheClaim>> {
        let dirname = key.as_ref().replace("/", "--");
        Box::pin(async move { Ok(CacheClaim::new([(dirname.as_str(), "chat_template.j2")])) })
    }

    fn try_from_contents<'a: 'this>(
        contents: &'a mut CacheContents,
        _: &'a std::collections::HashMap<String, crate::value::Value>,
    ) -> BoxFuture<'a, anyhow::Result<Self>> {
        Box::pin(async move {
            let Some((entry, source)) = contents.remove_with_filename("chat_template.j2") else {
                bail!("chat_template.j2 not exists")
            };
            let bytes = source.read_all().await?;
            let s = std::str::from_utf8(&bytes).context("Utf-8 conversion failed")?;
            Ok(ChatTemplate::new(entry.path(), s.to_owned()))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::LazyLock, vec};

    use ailoy_macros::multi_platform_test;
    use yare::parameterized;

    use super::*;
    use crate::{
        debug, to_value,
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

    #[parameterized(
        _0_only_user_message = {
            vec![
                Message::new(Role::User).with_contents(vec![Part::text(
                    "Introduce yourself in one sentence.",
                )]),
            ],
            vec![],
            "",
            "<|im_start|>user\nIntroduce yourself in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        },
        _1_system_and_user_message = {
            vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User).with_contents(vec![Part::text(
                    "Who are you? Answer it simply.",
                )]),
            ],
            vec![],
            "",
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWho are you? Answer it simply.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        },
        _2_thinking_effort = {
            vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User).with_contents(vec![Part::text(
                    "Which is bigger, a virus or a bacterium?",
                )]),
                Message::new(Role::Assistant)
                    .with_contents(vec![Part::text("A bacterium.")]),
            ],
            vec![],
            "medium",
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhich is bigger, a virus or a bacterium?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nA bacterium.<|im_end|>\n<|im_start|>assistant\n"
        },
        _3_message_with_thinking = {
            vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User)
                    .with_contents(vec![Part::text("Introduce yourself.")]),
                Message::new(Role::Assistant)
                    .with_thinking("\nOkay, the user wants me to introduce myself. Let me start by acknowledging their request. I should be friendly and open. I can say something like, \"Hi, I'm an AI assistant here. I'm here to help you with your questions!\" That's a good start. Now, I need to add some personal details or a brief introduction. Maybe mention my name or a trait, like being a language model. Let me check if that's needed. Oh, the user might want to know more about their role. So, include that. Make sure it's concise and positive. Alright, that should cover it.\n")
                    .with_contents(vec![Part::text("Hi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊")]),
            ],
            vec![],
            "medium",
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nIntroduce yourself.<|im_end|>\n<|im_start|>assistant\n<think>\nOkay, the user wants me to introduce myself. Let me start by acknowledging their request. I should be friendly and open. I can say something like, \"Hi, I'm an AI assistant here. I'm here to help you with your questions!\" That's a good start. Now, I need to add some personal details or a brief introduction. Maybe mention my name or a trait, like being a language model. Let me check if that's needed. Oh, the user might want to know more about their role. So, include that. Make sure it's concise and positive. Alright, that should cover it.\n</think>\n\nHi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊<|im_end|>\n<|im_start|>assistant\n"
        },
        _4_message_with_thinking_in_content = {
            vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User)
                    .with_contents(vec![Part::text("Introduce yourself.")]),
                Message::new(Role::Assistant)
                    .with_contents(vec![Part::text("<think>\nOkay, the user wants me to introduce myself. Let me start by acknowledging their request. I should be friendly and open. I can say something like, \"Hi, I'm an AI assistant here. I'm here to help you with your questions!\" That's a good start. Now, I need to add some personal details or a brief introduction. Maybe mention my name or a trait, like being a language model. Let me check if that's needed. Oh, the user might want to know more about their role. So, include that. Make sure it's concise and positive. Alright, that should cover it.\n</think>\n\nHi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊")]),
            ],
            vec![],
            "medium",
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nIntroduce yourself.<|im_end|>\n<|im_start|>assistant\n<think>\nOkay, the user wants me to introduce myself. Let me start by acknowledging their request. I should be friendly and open. I can say something like, \"Hi, I'm an AI assistant here. I'm here to help you with your questions!\" That's a good start. Now, I need to add some personal details or a brief introduction. Maybe mention my name or a trait, like being a language model. Let me check if that's needed. Oh, the user might want to know more about their role. So, include that. Make sure it's concise and positive. Alright, that should cover it.\n</think>\n\nHi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊<|im_end|>\n<|im_start|>assistant\n"
        },
        _5_omit_thinking_before_last_query = {
            vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User)
                    .with_contents(vec![Part::text("Introduce yourself.")]),
                Message::new(Role::Assistant)
                    .with_contents(vec![Part::text("<think>\nOkay, the user wants me to introduce myself. Let me start by acknowledging their request. I should be friendly and open. I can say something like, \"Hi, I'm an AI assistant here. I'm here to help you with your questions!\" That's a good start. Now, I need to add some personal details or a brief introduction. Maybe mention my name or a trait, like being a language model. Let me check if that's needed. Oh, the user might want to know more about their role. So, include that. Make sure it's concise and positive. Alright, that should cover it.\n</think>\n\nHi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊")]),
                Message::new(Role::User)
                    .with_contents(vec![Part::text("I love you.")]),
            ],
            vec![],
            "medium",
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nIntroduce yourself.<|im_end|>\n<|im_start|>assistant\nHi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊<|im_end|>\n<|im_start|>user\nI love you.<|im_end|>\n<|im_start|>assistant\n"
        },
        _6_using_tools = {
            vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User)
                    .with_contents(vec![Part::text("How is the current weather in Seoul?")]),
            ],
            TOOLS.clone(),
            "",
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_temperature\",\"description\":\"Get the current temperature at a location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the temperature for, in the format \\\"City, Country\\\"\"},\"unit\":{\"type\":\"string\",\"description\":\"The unit to return the temperature in.\",\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"location\",\"unit\"]},\"returns\":{\"type\":\"number\",\"description\":\"The current temperature at the specified location in the specified units, as a float.\"}}}\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_wind_speed\",\"description\":\"Get the current wind speed in km/h at a given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the wind speed for, in the format \\\"City, Country\\\"\"}},\"required\":[\"location\"]},\"returns\":{\"type\":\"number\",\"description\":\"The current wind speed at the given location in km/h, as a float.\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nHow is the current weather in Seoul?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        },
        _7_tool_response_message = {
            vec![
                Message::new(Role::System).with_contents(vec![Part::text(
                    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                )]),
                Message::new(Role::User)
                    .with_contents(vec![Part::text("How is the current weather in Seoul?")]),
                Message::new(Role::Assistant)
                    .with_tool_calls(vec![Part::function("get_current_temperature", to_value!({"location": "Seoul, South Korea", "unit": "celsius"}))]),
                Message::new(Role::Tool).with_id("get_current_temperature")
                    .with_contents(vec![Part::text("20.5")]),
            ],
            TOOLS.clone(),
            "",
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_temperature\",\"description\":\"Get the current temperature at a location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the temperature for, in the format \\\"City, Country\\\"\"},\"unit\":{\"type\":\"string\",\"description\":\"The unit to return the temperature in.\",\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"location\",\"unit\"]},\"returns\":{\"type\":\"number\",\"description\":\"The current temperature at the specified location in the specified units, as a float.\"}}}\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_wind_speed\",\"description\":\"Get the current wind speed in km/h at a given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the wind speed for, in the format \\\"City, Country\\\"\"}},\"required\":[\"location\"]},\"returns\":{\"type\":\"number\",\"description\":\"The current wind speed at the given location in km/h, as a float.\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nHow is the current weather in Seoul?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_current_temperature\", \"arguments\": {\"location\":\"Seoul, South Korea\",\"unit\":\"celsius\"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n20.5\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        },
        _8_parallel_tool_responses = {
            vec![
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
            ],
            TOOLS.clone(),
            "",
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_temperature\",\"description\":\"Get the current temperature at a location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the temperature for, in the format \\\"City, Country\\\"\"},\"unit\":{\"type\":\"string\",\"description\":\"The unit to return the temperature in.\",\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"location\",\"unit\"]},\"returns\":{\"type\":\"number\",\"description\":\"The current temperature at the specified location in the specified units, as a float.\"}}}\n{\"type\":\"function\",\"function\":{\"name\":\"get_current_wind_speed\",\"description\":\"Get the current wind speed in km/h at a given location.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\",\"description\":\"The location to get the wind speed for, in the format \\\"City, Country\\\"\"}},\"required\":[\"location\"]},\"returns\":{\"type\":\"number\",\"description\":\"The current wind speed at the given location in km/h, as a float.\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nHow is the current weather in Seoul?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_current_temperature\", \"arguments\": {\"location\":\"Seoul, South Korea\",\"unit\":\"celsius\"}}\n</tool_call><|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_current_wind_speed\", \"arguments\": {\"location\":\"Seoul, South Korea\"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n20.5\n</tool_response>\n<tool_response>\n30\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        }
    )]
    #[test_macro(multi_platform_test)]
    async fn test_chat_template(
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        think_effort: &'static str,
        expected: &'static str,
    ) {
        use futures::StreamExt;

        let cache = Cache::new();
        let key = "Qwen/Qwen3-0.6B";

        let mut template_strm = Box::pin(cache.try_create::<ChatTemplate>(key, None, None));
        let mut template: Option<ChatTemplate> = None;
        while let Some(progress) = template_strm.next().await {
            let mut progress = progress.unwrap();
            debug!(
                "{} ({} / {})",
                progress.comment, progress.current_task, progress.total_task
            );
            if progress.current_task == progress.total_task {
                template = progress.result.take();
            }
        }
        assert!(template.is_some());
        let template = template.unwrap();

        let think_effort: ThinkEffort = if !think_effort.is_empty() {
            use std::str::FromStr;

            ThinkEffort::from_str(think_effort).unwrap()
        } else {
            ThinkEffort::Disable
        };
        let prompt = template.apply(msgs, tools, Vec::new(), think_effort, true);
        assert_eq!(prompt.unwrap().as_str(), expected);
    }
}
