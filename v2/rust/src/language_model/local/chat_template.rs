use std::{
    pin::Pin,
    sync::{Mutex, MutexGuard, OnceLock},
};

use minijinja::{Environment, context};
use minijinja_contrib::{add_to_environment, pycompat::unknown_method_callback};

use crate::{
    cache::{Cache, CacheElement, TryFromCache},
    message::Message,
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

#[derive(Debug)]
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

    pub fn apply_with_vec(
        &self,
        messages: &Vec<Message>,
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        get_env()
            .get_template(&self.key)
            .unwrap()
            .render(context!(messages => messages, add_generation_prompt=>add_generation_prompt))
            .map_err(|e| format!("minijinja::render failed: {}", e.to_string()))
    }
}

impl TryFromCache for ChatTemplate {
    fn claim_files(
        _: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CacheElement>, String>>>> {
        let dirname = key.as_ref().replace("/", "--");
        Box::pin(async move { Ok(vec![CacheElement::new(dirname, "chat_template.j2")]) })
    }

    fn try_from_files(_: &Cache, files: Vec<(CacheElement, Vec<u8>)>) -> Result<Self, String> {
        let (elem, bytes) = files.get(0).unwrap();
        let s = std::str::from_utf8(&bytes).map_err(|_| "Utf-8 conversion failed".to_owned())?;
        Ok(ChatTemplate::new(elem.path(), s.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Part, Role};

    const QWEN3_CHAT_TEMPLATE: &str = include_str!("./data/Qwen--Qwen3-0.6B/chat_template.j2");

    #[test]
    fn test_qwen3_no_reasoning() {
        let ct = ChatTemplate::new("Qwen3".to_owned(), QWEN3_CHAT_TEMPLATE.to_owned());
        let msgs = vec![
            Message::with_content(Role::System, Part::from_text("You are an assistant.")),
            Message::with_content(Role::User, Part::from_text("Hi what's your name?")),
            Message::with_content(Role::Assistant, Part::from_text("You can call me Jaden.")),
            Message::with_content(Role::User, Part::from_text("Who made you?")),
        ];
        let expected = r#"<|im_start|>system
You are an assistant.<|im_end|>
<|im_start|>user
Hi what's your name?<|im_end|>
<|im_start|>assistant
<think>

</think>

You can call me Jaden.<|im_end|>
<|im_start|>user
Who made you?<|im_end|>
<|im_start|>assistant
<think>

</think>

"#;
        let result = ct.apply_with_vec(&msgs, true).unwrap();
        assert_eq!(expected, result);
    }
}
