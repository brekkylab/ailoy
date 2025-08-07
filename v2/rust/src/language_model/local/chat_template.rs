use std::pin::Pin;

use minijinja::{Environment, context};
use minijinja_contrib::add_to_environment;

use crate::{
    cache::{Cache, CacheElement, TryFromCache},
    message::Message,
};

#[derive(Debug)]
pub struct ChatTemplate<'a> {
    env: Environment<'a>,
}

impl<'a> ChatTemplate<'a> {
    pub fn new(source: &str) -> Self {
        let mut env = Environment::new();
        add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("_default", source.to_owned())
            .unwrap();
        Self { env }
    }

    pub fn get(&self) -> String {
        self.env
            .get_template("_default")
            .unwrap()
            .source()
            .to_owned()
    }

    pub fn apply_with_vec(
        &self,
        messages: &Vec<Message>,
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        let template = self.env.get_template("_default").unwrap();
        template
            .render(context!(messages => messages, add_generation_prompt=>add_generation_prompt))
            .map_err(|e| format!("minijinja::render failed: {}", e.to_string()))
    }

    pub fn apply_with_json(
        &self,
        messages: &str,
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        let messages: Vec<Message> = serde_json::from_str(messages).unwrap();
        self.apply_with_vec(&messages, add_generation_prompt)
    }

    pub fn apply(
        &self,
        messages: impl IntoIterator<Item = Message>,
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        let messages: Vec<Message> = messages.into_iter().collect();
        self.apply_with_vec(&messages, add_generation_prompt)
    }
}

impl<'a> TryFromCache for ChatTemplate<'a> {
    fn claim_files(
        _: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CacheElement>, String>>>> {
        let dirname = key.as_ref().replace("/", "--");
        Box::pin(async move { Ok(vec![CacheElement::new(dirname, "chat_template.j2")]) })
    }

    fn try_from_files(_: &Cache, files: Vec<(CacheElement, Vec<u8>)>) -> Result<Self, String> {
        let v = files.get(0).unwrap();
        let v = std::str::from_utf8(&v.1).map_err(|_| "Utf-8 conversion failed".to_owned())?;
        Ok(ChatTemplate::new(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Part, Role};

    const QWEN3_CHAT_TEMPLATE: &str = include_str!("./data/Qwen--Qwen3-0.6B/chat_template.j2");

    #[test]
    fn test_qwen3_no_reasoning() {
        let ct = ChatTemplate::new(QWEN3_CHAT_TEMPLATE);
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
