use std::{path::PathBuf, pin::Pin};

use minijinja::{Environment, context};

use crate::{
    cache::{Cache, TryFromCache},
    message::Message,
};

#[derive(Debug)]
pub struct ChatTemplate<'a> {
    env: Environment<'a>,
}

impl<'a> ChatTemplate<'a> {
    pub fn new(source: &str) -> Self {
        let mut env = Environment::new();
        let _ = env.add_template_owned("_default", source.to_owned());
        Self { env }
    }

    pub fn get(&self) -> String {
        self.env
            .get_template("_default")
            .unwrap()
            .source()
            .to_owned()
    }

    pub fn apply_with_vec(&self, messages: &Vec<Message>, add_generation_prompt: bool) -> String {
        let template = self.env.get_template("_default").unwrap();
        template
            .render(context!(messages => messages, add_generation_prompt=>add_generation_prompt))
            .unwrap()
    }

    pub fn apply_with_json(&self, messages: &str, add_generation_prompt: bool) -> String {
        let messages: Vec<Message> = serde_json::from_str(messages).unwrap();
        self.apply_with_vec(&messages, add_generation_prompt)
    }

    pub fn apply(
        &self,
        messages: impl IntoIterator<Item = Message>,
        add_generation_prompt: bool,
    ) -> String {
        let messages: Vec<Message> = messages.into_iter().collect();
        self.apply_with_vec(&messages, add_generation_prompt)
    }
}

impl<'a> TryFromCache for ChatTemplate<'a> {
    fn claim_files(
        _: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PathBuf>, String>>>> {
        let dir = key.as_ref().replace("/", "--");
        Box::pin(async move { Ok(vec![PathBuf::from(dir).join("chat_template.j2".to_owned())]) })
    }

    fn try_from_files(files: Vec<(PathBuf, Vec<u8>)>) -> Result<Self, String> {
        let v = files.get(0).unwrap();
        let v = std::str::from_utf8(&v.1).map_err(|_| "Utf-8 conversion failed".to_owned())?;
        Ok(ChatTemplate::new(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Part, Role};

    const QWEN3_CHAT_TEMPLATE: &str =
        include_str!("../../../data/Qwen--Qwen3-0.6B/chat_template.j2");

    #[test]
    fn test_qwen3_no_reasoning() {
        let ct = ChatTemplate::new(QWEN3_CHAT_TEMPLATE);
        let msgs = vec![
            Message::with_content(Role::System, Part::text("You are an assistant.")),
            Message::with_content(Role::User, Part::text("Hi what's your name?")),
            Message::with_content(Role::Assistant, Part::text("You can call me Jaden.")),
            Message::with_content(Role::User, Part::text("Who made you?")),
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
        let result = ct.apply_with_vec(&msgs, true);
        assert_eq!(expected, result);
    }
}
