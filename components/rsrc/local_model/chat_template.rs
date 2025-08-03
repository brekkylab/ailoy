use std::{
    sync::{Mutex, MutexGuard, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use minijinja::{Environment, context};

use crate::{cache::CacheUse, message::Message};

static ENV: OnceLock<Mutex<Environment>> = OnceLock::new();

fn get_env<'a>() -> MutexGuard<'a, Environment<'static>> {
    ENV.get_or_init(|| {
        let mut inner = Environment::new();
        minijinja_contrib::add_to_environment(&mut inner);
        inner.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        Mutex::new(inner)
    })
    .lock()
    .unwrap()
}

#[derive(Debug)]
pub struct ChatTemplate {
    key: String,
}

impl ChatTemplate {
    pub fn new(source: String) -> Self {
        // Uses the current system time (in microseconds) to generate a unique key
        let mut key: String;
        loop {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Unable to get system time");
            key = format!("{}", now.as_micros());
            if get_env().get_template(&key).is_ok() {
                continue;
            }
            break;
        }
        let _ = get_env().add_template_owned(key.clone(), source);
        Self { key }
    }

    pub fn get(&self) -> String {
        get_env()
            .get_template(&self.key)
            .unwrap()
            .source()
            .to_owned()
    }

    pub fn apply_with_vec(&self, messages: &Vec<Message>, add_generation_prompt: bool) -> String {
        let env = get_env();
        let template = env.get_template(&self.key).unwrap();
        template
            .render(context!(messages => messages, add_generation_prompt=>add_generation_prompt))
            .unwrap()
    }

    pub fn apply_with_json(&self, messages: &str, add_generation_prompt: bool) -> String {
        let messages: Vec<Message> = serde_json::from_str(messages).unwrap();
        self.apply_with_vec(&messages, add_generation_prompt)
    }

    pub fn apply<I>(&self, messages: I, add_generation_prompt: bool) -> String
    where
        I: IntoIterator<Item = Message>,
    {
        let messages: Vec<Message> = messages.into_iter().collect();
        self.apply_with_vec(&messages, add_generation_prompt)
    }
}

impl Drop for ChatTemplate {
    fn drop(&mut self) {
        get_env().remove_template(&self.key);
    }
}

mod ffi {
    use std::ffi::{c_char, c_int};

    use super::*;
    use crate::ffi::{from_const_char, to_char};

    #[unsafe(no_mangle)]
    pub extern "C" fn ailoy_chat_template_create(
        source: *const c_char,
        tmpl: *mut *mut ChatTemplate,
    ) -> c_int {
        let source = match from_const_char(source) {
            Ok(v) => v,
            Err(_) => return 1,
        };
        let t = Box::new(ChatTemplate::new(source.to_owned()));
        unsafe { *tmpl = Box::into_raw(t) };
        0
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn ailoy_chat_template_destroy(tmpl: *mut ChatTemplate) -> c_int {
        if tmpl.is_null() {
            return 1;
        }
        unsafe {
            let _ = Box::from_raw(tmpl);
        }
        0
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn ailoy_chat_template_get(
        tmpl: *const ChatTemplate,
        source: *mut *mut c_char,
    ) -> c_int {
        let tmpl = unsafe {
            match tmpl.as_ref() {
                Some(v) => v,
                None => return 1,
            }
        };
        unsafe { *source = to_char(&tmpl.get()) };
        0
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn ailoy_chat_template_apply(
        tmpl: *const ChatTemplate,
        messages: *const c_char,
        out: *mut *mut c_char,
    ) -> c_int {
        let tmpl = unsafe {
            match tmpl.as_ref() {
                Some(v) => v,
                None => return 1,
            }
        };
        let messages = match from_const_char(messages) {
            Ok(v) => v,
            Err(_) => return 2,
        };
        let rendered = tmpl.apply_with_json(messages, true);
        unsafe { *out = to_char(rendered.as_str()) };
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Part, Role};

    const QWEN3_CHAT_TEMPLATE: &str = include_str!("../../data/Qwen--Qwen3-0.6B/chat_template.j2");

    #[test]
    fn test_qwen3_no_reasoning() {
        let ct = ChatTemplate::new(QWEN3_CHAT_TEMPLATE.to_owned());
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
