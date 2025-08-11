use std::{
    pin::Pin,
    sync::{Mutex, MutexGuard, OnceLock},
};

use minijinja::{Environment, context};
use minijinja_contrib::{add_to_environment, pycompat::unknown_method_callback};

use crate::{
    cache::{Cache, CacheContents, CacheEntry, TryFromCache},
    value::{Message, ToolDescription},
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

    pub fn apply_with_vec(
        &self,
        tools: &Vec<ToolDescription>,
        messages: &Vec<Message>,
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        get_env()
            .get_template(&self.key)
            .unwrap()
            .render(context!(messages => messages, tools => tools, add_generation_prompt=>add_generation_prompt))
            .map_err(|e| format!("minijinja::render failed: {}", e.to_string()))
    }
}

impl TryFromCache for ChatTemplate {
    fn claim_files(
        _: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CacheEntry>, String>>>> {
        let dirname = key.as_ref().replace("/", "--");
        Box::pin(async move { Ok(vec![CacheEntry::new(dirname, "chat_template.j2")]) })
    }

    fn try_from_contents(contents: &mut CacheContents) -> Result<Self, String> {
        let Some((entry, bytes)) = contents.remove_with_filename("chat_template.j2") else {
            return Err("chat_template.j2 not exists".to_owned());
        };
        let s = std::str::from_utf8(&bytes).map_err(|_| "Utf-8 conversion failed".to_owned())?;
        Ok(ChatTemplate::new(entry.path(), s.to_owned()))
    }
}
