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
    do_reasoning: Arc<Mutex<bool>>,
}

impl ChatTemplate {
    pub fn new(key: String, source: String) -> Self {
        let mut env = get_env();
        if env.get_template(&key).is_err() {
            env.add_template_owned(key.clone(), source).unwrap();
        }

        Self {
            key,
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
    ) -> anyhow::Result<String> {
        let messages = messages.into_iter().collect::<Vec<_>>();
        let tools = tools.into_iter().collect::<Vec<_>>();
        let do_reasoning = *self.do_reasoning.lock().unwrap();
        let ctx = if tools.is_empty() {
            context!(messages => messages, add_generation_prompt=>add_generation_prompt, enable_thinking=>do_reasoning)
        } else {
            context!(messages => messages, tools => tools, add_generation_prompt=>add_generation_prompt, enable_thinking=>do_reasoning)
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
