use std::sync::Arc;

use futures::future::BoxFuture;
use indexmap::IndexMap;

use crate::model::InferLangModel;

/// Makes the models of the orgs it is set for, where the agent runs.
pub trait LangModelSet: Send + Sync {
    /// How it runs them, as `get_lang_model_provider` answers it: `"cli"`, `"api"`,
    /// `"bedrock"`, `"openrouter"`, `"self-hosted"` or `"custom"`.
    fn name(&self) -> String;

    /// Make `name` of `org`, `None` when the desc names only the org.
    fn make(
        &self,
        org: &str,
        name: Option<&str>,
    ) -> BoxFuture<'static, anyhow::Result<Arc<dyn InferLangModel>>>;
}

/// A model of your own, made by a function of this process from the desc's org and the name
/// after it.
impl<F> LangModelSet for F
where
    F: Fn(&str, Option<&str>) -> anyhow::Result<Arc<dyn InferLangModel>> + Send + Sync,
{
    fn name(&self) -> String {
        "custom".to_owned()
    }

    fn make(
        &self,
        org: &str,
        name: Option<&str>,
    ) -> BoxFuture<'static, anyhow::Result<Arc<dyn InferLangModel>>> {
        Box::pin(std::future::ready(self(org, name)))
    }
}

/// The models a spec may name, by the set each org's models are made through.
#[derive(Default)]
pub struct LangModelProvider {
    /// The set each org's models are made through, by the org.
    models: IndexMap<String, Arc<dyn LangModelSet>>,
}

impl LangModelProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Make the models of `org` through `set`, in place of the set it had.
    pub fn set(&mut self, org: impl Into<String>, set: impl LangModelSet + 'static) {
        self.models.insert(org.into(), Arc::new(set));
    }

    /// The set a desc's model is made through, the one set for its org, if there is one. The
    /// desc is `<org>` or `<org>/<name>`.
    pub fn get(&self, desc: &str) -> Option<Arc<dyn LangModelSet>> {
        let org = desc.split_once('/').map_or(desc, |(org, _)| org);
        self.models.get(org).cloned()
    }

    /// Stop making the models of `org`. Answers whether there was a set for it.
    pub fn remove(&mut self, org: &str) -> bool {
        self.models.shift_remove(org).is_some()
    }

    /// Make the model a desc of the form `<org>` or `<org>/<name>` names, through the set for
    /// its org. The answer is a future of its own, so a lock held on this provider need not be
    /// held while it is made.
    pub fn create(
        &self,
        desc: &str,
    ) -> anyhow::Result<BoxFuture<'static, anyhow::Result<Arc<dyn InferLangModel>>>> {
        let (org, name) = match desc.split_once('/') {
            Some((org, name)) => (org, Some(name)),
            None => (desc, None),
        };
        let set = self
            .models
            .get(org)
            .ok_or_else(|| anyhow::anyhow!("no provider set for lang model org '{org}'"))?;
        Ok(set.make(org, name))
    }
}
