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

    /// Whether it makes the models of `org` with a key a caller gives, in place of its own
    /// ([`make_with_key`](Self::make_with_key)).
    fn takes_key(&self, _org: &str) -> bool {
        false
    }

    /// Make `name` of `org`, as [`make`](Self::make) does, called with `key`, a caller's, in
    /// place of the one it is set up with. Refused by a set that [`takes_key`](Self::takes_key)
    /// not.
    fn make_with_key(
        &self,
        org: &str,
        _name: Option<&str>,
        _key: &str,
    ) -> BoxFuture<'static, anyhow::Result<Arc<dyn InferLangModel>>> {
        let why = anyhow::anyhow!("the models of '{org}' are made with no caller's key");
        Box::pin(std::future::ready(Err(why)))
    }
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

    /// Each org whose models are made here, with the set they are made through, in the order
    /// each was first set.
    pub fn list(&self) -> impl Iterator<Item = (&str, &Arc<dyn LangModelSet>)> {
        self.models.iter().map(|(org, set)| (org.as_str(), set))
    }

    /// Stop making the models of `org`. Answers whether there was a set for it.
    pub fn remove(&mut self, org: &str) -> bool {
        self.models.shift_remove(org).is_some()
    }

    /// Make the model a desc of the form `<org>` or `<org>/<name>` names, through the set for
    /// its org. The answer is a future of its own, so a lock held on this provider need not be
    /// held while it is made.
    ///
    /// With `key`, a caller's, the model is called with it in place of the key the set is set up
    /// with: refused by a set that [takes](LangModelSet::takes_key) none.
    pub fn create(
        &self,
        desc: &str,
        key: Option<&str>,
    ) -> anyhow::Result<BoxFuture<'static, anyhow::Result<Arc<dyn InferLangModel>>>> {
        let (org, name) = match desc.split_once('/') {
            Some((org, name)) => (org, Some(name)),
            None => (desc, None),
        };
        let set = self
            .models
            .get(org)
            .ok_or_else(|| anyhow::anyhow!("no provider set for lang model org '{org}'"))?;
        Ok(match key {
            Some(key) => set.make_with_key(org, name, key),
            None => set.make(org, name),
        })
    }
}
