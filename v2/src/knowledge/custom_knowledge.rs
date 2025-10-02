use std::{fmt::Debug, sync::Arc};

use ailoy_macros::multi_platform_async_trait;
use futures::future::BoxFuture;

use crate::{
    knowledge::{KnowledgeBehavior, KnowledgeRetrieveResult},
    utils::{MaybeSend, MaybeSync},
};

#[derive(Clone)]
pub struct CustomKnowledge {
    name: String,
    f: Arc<
        dyn Fn(String) -> BoxFuture<'static, anyhow::Result<Vec<KnowledgeRetrieveResult>>>
            + MaybeSend
            + MaybeSync,
    >,
}

impl Debug for CustomKnowledge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomKnowledge")
            .field("name", &self.name)
            .field("f", &"function")
            .finish()
    }
}

impl CustomKnowledge {
    pub fn new(
        name: impl Into<String>,
        f: Arc<
            dyn Fn(String) -> BoxFuture<'static, anyhow::Result<Vec<KnowledgeRetrieveResult>>>
                + MaybeSend
                + MaybeSync,
        >,
    ) -> Self {
        Self {
            name: name.into(),
            f,
        }
    }
}

#[multi_platform_async_trait]
impl KnowledgeBehavior for CustomKnowledge {
    fn name(&self) -> String {
        "about-ailoy".into()
    }

    async fn retrieve(&self, query: String) -> anyhow::Result<Vec<KnowledgeRetrieveResult>> {
        (self.f)(query).await
    }
}
