use std::{fmt::Debug, sync::Arc};

use ailoy_macros::multi_platform_async_trait;
use futures::future::BoxFuture;

use crate::{
    knowledge::{KnowledgeBehavior, KnowledgeConfig},
    utils::{MaybeSend, MaybeSync},
    value::Document,
};

#[derive(Clone)]
pub struct CustomKnowledge {
    f: Arc<
        dyn Fn(String, KnowledgeConfig) -> BoxFuture<'static, anyhow::Result<Vec<Document>>>
            + MaybeSend
            + MaybeSync,
    >,
}

impl Debug for CustomKnowledge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomKnowledge")
            .field("f", &"function")
            .finish()
    }
}

impl CustomKnowledge {
    pub fn new(
        f: Arc<
            dyn Fn(String, KnowledgeConfig) -> BoxFuture<'static, anyhow::Result<Vec<Document>>>
                + MaybeSend
                + MaybeSync,
        >,
    ) -> Self {
        Self { f }
    }
}

#[multi_platform_async_trait]
impl KnowledgeBehavior for CustomKnowledge {
    async fn retrieve(
        &self,
        query: String,
        config: KnowledgeConfig,
    ) -> anyhow::Result<Vec<Document>> {
        (self.f)(query, config).await
    }
}

#[cfg(test)]
mod tests {
    use ailoy_macros::multi_platform_test;
    use futures::{FutureExt, stream::StreamExt};

    use super::*;
    use crate::{agent::Agent, knowledge::Knowledge, model::LangModel, value::Part};

    #[multi_platform_test]
    async fn test_custom_knowledge_with_agent() -> anyhow::Result<()> {
        let knowledge = Knowledge::new_custom(CustomKnowledge::new(Arc::new(|_, _| {
            async {
                let documents = vec![
                    Document {
                        id: String::from("1"),
                        title: None,
                        text: "Ailoy is an awesome AI agent framework.".into(),
                    },
                    Document {
                        id: String::from("2"),
                        title: None,
                        text: "Ailoy supports Python, Javascript and Rust.".into(),
                    },
                    Document {
                        id: String::from("3"),
                        title: None,
                        text: "Ailoy enables running LLMs in local environment easily.".into(),
                    },
                ];
                Ok(documents)
            }
            .boxed()
        })));
        let model = LangModel::try_new_local("Qwen/Qwen3-0.6B").await.unwrap();
        let mut agent = Agent::new(model, vec![]);

        agent.set_knowledge(knowledge);

        let mut strm = Box::pin(agent.run(vec![Part::Text {
            text: "What is Ailoy?".into(),
        }]));
        while let Some(out) = strm.next().await {
            let out = out.unwrap();
            println!("{:?}", out);
        }

        Ok(())
    }
}
