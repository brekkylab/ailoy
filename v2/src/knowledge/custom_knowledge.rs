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

#[cfg(test)]
mod tests {
    use ailoy_macros::multi_platform_test;
    use futures::{FutureExt, stream::StreamExt};

    use super::*;
    use crate::{agent::Agent, knowledge::Knowledge, model::LangModel, value::Part};

    #[multi_platform_test]
    async fn test_custom_knowledge_with_agent() -> anyhow::Result<()> {
        let knowledge = Knowledge::new_custom(CustomKnowledge::new(
            "about-ailoy",
            Arc::new(|_| {
                async {
                    let documents = vec![
                        KnowledgeRetrieveResult {
                            document: "Ailoy is an awesome AI agent framework.".into(),
                            metadata: None,
                        },
                        KnowledgeRetrieveResult {
                            document: "Ailoy supports Python, Javascript and Rust.".into(),
                            metadata: None,
                        },
                        KnowledgeRetrieveResult {
                            document: "Ailoy enables running LLMs in local environment easily."
                                .into(),
                            metadata: None,
                        },
                    ];
                    Ok(documents)
                }
                .boxed()
            }),
        ));
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
