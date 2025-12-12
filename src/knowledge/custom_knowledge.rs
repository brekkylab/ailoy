use std::{fmt::Debug, sync::Arc};

use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};

use crate::{
    knowledge::{KnowledgeBehavior, KnowledgeConfig},
    utils::BoxFuture,
    value::Document,
};

#[maybe_send_sync]
pub(super) type CustomKnowledgeRetrieveFunc =
    dyn Fn(String, KnowledgeConfig) -> BoxFuture<'static, anyhow::Result<Vec<Document>>>;

#[derive(Clone)]
pub struct CustomKnowledge {
    f: Arc<CustomKnowledgeRetrieveFunc>,
}

impl Debug for CustomKnowledge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomKnowledge")
            .field("f", &"function")
            .finish()
    }
}

impl CustomKnowledge {
    pub fn new(f: Arc<CustomKnowledgeRetrieveFunc>) -> Self {
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
    use crate::{
        agent::Agent,
        boxed,
        knowledge::Knowledge,
        model::LangModel,
        value::{Message, Part, Role},
    };

    #[multi_platform_test]
    async fn test_custom_knowledge_with_agent() -> anyhow::Result<()> {
        let knowledge = Knowledge::new_custom(CustomKnowledge::new(Arc::new(|_, _| {
            boxed!(async {
                let documents = vec![
                    Document {
                        id: "1".to_owned(),
                        title: None,
                        text: "Ailoy is an awesome AI agent framework.".to_owned(),
                    },
                    Document {
                        id: "2".to_owned(),
                        title: None,
                        text: "Ailoy supports Python, Javascript and Rust.".to_owned(),
                    },
                    Document {
                        id: "3".to_owned(),
                        title: None,
                        text: "Ailoy enables running LLMs in local environment easily.".to_owned(),
                    },
                ];
                Ok(documents)
            })
        })));
        let model = LangModel::try_new_local("Qwen/Qwen3-0.6B", None)
            .await
            .unwrap();
        let mut agent = Agent::new(model, vec![], None);

        agent.set_knowledge(knowledge);

        let mut strm = Box::pin(agent.run_delta(
            vec![Message::new(Role::User).with_contents(vec![Part::Text {
                text: "What is Ailoy?".into(),
            }])],
            None,
        ));
        while let Some(out) = strm.next().await {
            let out = out.unwrap();
            println!("{:?}", out);
        }

        Ok(())
    }
}
