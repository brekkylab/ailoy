use ailoy_macros::multi_platform_async_trait;

use crate::{
    knowledge::{KnowledgeConfig, base::KnowledgeBehavior},
    model::{EmbeddingModel, EmbeddingModelInference},
    value::Document,
    vector_store::{VectorStore, VectorStoreRetrieveResult},
};

#[derive(Debug, Clone)]
pub struct VectorStoreKnowledge {
    store: VectorStore,
    embedding_model: EmbeddingModel,
}

impl From<VectorStoreRetrieveResult> for Document {
    fn from(value: VectorStoreRetrieveResult) -> Self {
        let title = if let Some(metadata) = &value.metadata
            && let Some(raw_title) = metadata.get("title")
        {
            Some(match raw_title {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
        } else {
            None
        };
        Self {
            id: value.id,
            title,
            text: value.document,
        }
    }
}

impl VectorStoreKnowledge {
    pub fn new(store: VectorStore, embedding_model: EmbeddingModel) -> Self {
        Self {
            store,
            embedding_model: embedding_model,
        }
    }
}

#[multi_platform_async_trait]
impl KnowledgeBehavior for VectorStoreKnowledge {
    async fn retrieve(
        &self,
        query: String,
        config: KnowledgeConfig,
    ) -> anyhow::Result<Vec<Document>> {
        let query_embedding = self.embedding_model.infer(query.into()).await?;
        let results = self
            .store
            .retrieve(query_embedding, config.top_k.unwrap_or_default() as usize)
            .await?
            .into_iter()
            .map(|res| res.into())
            .collect::<Vec<_>>();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ailoy_macros::multi_platform_test;
    use futures::{lock::Mutex, stream::StreamExt};

    use super::*;
    use crate::{
        agent::Agent,
        knowledge::{Knowledge, KnowledgeTool},
        model::{EmbeddingModel, LangModel},
        tool::{Tool, ToolBehavior as _},
        value::Part,
        vector_store::{FaissStore, VectorStoreAddInput},
    };

    async fn prepare_knowledge() -> anyhow::Result<Knowledge> {
        let mut store = VectorStore::new_faiss(FaissStore::new(1024).await.unwrap());
        let embedding_model = EmbeddingModel::new_local("BAAI/bge-m3").await.unwrap();

        let doc0: String = "Ailoy is an awesome AI agent framework.".into();
        store
            .add_vector(VectorStoreAddInput {
                embedding: embedding_model.infer(doc0.clone()).await.unwrap(),
                document: doc0,
                metadata: None,
            })
            .await?;
        let doc1: String = "Langchain is a library".into();
        store
            .add_vector(VectorStoreAddInput {
                embedding: embedding_model.infer(doc1.clone()).await.unwrap(),
                document: doc1,
                metadata: None,
            })
            .await?;

        let knowledge = Knowledge::new_vector_store(store, embedding_model);
        Ok(knowledge)
    }

    #[multi_platform_test]
    async fn test_vectorstore_knowledge_with_agent() -> anyhow::Result<()> {
        let knowledge = prepare_knowledge().await?;
        let model = LangModel::try_new_local("Qwen/Qwen3-0.6B").await.unwrap();
        let agent = Arc::new(Mutex::new(Agent::new(model, vec![])));

        // Testing as knowledge
        {
            let mut agent_guard = agent.lock().await;
            agent_guard.set_knowledge(knowledge.clone());

            let mut strm = Box::pin(agent_guard.run(vec![Part::text("What is Ailoy?")]));
            while let Some(out_opt) = strm.next().await {
                let out = out_opt.unwrap();
                if out.aggregated.is_some() {
                    println!("{:?}", out.aggregated.unwrap());
                }
            }
        }
        // Remove knowledge
        {
            let mut agent_guard = agent.lock().await;
            agent_guard.remove_knowledge();
            agent_guard.clear_messages().await?;
        }

        // Testing as tool
        {
            let mut agent_guard = agent.lock().await;
            let tool = KnowledgeTool::from(knowledge);
            agent_guard
                .add_tool(Tool::new_knowledge(tool.clone()))
                .await?;

            let mut strm = Box::pin(agent_guard.run(vec![Part::text(format!(
                "What is Ailoy? Answer by calling tool '{}'",
                tool.get_description().name
            ))]));
            while let Some(output) = strm.next().await {
                let output = output.unwrap();
                if output.aggregated.is_some() {
                    println!("{:?}", output.aggregated.unwrap());
                }
            }
        }

        Ok(())
    }
}
