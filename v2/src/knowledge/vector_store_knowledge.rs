use std::sync::Arc;

use ailoy_macros::multi_platform_async_trait;
use futures::lock::Mutex;

use crate::{
    knowledge::base::{KnowledgeBehavior, KnowledgeRetrieveResult},
    model::{EmbeddingModel, EmbeddingModelInference},
    vector_store::{VectorStore, VectorStoreRetrieveResult},
};

#[derive(Debug, Clone)]
pub struct VectorStoreKnowledge {
    store: Arc<Mutex<dyn VectorStore>>,
    embedding_model: EmbeddingModel,
}

impl From<VectorStoreRetrieveResult> for KnowledgeRetrieveResult {
    fn from(value: VectorStoreRetrieveResult) -> Self {
        Self {
            document: value.document,
            metadata: value.metadata,
        }
    }
}

impl VectorStoreKnowledge {
    pub fn new(store: impl VectorStore + 'static, embedding_model: EmbeddingModel) -> Self {
        Self {
            store: Arc::new(Mutex::new(store)),
            embedding_model: embedding_model,
        }
    }
}

#[multi_platform_async_trait]
impl KnowledgeBehavior for VectorStoreKnowledge {
    async fn retrieve(
        &self,
        query: String,
        top_k: u32,
    ) -> anyhow::Result<Vec<KnowledgeRetrieveResult>> {
        let query_embedding = self.embedding_model.infer(query.into()).await?;
        let results = {
            let store = self.store.lock().await;
            store.retrieve(query_embedding, top_k as usize).await
        }?
        .into_iter()
        .map(|res| res.into())
        .collect::<Vec<KnowledgeRetrieveResult>>();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use ailoy_macros::multi_platform_test;
    use futures::stream::StreamExt;
    use serde_json::json;

    use super::*;
    use crate::{
        agent::{Agent, SystemMessageRenderer},
        knowledge::{Knowledge, KnowledgeTool},
        model::{EmbeddingModel, LangModel},
        tool::{Tool, ToolBehavior as _},
        value::{Part, Value},
        vector_store::{FaissStore, VectorStoreAddInput},
    };

    async fn prepare_knowledge() -> anyhow::Result<Knowledge> {
        let mut store = FaissStore::new(1024).await.unwrap();
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
    async fn test_vectorstore_knowledge() -> anyhow::Result<()> {
        let knowledge = prepare_knowledge().await?;

        // Testing with renderer
        let retrieved = knowledge.retrieve("What is Ailoy?".into(), 1).await?;
        let renderer = SystemMessageRenderer::new();
        let rendered = renderer
            .render("This is a system message.".into(), Some(retrieved))
            .unwrap();
        println!("Rendered results: {:?}", rendered);

        // Testing with tool call
        let tool = KnowledgeTool::from(knowledge);
        let args = serde_json::from_value::<Value>(json!({
            "query": "What is Langchain?"
        }))?;
        let tool_result = tool.run(args).await?;
        println!("Tool call results: {:?}", tool_result);

        Ok(())
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
