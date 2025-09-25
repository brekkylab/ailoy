use std::sync::Arc;

use ailoy_macros::multi_platform_async_trait;
use anyhow::{Result, anyhow};
use dedent::dedent;
use futures::lock::Mutex;
use minijinja::{Environment, context};
use minijinja_contrib::{add_to_environment, pycompat::unknown_method_callback};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json, json};

use crate::model::EmbeddingModel;
use crate::tool::Tool;
use crate::utils::{MaybeSend, MaybeSync};
use crate::value::{Part, ToolCallArg, ToolDesc};

pub type Embedding = Vec<f32>;
pub type Metadata = Map<String, Json>;

#[derive(Debug)]
pub struct AddInput {
    pub embedding: Embedding,
    pub document: String,
    pub metadata: Option<Metadata>,
}

#[derive(Debug)]
pub struct GetResult {
    pub id: String,
    pub document: String,
    pub metadata: Option<Metadata>,
    pub embedding: Embedding,
}

#[derive(Debug)]
pub struct RetrieveResult {
    pub id: String,
    pub document: String,
    pub metadata: Option<Metadata>,
    pub distance: f32,
}

#[multi_platform_async_trait]
pub trait VectorStore: MaybeSend + MaybeSync {
    async fn add_vector(&mut self, input: AddInput) -> Result<String>;
    async fn add_vectors(&mut self, inputs: Vec<AddInput>) -> Result<Vec<String>>;
    async fn get_by_id(&self, id: &str) -> Result<Option<GetResult>>;
    async fn get_by_ids(&self, ids: &[&str]) -> Result<Vec<GetResult>>;
    async fn retrieve(
        &self,
        query_embedding: Embedding,
        top_k: usize,
    ) -> Result<Vec<RetrieveResult>>;
    async fn batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: usize,
    ) -> Result<Vec<Vec<RetrieveResult>>>;
    async fn remove_vector(&mut self, id: &str) -> Result<()>;
    async fn remove_vectors(&mut self, ids: &[&str]) -> Result<()>;
    async fn clear(&mut self) -> Result<()>;

    async fn count(&self) -> Result<usize>;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct KnowledgeRetrieveResult {
    pub document: String,
    pub metadata: Option<Metadata>,
}

impl From<RetrieveResult> for KnowledgeRetrieveResult {
    fn from(value: RetrieveResult) -> Self {
        Self {
            document: value.document,
            metadata: value.metadata,
        }
    }
}

#[multi_platform_async_trait]
pub trait Knowledge: std::fmt::Debug + MaybeSend + MaybeSync {
    fn name(&self) -> String;

    async fn retrieve(&self, query: String) -> Result<Vec<KnowledgeRetrieveResult>>;
}

pub struct SystemMessageRenderer {
    template: String,
    mj_env: Environment<'static>,
}

impl SystemMessageRenderer {
    pub fn new() -> Self {
        let default_template = dedent!(r#"
        {%- if content %}
            {{- content }}
        {%- endif %}
        {%- if knowledge_results %}
            {{- '\n\n# Knowledges\n\nBelow is a list of documents retrieved from knowledge bases. Try to answer user\'s question based on the provided knowledges.\n' }}
            {{- "<documents>\n" }}
            {%- for item in knowledge_results %}
            {{- "<document>\n" }}
                {{- item.document + '\n' }}
            {{- "</document>\n" }}
            {%- endfor %}
            {{- "</documents>\n" }}
        {%- endif %}
        "#).to_string();
        let default_mj_env = Self::_create_mj_env(default_template.clone());

        Self {
            template: default_template,
            mj_env: default_mj_env,
        }
    }

    pub fn with_template(self, template: String) -> Self {
        let mj_env = Self::_create_mj_env(template.clone());
        Self {
            template,
            mj_env,
            ..self
        }
    }

    pub fn template(&self) -> &String {
        &self.template
    }

    pub fn _create_mj_env(template: String) -> Environment<'static> {
        let mut e = Environment::new();
        add_to_environment(&mut e);
        e.set_unknown_method_callback(unknown_method_callback);
        e.add_template_owned("template", template).unwrap();
        e
    }

    pub fn render(
        &self,
        content: String,
        knowledge_results: Option<Vec<KnowledgeRetrieveResult>>,
    ) -> Result<String> {
        let ctx = context!(content => content, knowledge_results => knowledge_results);
        let rendered = self
            .mj_env
            .get_template("template")
            .unwrap()
            .render(ctx)
            .map_err(|e| anyhow!(format!("minijinja::render failed: {}", e.to_string())))?;

        Ok(rendered)
    }
}

#[derive(Clone)]
pub struct KnowledgeTool {
    inner: Arc<dyn Knowledge>,
    desc: ToolDesc,
    stringify: Arc<dyn Fn(Vec<KnowledgeRetrieveResult>) -> Result<String> + MaybeSend + MaybeSync>,
}

impl std::fmt::Debug for KnowledgeTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KnowledgeTool")
            .field("desc", &self.desc)
            .field("inner", &self.inner)
            .field("stringify", &"(Function)")
            .finish()
    }
}

impl KnowledgeTool {
    pub fn from(knowledge: impl Knowledge + 'static) -> Self {
        let default_desc = ToolDesc::new(
            format!("retrieve-{}", knowledge.name()),
            "Retrieve the relevant context from knowledge base.".into(),
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The input query to search for the relevant context."
                    }
                },
                "required": ["query"]
            }),
            None,
        )
        .unwrap();

        let default_stringify = Arc::new(|results: Vec<KnowledgeRetrieveResult>| {
            Ok(serde_json::to_value(results)
                .map_err(|e| anyhow!(e.to_string()))?
                .to_string())
        });

        Self {
            desc: default_desc,
            inner: Arc::new(knowledge),
            stringify: default_stringify,
        }
    }

    pub fn with_description(self, desc: ToolDesc) -> Self {
        Self { desc, ..self }
    }

    pub fn with_stringify(
        self,
        stringify: Arc<
            dyn Fn(Vec<KnowledgeRetrieveResult>) -> Result<String> + MaybeSend + MaybeSync,
        >,
    ) -> Self {
        Self { stringify, ..self }
    }
}

#[multi_platform_async_trait]
impl Tool for KnowledgeTool {
    fn get_description(&self) -> ToolDesc {
        self.desc.clone()
    }

    async fn run(&self, args: ToolCallArg) -> Result<Vec<Part>, String> {
        let args = match args.as_object() {
            Some(a) => a,
            None => {
                return Ok(vec![Part::Text(
                    "Error: Invalid arguments: expected object".into(),
                )]);
            }
        };

        let query = match args.get("query") {
            Some(query) => query.to_string(),
            None => {
                return Ok(vec![Part::Text(
                    "Error: Missing required 'query' string".into(),
                )]);
            }
        };

        let results = match self.inner.retrieve(query).await {
            Ok(results) => results,
            Err(e) => {
                return Ok(vec![Part::Text(e.to_string())]);
            }
        };

        let rendered = match (self.stringify)(results) {
            Ok(text) => text,
            Err(e) => {
                return Ok(vec![Part::Text(e.to_string())]);
            }
        };

        Ok(vec![Part::Text(rendered)])
    }
}

#[derive(Debug, Clone)]
pub struct VectorStoreKnowledge {
    name: String,
    store: Arc<Mutex<dyn VectorStore>>,
    embedding_model: Arc<Mutex<dyn EmbeddingModel>>,
    top_k: usize,
}

impl VectorStoreKnowledge {
    pub fn new(
        name: impl Into<String>,
        store: impl VectorStore + 'static,
        embedding_model: impl EmbeddingModel + 'static,
    ) -> Self {
        let name = name.into();
        let default_top_k = 5 as usize;

        Self {
            name: name,
            store: Arc::new(Mutex::new(store)),
            embedding_model: Arc::new(Mutex::new(embedding_model)),
            top_k: default_top_k,
        }
    }

    pub fn with_top_k(self, top_k: usize) -> Self {
        Self { top_k, ..self }
    }
}

#[multi_platform_async_trait]
impl Knowledge for VectorStoreKnowledge {
    fn name(&self) -> String {
        self.name.clone()
    }

    async fn retrieve(&self, query: String) -> Result<Vec<KnowledgeRetrieveResult>> {
        let query_embedding = {
            let mut model = self.embedding_model.lock().await;
            model.run(query.into()).await
        }?;

        let results = {
            let store = self.store.lock().await;
            store.retrieve(query_embedding, self.top_k).await
        }?
        .into_iter()
        .map(|res| res.into())
        .collect::<Vec<KnowledgeRetrieveResult>>();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use futures::stream::StreamExt;

    use ailoy_macros::multi_platform_test;

    use super::*;
    use crate::agent::Agent;
    use crate::knowledge_base::FaissStore;
    use crate::model::{LocalEmbeddingModel, LocalLanguageModel};
    use crate::value::MessageAggregator;

    async fn prepare_knowledge() -> Result<VectorStoreKnowledge> {
        let mut store = FaissStore::new(1024).await.unwrap();
        let mut embedding_model = LocalEmbeddingModel::new("BAAI/bge-m3").await.unwrap();

        let doc0: String = "Ailoy is an awesome AI agent framework.".into();
        store
            .add_vector(AddInput {
                embedding: embedding_model.run(doc0.clone()).await.unwrap(),
                document: doc0,
                metadata: None,
            })
            .await?;
        let doc1: String = "Langchain is a library".into();
        store
            .add_vector(AddInput {
                embedding: embedding_model.run(doc1.clone()).await.unwrap(),
                document: doc1,
                metadata: None,
            })
            .await?;

        let knowledge = VectorStoreKnowledge::new("my-store", store, embedding_model).with_top_k(1);
        Ok(knowledge)
    }

    #[multi_platform_test]
    async fn test_vectorstore_knowledge() -> Result<()> {
        let knowledge = prepare_knowledge().await?;

        // Testing with renderer
        let retrieved = knowledge.retrieve("What is Ailoy?".into()).await?;
        let renderer = SystemMessageRenderer::new();
        let rendered = renderer
            .render("This is a system message.".into(), Some(retrieved))
            .unwrap();
        println!("Rendered results: {:?}", rendered);

        // Testing with tool call
        let tool = KnowledgeTool::from(knowledge);
        let args = serde_json::from_value::<ToolCallArg>(json!({
            "query": "What is Langchain?"
        }))?;
        let tool_result = tool.run(args).await.map_err(|e| anyhow!(e))?;
        println!("Tool call results: {:?}", tool_result);

        Ok(())
    }

    #[multi_platform_test]
    async fn test_vectorstore_knowledge_with_agent() -> Result<()> {
        let knowledge = prepare_knowledge().await?;
        let renderer = SystemMessageRenderer::new();
        let model = LocalLanguageModel::new("Qwen/Qwen3-0.6B").await.unwrap();
        let agent = Arc::new(Mutex::new(
            Agent::new(model, vec![]).with_system_message_renderer(renderer),
        ));
        let mut agg = MessageAggregator::new();

        // Testing as knowledge
        {
            let mut agent_guard = agent.lock().await;
            agent_guard.add_knowledge(knowledge.clone()).await?;

            let mut strm = Box::pin(agent_guard.run(vec![Part::Text("What is Ailoy?".into())]));
            while let Some(delta_opt) = strm.next().await {
                let delta = delta_opt.unwrap();
                if let Some(msg) = agg.update(delta) {
                    println!("{:?}", msg);
                }
            }
        }
        // Remove knowledge
        {
            let mut agent_guard = agent.lock().await;
            agent_guard.remove_knowledge(knowledge.name()).await?;
            agent_guard.clear_messages().await?;
        }

        // Testing as tool
        {
            let mut agent_guard = agent.lock().await;
            // Example of customizing with_stringify
            let tool = KnowledgeTool::from(knowledge).with_stringify(Arc::new(|results| {
                Ok(serde_json::to_value(
                    results
                        .iter()
                        .map(|res| res.document.clone())
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| anyhow!(e.to_string()))?
                .to_string())
            }));
            agent_guard.add_tool(Arc::new(tool.clone())).await?;

            let mut strm = Box::pin(agent_guard.run(vec![Part::Text(format!(
                "What is Ailoy? Answer by calling tool '{}'",
                tool.get_description().name
            ))]));
            while let Some(delta_opt) = strm.next().await {
                let delta = delta_opt.unwrap();
                if let Some(msg) = agg.update(delta) {
                    println!("{:?}", msg);
                }
            }
        }

        Ok(())
    }
}
