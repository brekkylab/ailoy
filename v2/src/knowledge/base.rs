use std::sync::Arc;

use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};
use serde::{Deserialize, Serialize};

use crate::{
    knowledge::{CustomKnowledge, VectorStoreKnowledge},
    model::EmbeddingModel,
    to_value,
    tool::ToolBehavior,
    value::{ToolDesc, Value},
    vector_store::VectorStore,
};

type Metadata = serde_json::Map<String, serde_json::Value>;

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(from_wasm_abi, into_wasm_abi))]
pub struct KnowledgeRetrieveResult {
    pub document: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

#[maybe_send_sync]
#[multi_platform_async_trait]
pub trait KnowledgeBehavior: std::fmt::Debug {
    async fn retrieve(
        &self,
        query: String,
        top_k: u32,
    ) -> anyhow::Result<Vec<KnowledgeRetrieveResult>>;
}

#[derive(Clone)]
pub struct KnowledgeTool {
    inner: Arc<dyn KnowledgeBehavior>,
    desc: ToolDesc,
}

impl std::fmt::Debug for KnowledgeTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KnowledgeTool")
            .field("desc", &self.desc)
            .field("inner", &self.inner)
            .finish()
    }
}

impl KnowledgeTool {
    pub fn from(knowledge: impl KnowledgeBehavior + 'static) -> Self {
        let default_desc = ToolDesc {
            name: "retrieve-from-knowledge".into(),
            description: Some("Retrieve the relevant context from knowledge base.".into()),
            parameters: to_value!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The input query to search for the relevant context."
                    }
                },
                "required": ["query"]
            }),
            returns: None,
        };

        Self {
            desc: default_desc,
            inner: Arc::new(knowledge),
        }
    }

    pub fn with_description(self, desc: ToolDesc) -> Self {
        Self { desc, ..self }
    }
}

#[multi_platform_async_trait]
impl ToolBehavior for KnowledgeTool {
    fn get_description(&self) -> ToolDesc {
        self.desc.clone()
    }

    async fn run(&self, args: Value) -> anyhow::Result<Value> {
        let args = match args.as_object() {
            Some(a) => a,
            None => {
                return Ok("Error: Invalid arguments: expected object".into());
            }
        };

        let query = match args.get("query") {
            Some(query) => match query.as_str() {
                Some(v) => v,
                None => return Ok("Error: Field `query` is not string".into()),
            },
            None => {
                return Ok("Error: Missing required 'query' string".into());
            }
        };

        let results = match self.inner.retrieve(query.into(), 1).await {
            Ok(results) => results,
            Err(e) => {
                return Ok(e.to_string().into());
            }
        };

        let val = serde_json::to_value(results).unwrap();
        Ok(val.into())
    }
}

#[derive(Debug, Clone)]
pub enum KnowledgeInner {
    VectorStore(VectorStoreKnowledge),
    Custom(CustomKnowledge),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct Knowledge {
    inner: KnowledgeInner,
}

impl Knowledge {
    pub fn new_vector_store(store: VectorStore, embedding_model: EmbeddingModel) -> Self {
        Self {
            inner: KnowledgeInner::VectorStore(VectorStoreKnowledge::new(store, embedding_model)),
        }
    }

    pub fn new_custom(knowledge: CustomKnowledge) -> Self {
        Self {
            inner: KnowledgeInner::Custom(knowledge),
        }
    }
}

#[multi_platform_async_trait]
impl KnowledgeBehavior for Knowledge {
    async fn retrieve(
        &self,
        query: String,
        top_k: u32,
    ) -> anyhow::Result<Vec<KnowledgeRetrieveResult>> {
        match &self.inner {
            KnowledgeInner::VectorStore(knowledge) => knowledge.retrieve(query, top_k).await,
            KnowledgeInner::Custom(knowledge) => knowledge.retrieve(query, top_k).await,
        }
    }
}

#[cfg(feature = "wasm")]
mod wasm {
    use wasm_bindgen::prelude::*;

    use super::*;
    use crate::tool::Tool;

    #[wasm_bindgen]
    impl Knowledge {
        #[wasm_bindgen(js_name = "newVectorStore")]
        pub fn new_vector_store_js(store: &VectorStore, embedding_model: &EmbeddingModel) -> Self {
            Self::new_vector_store(store.clone(), embedding_model.clone())
        }

        #[wasm_bindgen(js_name = "retrieve")]
        pub async fn retrieve_js(
            &self,
            query: String,
            top_k: u32,
        ) -> Result<Vec<KnowledgeRetrieveResult>, js_sys::Error> {
            self.retrieve(query, top_k)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = "asTool")]
        pub fn as_tool(self) -> Tool {
            let tool = KnowledgeTool::from(self);
            Tool::new_knowledge(tool)
        }
    }
}
