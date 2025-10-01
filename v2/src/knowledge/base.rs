use std::sync::Arc;

use ailoy_macros::multi_platform_async_trait;
use anyhow::Result;
use serde::{Deserialize, Serialize};

// use serde_json::{Map, Value, json};
use crate::{
    to_value,
    tool::Tool,
    utils::{MaybeSend, MaybeSync},
    value::{ToolDesc, Value},
};

type Metadata = serde_json::Map<String, serde_json::Value>;

#[derive(Debug, Serialize, Deserialize)]
pub struct KnowledgeRetrieveResult {
    pub document: String,
    pub metadata: Option<Metadata>,
}

#[multi_platform_async_trait]
pub trait Knowledge: std::fmt::Debug + MaybeSend + MaybeSync {
    fn name(&self) -> String;

    async fn retrieve(&self, query: String) -> Result<Vec<KnowledgeRetrieveResult>>;
}

#[derive(Clone)]
pub struct KnowledgeTool {
    inner: Arc<dyn Knowledge>,
    desc: ToolDesc,
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
        let default_desc = ToolDesc {
            name: format!("retrieve-{}", knowledge.name()),
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
impl Tool for KnowledgeTool {
    fn get_description(&self) -> ToolDesc {
        self.desc.clone()
    }

    async fn run(&self, args: Value) -> Result<Value, String> {
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

        let results = match self.inner.retrieve(query.into()).await {
            Ok(results) => results,
            Err(e) => {
                return Ok(e.to_string().into());
            }
        };

        let val = serde_json::to_value(results).unwrap();
        Ok(val.into())
    }
}

#[cfg(test)]
mod tests {
    use ailoy_macros::multi_platform_test;
    use futures::stream::StreamExt;

    use super::*;
    use crate::{agent::Agent, model::LangModel, value::Part};

    #[derive(Debug)]
    struct CustomKnowledge {}

    #[multi_platform_async_trait]
    impl Knowledge for CustomKnowledge {
        fn name(&self) -> String {
            "about-ailoy".into()
        }

        async fn retrieve(&self, _query: String) -> Result<Vec<KnowledgeRetrieveResult>> {
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
                    document: "Ailoy enables running LLMs in local environment easily.".into(),
                    metadata: None,
                },
            ];
            Ok(documents)
        }
    }

    #[multi_platform_test]
    async fn test_custom_knowledge_with_agent() -> Result<()> {
        let knowledge = CustomKnowledge {};
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
