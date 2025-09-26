use std::sync::Arc;

use ailoy_macros::multi_platform_async_trait;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
    tool::Tool,
    utils::{MaybeSend, MaybeSync},
    value::{Part, ToolCallArg, ToolDesc},
};

type Metadata = Map<String, Value>;

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

#[cfg(test)]
mod tests {
    use ailoy_macros::multi_platform_test;
    use futures::stream::StreamExt;

    use super::*;
    use crate::{
        agent::Agent,
        model::LocalLanguageModel,
        value::{MessageAggregator, Part},
    };

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
        let model = LocalLanguageModel::new("Qwen/Qwen3-0.6B").await.unwrap();
        let mut agent = Agent::new(model, vec![]);
        let mut agg = MessageAggregator::new();

        agent.add_knowledge(knowledge).await?;

        let mut strm = Box::pin(agent.run(vec![Part::Text("What is Ailoy?".into())]));
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                println!("{:?}", msg);
            }
        }

        Ok(())
    }
}
