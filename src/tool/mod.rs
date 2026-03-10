mod knowledge;

use serde::{Deserialize, Serialize};

use crate::ToolDesc;

#[cfg(feature = "rt")]
pub use rt::{ToolFunc, ToolFuture, ToolRuntime};

/// Describes a language model
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Tool {
    Builtin { name: String },
    MCP { url: String },
}

#[cfg(feature = "rt")]
mod rt {
    use std::sync::Arc;

    use futures::future::BoxFuture;

    use super::*;
    use crate::{Message, Part, Role, datatype::Value};

    pub type ToolFuture<'a> = BoxFuture<'a, Value>;
    pub type ToolFunc<'a> = dyn Fn(Value) -> ToolFuture<'a>;

    #[derive(Clone)]
    pub struct ToolRuntime<'a> {
        desc: ToolDesc,
        f: Arc<ToolFunc<'a>>,
    }

    impl<'a> ToolRuntime<'a> {
        pub fn try_new(tool: Tool) -> anyhow::Result<Self> {
            match tool {
                Tool::Builtin { name } => {
                    if &name == "knowledge.search" {
                        let (desc, f) = knowledge::create_knowledge_search_tool();
                        Ok(Self { desc, f })
                    } else {
                        Err(anyhow::anyhow!("Unknown builtin tool"))
                    }
                }
                Tool::MCP { url: _url } => todo!(),
            }
        }

        pub fn can_run(&self, tool_call: &Part) -> bool {
            let (_, name, _) = tool_call.as_function().unwrap();
            name == self.desc.name
        }

        pub async fn run(&self, tool_call: Part) -> Message {
            let (_, _, args) = tool_call.as_function().unwrap();
            let result = (self.f)(args.clone()).await;
            Message::new(Role::Tool).with_contents([Part::Value { value: result }])
        }
    }
}
