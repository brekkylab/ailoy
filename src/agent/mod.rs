use serde::{Deserialize, Serialize};

use crate::{LangModel, Tool};

#[cfg(feature = "rt")]
pub use rt::AgentRuntime;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentDesc {
    pub name: String,
    pub desc: String,
    pub skills: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Agent {
    pub lm: LangModel,
    pub tools: Vec<Tool>,
    pub desc: Option<AgentDesc>,
}

impl Agent {
    pub fn new(lm: LangModel) -> Self {
        Self {
            lm,
            tools: vec![],
            desc: None,
        }
    }

    pub fn with_tools(mut self, tools: impl IntoIterator<Item = impl Into<Tool>>) -> Self {
        self.tools = tools.into_iter().map(|v| v.into()).collect();
        self
    }

    pub fn with_desc(mut self, desc: AgentDesc) -> Self {
        self.desc = Some(desc);
        self
    }
}

#[cfg(feature = "rt")]
mod rt {

    use crate::{LangModelProvider, LangModelRuntime, Message, ToolRuntime};

    use super::*;

    pub struct AgentProvider {
        lm: LangModelProvider,
    }

    pub struct AgentRuntime<'a> {
        lm: LangModelRuntime,
        tools: Vec<ToolRuntime<'a>>,
        history: Vec<Message>,
    }

    impl<'a> AgentRuntime<'a> {
        pub fn new(agent: Agent, provider: AgentProvider) -> Self {
            Self {
                lm: LangModelRuntime::new(agent.lm, provider.lm),
                tools: agent
                    .tools
                    .into_iter()
                    .map(|v| ToolRuntime::try_new(v).unwrap())
                    .collect(),
                history: vec![],
            }
        }

        pub async fn run(&mut self, query: Message) -> anyhow::Result<Message> {
            todo!()
        }
    }
}
