use std::sync::Arc;

use anyhow::Context;
use futures::{StreamExt, lock::Mutex};

use crate::{
    knowledge::{Knowledge, KnowledgeBehavior as _, KnowledgeConfig},
    model::{InferenceConfig, LangModel, LangModelInference as _},
    tool::{Tool, ToolBehavior as _},
    utils::{BoxStream, log},
    value::{Delta, FinishReason, Message, MessageDelta, Part, PartDelta, Role},
};

#[derive(Clone)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub struct Agent {
    lm: LangModel,
    tools: Vec<Tool>,
    messages: Arc<Mutex<Vec<Message>>>,
    knowledge: Option<Knowledge>,
}

/// The yielded value from agent.run().
#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all))]
pub struct AgentResponse {
    /// The message delta per iteration.
    pub delta: MessageDelta,
    /// Optional finish reason. If this is Some, the message aggregation is finalized and stored in `aggregated`.
    pub finish_reason: Option<FinishReason>,
    /// Optional aggregated message.
    pub aggregated: Option<Message>,
}

impl Agent {
    pub fn new(lm: LangModel, tools: impl IntoIterator<Item = Tool>) -> Self {
        Self {
            lm,
            tools: tools.into_iter().collect(),
            messages: Arc::new(Mutex::new(Vec::new())),
            knowledge: None,
        }
    }

    pub fn with_knowledge(mut self, knowledge: Knowledge) -> Self {
        self.knowledge = Some(knowledge);
        self
    }

    pub fn get_lm(&self) -> LangModel {
        self.lm.clone()
    }

    pub fn get_tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }

    pub fn knowledge(&self) -> Option<Knowledge> {
        self.knowledge.clone()
    }

    pub async fn clear_messages(&mut self) -> anyhow::Result<()> {
        self.messages.lock().await.clear();
        Ok(())
    }

    pub async fn add_tools(&mut self, tools: Vec<Tool>) -> anyhow::Result<()> {
        for tool in tools.iter() {
            let tool_name = tool.get_description().name;

            // If the tool with same name already exists, skip adding the tool.
            if self
                .tools
                .iter()
                .find(|t| t.get_description().name == tool_name)
                .is_some()
            {
                log::warn(format!(
                    "Tool \"{}\" is already registered. Skip adding the tool.",
                    tool_name
                ));
                continue;
            }

            self.tools.push(tool.clone());
        }

        Ok(())
    }

    pub async fn add_tool(&mut self, tool: Tool) -> anyhow::Result<()> {
        self.add_tools(vec![tool]).await
    }

    pub async fn remove_tools(&mut self, tool_names: Vec<String>) -> anyhow::Result<()> {
        self.tools.retain(|t| {
            let tool_name = t.get_description().name;
            // Remove the tool if its name belongs to `tool_names`
            !tool_names.contains(&tool_name)
        });
        Ok(())
    }

    pub async fn remove_tool(&mut self, tool_name: String) -> anyhow::Result<()> {
        self.remove_tools(vec![tool_name]).await
    }

    pub fn set_knowledge(&mut self, knowledge: Knowledge) {
        self.knowledge = Some(knowledge);
    }

    pub fn remove_knowledge(&mut self) {
        self.knowledge = None;
    }

    pub fn run<'a>(
        &'a mut self,
        // messages: Vec<Message>,
        contents: Vec<Part>,
    ) -> BoxStream<'a, anyhow::Result<AgentResponse>> {
        // Messages used for the current run round
        let mut messages: Vec<Message> = vec![];
        // For storing message history except system message
        let messages_history = self.messages.clone();

        let tools = self.tools.clone();
        let strm = async_stream::try_stream! {
            // Get documents
            let docs = if let Some(knowledge) = self.knowledge.clone() {
                let query_str = contents.iter().filter(|p| p.is_text()).map(|p| p.as_text().unwrap().to_owned()).collect::<Vec<_>>().join("\n\n");
                knowledge.retrieve(query_str, KnowledgeConfig::default()).await?
            } else {
                vec![]
            };

            // Add message histories to messages
            for msg in messages_history.lock().await.iter() {
                messages.push(msg.clone());
            }

            let user_message = Message::new(Role::User).with_contents(contents);
            // Add user message to messages
            messages.push(user_message.clone());
            // Add user message to histories
            messages_history.lock().await.push(user_message);

            let tool_descs = self
                .tools
                .iter()
                .map(|v| v.get_description())
                .collect::<Vec<_>>();
            loop {
                let mut assistant_msg_delta = MessageDelta::new().with_role(Role::Assistant);
                let mut assistant_msg: Option<Message> = None;
                {
                    let mut model = self.lm.clone();
                    let mut strm = model.infer(messages.clone(), tool_descs.clone(), docs.clone(), InferenceConfig::default());
                    while let Some(out) = strm.next().await {
                        let out = out?;
                        assistant_msg_delta = assistant_msg_delta.aggregate(out.clone().delta).context("Aggregation failed")?;

                        // Message aggregation is finalized if finish_reason does exist
                        if out.finish_reason.is_some() {
                            assistant_msg = Some(assistant_msg_delta.clone().finish()?);
                        }
                        yield AgentResponse {
                            delta: out.delta.clone(),
                            finish_reason: out.finish_reason.clone(),
                            aggregated: assistant_msg.clone(),
                        };
                    }
                }
                let assistant_msg = assistant_msg.unwrap();
                // Add assistant message to messages
                messages.push(assistant_msg.clone());
                // Add assistant message to histories
                messages_history.lock().await.push(assistant_msg.clone());

                // Handling tool calls if exist
                if let Some(tool_calls) = assistant_msg.tool_calls && !tool_calls.is_empty() {
                    for part in &tool_calls {
                        let Some((id, name, args)) = part.as_function() else { continue; };
                        let tool = tools.iter().find(|v| v.get_description().name == name).unwrap().clone();
                        let resp = tool.run(args.clone()).await?;
                        let mut tool_msg_delta = MessageDelta::new().with_role(Role::Tool).with_contents([PartDelta::Value{value: resp.clone()}]);
                        let mut tool_msg = Message::new(Role::Tool).with_contents([Part::Value{value: resp.clone()}]);
                        if let Some(id) = id {
                            tool_msg_delta = tool_msg_delta.with_id(id);
                            tool_msg = tool_msg.with_id(id);
                        }
                        yield AgentResponse {
                            delta: tool_msg_delta,
                            finish_reason: Some(FinishReason::Stop{}),
                            aggregated: Some(tool_msg.clone()),
                        };

                        // Add tool message to messages
                        messages.push(tool_msg.clone());
                        // Add tool message to histories
                        messages_history.lock().await.push(tool_msg);
                    }
                } else {
                    break;
                }
            }
        };
        Box::pin(strm)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn run_simple_chat() -> anyhow::Result<()> {
        use futures::StreamExt;

        use super::*;
        use crate::model::LangModel;

        let model = LangModel::try_new_local("Qwen/Qwen3-0.6B").await.unwrap();
        let mut agent = Agent::new(model, Vec::new());

        let mut strm = Box::pin(agent.run(vec![Part::text("Hi what's your name?")]));
        while let Some(output) = strm.next().await {
            let output = output.unwrap();
            println!("delta: {:?}", output.delta);
            if output.aggregated.is_some() {
                println!("message: {:?}", output.aggregated.unwrap());
            }
        }

        Ok(())
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn run_tool_call() {
        use futures::StreamExt;

        use super::*;
        use crate::{model::LangModel, to_value, value::ToolDesc};

        let model = LangModel::try_new_local("Qwen/Qwen3-0.6B").await.unwrap();

        let tool_desc = ToolDesc::new(
            "temperature".to_owned(),
            Some("Get current temperature".to_owned()),
            to_value!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["Celsius", "Fahrenheit"]
                    }
                },
                "required": ["location", "unit"]
            }),
            Some(to_value!({
                "type": "number",
                "description": "Null if the given city name is unavailable.",
                "nullable": true,
            })),
        );
        let tools = vec![Tool::new_function(
            tool_desc,
            Arc::new(|args| {
                if args
                    .as_object()
                    .unwrap()
                    .get("unit")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    == "Celsius"
                {
                    to_value!("40")
                } else {
                    to_value!("104")
                }
            }),
        )];

        let mut agent = Agent::new(model, tools);

        let mut strm = Box::pin(agent.run(vec![Part::text("How much hot currently in Dubai?")]));
        while let Some(output) = strm.next().await {
            let output = output.unwrap();
            println!("delta: {:?}", output.delta);
            if output.aggregated.is_some() {
                println!("message: {:?}", output.aggregated.unwrap());
            }
        }
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn run_mcp_stdio_tool_call() {
        use futures::StreamExt;
        use rmcp::transport::ConfigureCommandExt;

        use super::*;
        use crate::{model::LangModel, tool::MCPClient};

        let model = LangModel::try_new_local("Qwen/Qwen3-0.6B").await.unwrap();
        let mut agent = Agent::new(model, Vec::new());

        let command = tokio::process::Command::new("uvx").configure(|cmd| {
            cmd.arg("mcp-server-time");
        });
        let mcp_client = MCPClient::from_stdio(command).await.unwrap();
        let mcp_tools = mcp_client
            .tools
            .into_iter()
            .map(|tool| Tool::new_mcp(tool))
            .collect();
        agent.add_tools(mcp_tools).await.unwrap();

        let agent_tools = agent.get_tools();
        assert_eq!(agent_tools.len(), 2);
        assert_eq!(agent_tools[0].get_description().name, "get_current_time");
        assert_eq!(agent_tools[1].get_description().name, "convert_time");

        let mut strm = Box::pin(agent.run(vec![Part::text(
            "What time is it now in America/New_York timezone?",
        )]));
        while let Some(output) = strm.next().await {
            let output = output.unwrap();
            println!("delta: {:?}", output.delta);
            if output.aggregated.is_some() {
                println!("message: {:?}", output.aggregated.unwrap());
            }
        }
    }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::{
        Bound, Py, PyAny, PyRef, PyResult, Python,
        exceptions::{PyStopAsyncIteration, PyStopIteration},
        pyclass, pymethods,
        types::PyType,
    };
    use pyo3_stub_gen_derive::{gen_stub_pyclass, gen_stub_pymethods};

    use super::*;

    fn spawn<'a>(
        mut agent: Agent,
        contents: Vec<Part>,
        // config: InferenceConfig,
    ) -> anyhow::Result<(
        &'a tokio::runtime::Runtime,
        async_channel::Receiver<anyhow::Result<AgentResponse>>,
    )> {
        let (tx, rx) = async_channel::unbounded::<anyhow::Result<AgentResponse>>();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.spawn(async move {
            let mut stream = agent.run(contents).boxed();

            while let Some(item) = stream.next().await {
                if tx.send(item).await.is_err() {
                    break; // Exit if consumer vanished
                }
                // Add a yield point to allow other tasks to run
                tokio::task::yield_now().await;
            }
        });
        Ok((rt, rx))
    }

    #[gen_stub_pyclass]
    #[pyclass(unsendable)]
    pub struct AgentRunIterator {
        rx: async_channel::Receiver<anyhow::Result<AgentResponse>>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl AgentRunIterator {
        fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        #[gen_stub(override_return_type(type_repr = "typing.Awaitable[AgentResponse]"))]
        fn __anext__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            let rx: async_channel::Receiver<Result<AgentResponse, anyhow::Error>> = self.rx.clone();
            let fut = async move {
                match rx.recv().await {
                    Ok(res) => res.map_err(Into::into),
                    Err(_) => Err(PyStopAsyncIteration::new_err(())),
                }
            };
            let py_fut = pyo3_async_runtimes::tokio::future_into_py(py, fut)?.unbind();
            Ok(py_fut.into())
        }
    }

    #[gen_stub_pyclass]
    #[pyclass(unsendable)]
    pub struct AgentRunSyncIterator {
        rt: &'static tokio::runtime::Runtime,
        rx: async_channel::Receiver<anyhow::Result<AgentResponse>>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl AgentRunSyncIterator {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(&mut self, py: Python<'_>) -> PyResult<AgentResponse> {
            let item = py.detach(|| self.rt.block_on(self.rx.recv()));
            match item {
                Ok(res) => res.map_err(Into::into),
                Err(_) => Err(PyStopIteration::new_err(())),
            }
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl Agent {
        #[new]
        #[pyo3(signature = (lm, tools = None))]
        fn __new__(lm: LangModel, tools: Option<Vec<Tool>>) -> Self {
            Agent::new(lm, tools.unwrap_or_default())
        }

        #[classmethod]
        #[gen_stub(override_return_type(type_repr = "typing.Awaitable[Agent]"))]
        #[pyo3(signature = (lm, tools = None))]
        fn create<'py>(
            _cls: &Bound<'py, PyType>,
            py: Python<'py>,
            lm: LangModel,
            tools: Option<Vec<Tool>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let fut = async move {
                Python::attach(|py| Py::new(py, Agent::new(lm, tools.unwrap_or(vec![]))))
            };
            pyo3_async_runtimes::tokio::future_into_py(py, fut)
        }

        fn __repr__(&self) -> String {
            format!(
                "Agent(lm={}, tools=[{} items])",
                self.get_lm().__repr__(),
                self.tools.len()
            )
        }

        // #[pyo3(signature = (messages, config=None))]
        #[pyo3(name="run", signature = (contents))]
        fn run_py(
            &mut self,
            // messages: Vec<Message>,
            contents: Vec<Part>,
            // config: Option<InferenceConfig>,
        ) -> anyhow::Result<AgentRunIterator> {
            let (_, rx) = spawn(
                self.clone(),
                contents,
                // messages,
                // config.unwrap_or(InferenceConfig::default()),
            )?;
            Ok(AgentRunIterator { rx })
        }

        // #[pyo3(signature = (messages, config=None))]
        #[pyo3(name="run_sync", signature = (contents))]
        fn run_sync_py(
            &mut self,
            // messages: Vec<Message>,
            contents: Vec<Part>,
            // config: Option<InferenceConfig>,
        ) -> anyhow::Result<AgentRunSyncIterator> {
            let (rt, rx) = spawn(
                self.clone(),
                contents,
                // messages,
                // config.unwrap_or(InferenceConfig::default()),
            )?;
            Ok(AgentRunSyncIterator { rt, rx })
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl AgentResponse {
        fn __repr__(&self) -> String {
            format!(
                "AgentResponse(delta={}, finish_reason={}, aggregated={})",
                self.delta.__repr__(),
                self.finish_reason
                    .clone()
                    .map(|finish_reason| finish_reason.__repr__())
                    .unwrap_or("None".to_owned()),
                self.aggregated
                    .clone()
                    .map(|aggregated| aggregated.__repr__())
                    .unwrap_or("None".to_owned()),
            )
        }
    }
}
