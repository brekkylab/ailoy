use std::sync::Arc;

use anyhow::Context;
use futures::{StreamExt, lock::Mutex};
use serde::{Deserialize, Serialize};

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
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct Agent {
    lm: LangModel,
    tools: Vec<Tool>,
    messages: Arc<Mutex<Vec<Message>>>,
    knowledge: Option<Knowledge>,
}

/// The yielded value from agent.run().
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
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

    pub async fn clear_messages(&mut self) {
        self.messages.lock().await.clear();
    }

    pub fn add_tools(&mut self, tools: Vec<Tool>) {
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
    }

    pub fn add_tool(&mut self, tool: Tool) {
        self.add_tools(vec![tool]);
    }

    pub fn remove_tools(&mut self, tool_names: Vec<String>) {
        self.tools.retain(|t| {
            let tool_name = t.get_description().name;
            // Remove the tool if its name belongs to `tool_names`
            !tool_names.contains(&tool_name)
        });
    }

    pub fn remove_tool(&mut self, tool_name: String) {
        self.remove_tools(vec![tool_name]);
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
        config: Option<InferenceConfig>,
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
                    let mut strm = model.infer(messages.clone(), tool_descs.clone(), docs.clone(), config.clone().unwrap_or_default());
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

#[cfg(feature = "nodejs")]
mod node {
    use std::sync::Arc;

    use futures::lock::Mutex;
    use napi::{JsSymbol, Status, bindgen_prelude::*};
    use napi_derive::napi;
    use tokio::sync::mpsc;

    use super::*;
    use crate::ffi::node::common::get_or_create_runtime;

    #[napi(object)]
    pub struct AgentRunIteratorResult {
        pub value: AgentResponse,
        pub done: bool,
    }

    #[derive(Clone)]
    #[napi]
    pub struct AgentRunIterator {
        rx: Arc<Mutex<mpsc::UnboundedReceiver<anyhow::Result<AgentResponse>>>>,
    }

    #[napi]
    impl AgentRunIterator {
        #[napi(js_name = "[Symbol.asyncIterator]")]
        pub fn async_iterator(&self) -> &Self {
            // This is a dummy function to add typing for Symbol.asyncIterator
            self
        }

        #[napi]
        pub async unsafe fn next(&mut self) -> napi::Result<AgentRunIteratorResult> {
            let mut rx = self.rx.lock().await;
            match rx.recv().await {
                Some(Ok(response)) => Ok(AgentRunIteratorResult {
                    value: response,
                    done: false,
                }),
                Some(Err(e)) => Err(napi::Error::new(Status::GenericFailure, e)),
                None => Ok(AgentRunIteratorResult {
                    value: AgentResponse::default(),
                    done: true,
                }),
            }
        }
    }

    impl AgentRunIterator {
        fn to_async_iterator<'a>(self, env: Env) -> napi::Result<Object<'a>> {
            let mut obj = Object::new(&env)?;

            let global = env.get_global()?;
            let symbol: Function = global.get_named_property("Symbol")?;
            let symbol_async_iterator: JsSymbol = symbol.get_named_property("asyncIterator")?;

            let func: Function<(), AgentRunIterator> =
                env.create_function_from_closure("asyncIterator", move |_| Ok(self.clone()))?;

            obj.set_property(symbol_async_iterator, func)?;

            Ok(obj)
        }
    }

    #[napi]
    impl Agent {
        #[napi(constructor)]
        pub fn new_js(lm: &LangModel, tools: Option<Vec<&Tool>>) -> Self {
            Self::new(
                lm.clone(),
                tools.unwrap_or(vec![]).iter().map(|&t| t.clone()),
            )
        }

        #[napi(js_name = "addTool")]
        pub unsafe fn add_tool_js(&mut self, tool: &Tool) {
            self.add_tool(tool.clone())
        }

        #[napi(js_name = "addTools")]
        pub unsafe fn add_tools_js(&mut self, tools: Vec<&Tool>) {
            self.add_tools(tools.iter().map(|&t| t.clone()).collect())
        }

        #[napi(js_name = "removeTool")]
        pub unsafe fn remove_tool_js(&mut self, tool_name: String) {
            self.remove_tool(tool_name)
        }

        #[napi(js_name = "removeTools")]
        pub unsafe fn remove_tools_js(&mut self, tool_names: Vec<String>) {
            self.remove_tools(tool_names)
        }

        #[napi(js_name = "setKnowledge")]
        pub unsafe fn set_knowledge_js(&mut self, knowledge: &Knowledge) {
            self.set_knowledge(knowledge.clone())
        }

        #[napi(js_name = "removeKnowledge")]
        pub unsafe fn remove_knowledge_js(&mut self) {
            self.remove_knowledge()
        }

        #[napi(js_name = "run", ts_return_type = "AgentRunIterator")]
        pub fn run_js<'a>(
            &'a mut self,
            env: Env,
            contents: Vec<Part>,
            config: Option<InferenceConfig>,
        ) -> napi::Result<Object<'a>> {
            let (tx, rx) = mpsc::unbounded_channel::<anyhow::Result<AgentResponse>>();
            let rt = get_or_create_runtime();
            let mut agent = self.clone();

            rt.spawn(async move {
                let mut stream = agent.run(contents, config).boxed();

                while let Some(item) = stream.next().await {
                    if tx
                        .send(item.map_err(|e| anyhow::anyhow!(e.to_string())))
                        .is_err()
                    {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            });

            let it = AgentRunIterator {
                rx: Arc::new(Mutex::new(rx)),
            };
            it.to_async_iterator(env)
        }
    }
}

#[cfg(feature = "wasm")]
mod wasm {
    use wasm_bindgen::prelude::*;

    use super::*;
    use crate::ffi::web::stream_to_async_iterable;

    #[wasm_bindgen]
    impl Agent {
        /// Construct a new Agent instance with provided `LangModel` and `Tool`s.
        ///
        /// Note that the ownership of `tools` is moved to the agent, which means you can't directly accessible to `tools` after the agent is initialized.
        /// If you still want to reuse the `tools`, try to use `addTool()` multiple times instead.
        #[wasm_bindgen(constructor)]
        pub fn new_js(lm: &LangModel, tools: Option<Vec<Tool>>) -> Self {
            Self::new(lm.clone(), tools.unwrap_or(vec![]))
        }

        #[wasm_bindgen(js_name = "addTool")]
        pub fn add_tool_js(&mut self, tool: &Tool) {
            self.add_tool(tool.clone())
        }

        #[wasm_bindgen(js_name = "removeTool")]
        pub fn remove_tool_js(&mut self, #[wasm_bindgen(js_name = "toolName")] tool_name: String) {
            self.remove_tool(tool_name)
        }

        #[wasm_bindgen(js_name = "setKnowledge")]
        pub fn set_knowledge_js(&mut self, knowledge: &Knowledge) {
            self.set_knowledge(knowledge.clone())
        }

        #[wasm_bindgen(js_name = "removeKnowledge")]
        pub fn remove_knowledge_js(&mut self) {
            self.remove_knowledge()
        }

        #[wasm_bindgen(
            js_name = "run",
            unchecked_return_type = "AsyncIterable<AgentResponse>"
        )]
        pub fn run_js(&self, contents: Vec<Part>, config: Option<InferenceConfig>) -> JsValue {
            let mut agent = self.clone();
            let stream = async_stream::stream! {
                let mut inner_stream = agent.run(contents, config);
                while let Some(item) = inner_stream.next().await {
                    yield item;
                }
            };
            let js_stream = Box::pin(stream.map(|response| {
                response
                    .map(|resp| resp.into())
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            }));

            stream_to_async_iterable(js_stream).into()
        }
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

        let mut strm = Box::pin(agent.run(vec![Part::text("Hi what's your name?")], None));
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
        use crate::{
            model::LangModel,
            to_value,
            tool::ToolFunc,
            value::{ToolDesc, Value},
        };

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
            Arc::<Box<ToolFunc>>::new(Box::new(move |args: Value| {
                Box::pin(async move {
                    use anyhow::bail;

                    let unit = args
                        .as_object()
                        .unwrap()
                        .get("unit")
                        .unwrap()
                        .as_str()
                        .unwrap();
                    match unit {
                        "Celsius" => Ok(to_value!("40")),
                        "Fahrenheit" => Ok(to_value!("104")),
                        _ => bail!(""),
                    }
                })
            })),
        )];

        let mut agent = Agent::new(model, tools);

        let mut strm =
            Box::pin(agent.run(vec![Part::text("How much hot currently in Dubai?")], None));
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
            .get_tools()
            .iter()
            .map(|tool| Tool::new_mcp(tool.clone()))
            .collect();
        agent.add_tools(mcp_tools);

        let agent_tools = agent.get_tools();
        assert_eq!(agent_tools.len(), 2);
        assert_eq!(agent_tools[0].get_description().name, "get_current_time");
        assert_eq!(agent_tools[1].get_description().name, "convert_time");

        let mut strm = Box::pin(agent.run(
            vec![Part::text(
                "What time is it now in America/New_York timezone?",
            )],
            None,
        ));
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
        config: Option<InferenceConfig>,
    ) -> anyhow::Result<(
        &'a tokio::runtime::Runtime,
        async_channel::Receiver<anyhow::Result<AgentResponse>>,
    )> {
        let (tx, rx) = async_channel::unbounded::<anyhow::Result<AgentResponse>>();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.spawn(async move {
            let mut stream = agent.run(contents, config).boxed();

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

        pub fn __repr__(&self) -> String {
            format!(
                "Agent(lm={}, tools=[{} items])",
                self.get_lm().__repr__(),
                self.tools.len()
            )
        }

        #[getter]
        fn lm(&self) -> LangModel {
            self.get_lm()
        }

        #[getter]
        fn tools(&self) -> Vec<Tool> {
            self.get_tools()
        }

        #[pyo3(name="add_tools", signature = (tools))]
        fn add_tools_py(&mut self, tools: Vec<Tool>) {
            self.add_tools(tools);
        }

        #[pyo3(name="add_tool", signature = (tool))]
        fn add_tool_py(&mut self, tool: Tool) {
            self.add_tool(tool);
        }

        #[pyo3(name="remove_tools", signature = (tool_names))]
        fn remove_tools_py(&mut self, tool_names: Vec<String>) {
            self.remove_tools(tool_names);
        }

        #[pyo3(name="remove_tool", signature = (tool_name))]
        fn remove_tool_py(&mut self, tool_name: String) {
            self.remove_tool(tool_name);
        }

        // #[pyo3(name="run", signature = (messages, config=None))]
        #[pyo3(name="run", signature = (contents, config=None))]
        fn run_py(
            &mut self,
            // messages: Vec<Message>,
            contents: Vec<Part>,
            config: Option<InferenceConfig>,
        ) -> anyhow::Result<AgentRunIterator> {
            let (_, rx) = spawn(
                self.clone(),
                // messages,
                contents,
                config,
            )?;
            Ok(AgentRunIterator { rx })
        }

        // #[pyo3(name="run_sync", signature = (messages, config=None))]
        #[pyo3(name="run_sync", signature = (contents, config=None))]
        fn run_sync_py(
            &mut self,
            // messages: Vec<Message>,
            contents: Vec<Part>,
            config: Option<InferenceConfig>,
        ) -> anyhow::Result<AgentRunSyncIterator> {
            let (rt, rx) = spawn(
                self.clone(),
                // messages,
                contents,
                config,
            )?;
            Ok(AgentRunSyncIterator { rt, rx })
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl AgentResponse {
        pub fn __repr__(&self) -> String {
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
