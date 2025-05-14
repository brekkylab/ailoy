import asyncio
import json
import subprocess
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
)
from urllib.parse import urlencode, urlparse, urlunparse

import jmespath
import mcp
import mcp.types as mcp_types
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ailoy.ailoy_py import generate_uuid
from ailoy.runtime import AsyncRuntime

## Types for OpenAI API-compatible data structures


class SystemMessage(BaseModel):
    role: Literal["system"]
    content: str


class UserMessage(BaseModel):
    role: Literal["user"]
    content: str


class AIOutputTextMessage(BaseModel):
    role: Literal["assistant"]
    content: str
    reasoning: Optional[bool] = None


class AIToolCallMessage(BaseModel):
    role: Literal["assistant"]
    content: None
    tool_calls: List["ToolCall"]


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: "ToolCallFunction"


class ToolCallFunction(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ToolCallResultMessage(BaseModel):
    role: Literal["tool"]
    name: str
    tool_call_id: str
    content: str


Message = Union[
    SystemMessage,
    UserMessage,
    AIOutputTextMessage,
    AIToolCallMessage,
    ToolCallResultMessage,
]


class MessageDelta(BaseModel):
    finish_reason: Optional[Literal["stop", "tool_calls", "length", "error"]]
    message: Message


## Types for LLM Model Definitions

TVMModelName = Literal["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b"]
OpenAIModelName = Literal["gpt-4o"]
AvailableModel = Union[TVMModelName, OpenAIModelName]


class TVMModel(BaseModel):
    name: TVMModelName
    quantization: Optional[Literal["q4f16_1"]] = None
    mode: Optional[Literal["interactive"]] = None


class OpenAIModel(BaseModel):
    name: OpenAIModelName
    api_key: str


class ModelDescription(BaseModel):
    model_id: str
    component_type: str
    default_system_message: Optional[str] = None


model_descriptions: Dict[AvailableModel, ModelDescription] = {
    "qwen3-0.6b": ModelDescription(
        model_id="Qwen/Qwen3-0.6B",
        component_type="tvm_language_model",
        default_system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    ),
    "qwen3-1.7b": ModelDescription(
        model_id="Qwen/Qwen3-1.7B",
        component_type="tvm_language_model",
        default_system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    ),
    "qwen3-4b": ModelDescription(
        model_id="Qwen/Qwen3-4B",
        component_type="tvm_language_model",
        default_system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    ),
    "qwen3-8b": ModelDescription(
        model_id="Qwen/Qwen3-8B",
        component_type="tvm_language_model",
        default_system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    ),
    "gpt-4o": ModelDescription(
        model_id="gpt-4o",
        component_type="openai",
    ),
}


class ModelInfo(BaseModel):
    component_name: str
    component_type: Literal["tvm_language_model", "openai"]
    attrs: Dict[str, Any]


## Types for agent's responses


class AgentResponseBase(BaseModel):
    type: Literal["output_text", "tool_call", "tool_call_result", "reasoning", "error"]
    end_of_turn: bool
    role: Literal["assistant", "tool"]
    content: Any


class AgentResponseOutputText(AgentResponseBase):
    type: Literal["output_text", "reasoning"]
    role: Literal["assistant"]
    content: str


class AgentResponseToolCall(AgentResponseBase):
    type: Literal["tool_call"]
    role: Literal["assistant"]
    content: ToolCall


class AgentResponseToolCallResult(AgentResponseBase):
    type: Literal["tool_call_result"]
    role: Literal["tool"]
    content: ToolCallResultMessage


class AgentResponseError(AgentResponseBase):
    type: Literal["error"]
    role: Literal["assistant"]
    content: str


AgentResponse = Union[
    AgentResponseOutputText,
    AgentResponseToolCall,
    AgentResponseToolCallResult,
    AgentResponseError,
]

## Types and functions related to Tools

ToolDefinition = Union["UniversalToolDefinition", "RESTAPIToolDefinition"]


class ToolDescription(BaseModel):
    name: str
    description: str
    parameters: "ToolParameters"
    return_type: Optional[Dict[str, Any]] = Field(default=None, alias="return")
    model_config = ConfigDict(populate_by_name=True)


class ToolParameters(BaseModel):
    type: Literal["object"]
    properties: Dict[str, "ToolParametersProperty"]
    required: Optional[List[str]] = []


class ToolParametersProperty(BaseModel):
    type: Literal["string", "number", "boolean", "object", "array", "null"]
    description: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class UniversalToolDefinition(BaseModel):
    type: Literal["universal"]
    description: ToolDescription
    behavior: "UniversalToolBehavior"


class UniversalToolBehavior(BaseModel):
    output_path: Optional[str] = Field(default=None, alias="outputPath")
    model_config = ConfigDict(populate_by_name=True)


class RESTAPIToolDefinition(BaseModel):
    type: Literal["restapi"]
    description: ToolDescription
    behavior: "RESTAPIBehavior"


class RESTAPIBehavior(BaseModel):
    base_url: str = Field(alias="baseURL")
    method: Literal["GET", "POST", "PUT", "DELETE"]
    authentication: Optional[Literal["bearer"]] = None
    headers: Optional[Dict[str, str]] = None
    body: Optional[str] = None
    output_path: Optional[str] = Field(default=None, alias="outputPath")
    model_config = ConfigDict(populate_by_name=True)


class Tool:
    def __init__(
        self,
        desc: ToolDescription,
        call_fn: Callable[[AsyncRuntime, Any], Awaitable[Any]],
    ):
        self.desc = desc
        self.call = call_fn  # should be async def or callable


class ToolAuthenticator(ABC):
    @abstractmethod
    def apply(self, request: Dict[str, Any]) -> Dict[str, Any]:
        pass


class BearerAuthenticator(ToolAuthenticator):
    def __init__(self, token: str, bearer_format: str = "Bearer"):
        self.token = token
        self.bearer_format = bearer_format

    def apply(self, request: Dict[str, Any]) -> Dict[str, Any]:
        headers = request.get("headers", {})
        headers["Authorization"] = f"{self.bearer_format} {self.token}"
        return {**request, "headers": headers}


class Agent:
    def __init__(
        self,
        runtime: AsyncRuntime,
        model_name: AvailableModel,
        tools: Optional[List[Tool]] = None,
        system_message: Optional[str] = None,
        **kwargs,
    ):
        self._runtime = runtime
        self._initialized = False

        try:
            if model_name in get_args(TVMModelName):
                model = TVMModel(name=model_name, **kwargs)
            elif model_name in get_args(OpenAIModelName):
                model = OpenAIModel(name=model_name, **kwargs)
            else:
                raise ValueError(f"Model {model_name} not supported")
        except ValidationError as e:
            raise ValueError(f"Failed to validate model parameters\n{e}")

        model_attrs = {k: v for k, v in model.model_dump(exclude_none=True).items() if k != "name"}

        if model_name not in model_descriptions:
            raise ValueError(f'Model "{model_name}" is not available')

        model_desc = model_descriptions[model_name]
        self._model_info = ModelInfo(
            component_type=model_desc.component_type,
            component_name=generate_uuid(),
            attrs={
                "model": model_desc.model_id,
                **model_attrs,
            },
        )
        self._messages: List[Message] = []

        system_message = system_message if system_message is not None else model_desc.default_system_message
        if system_message is not None:
            self._messages.append(SystemMessage(role="system", content=system_message))

        self._tools: List[Tool] = tools if tools is not None else []

    def __del__(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.deinitialize())
            else:
                loop.run_until_complete(self.deinitialize())
        except Exception:
            pass

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self._runtime.define(
            self._model_info.component_type,
            self._model_info.component_name,
            self._model_info.attrs,
        )
        self._initialized = True

    async def deinitialize(self) -> None:
        if not self._initialized:
            return
        await self._runtime.delete(self._model_info.component_name)
        self._initialized = False

    async def run(
        self,
        message: str,
        do_reasoning: Optional[bool] = None,
        ignore_reasoning: Optional[bool] = None,
    ) -> AsyncGenerator[AgentResponse, None]:
        self._messages.append(UserMessage(role="user", content=message))

        while True:
            infer_args = {
                "messages": [msg.model_dump() for msg in self._messages],
                "tools": [{"type": "function", "function": t.desc.model_dump()} for t in self._tools],
            }
            if do_reasoning is not None:
                infer_args["reasoning"] = do_reasoning
            if ignore_reasoning is not None:
                infer_args["ignore_reasoning"] = ignore_reasoning

            async for resp in self._runtime.call_iter_method(self._model_info.component_name, "infer", infer_args):
                delta = MessageDelta.model_validate(resp)

                if delta.finish_reason is None:
                    output_msg = AIOutputTextMessage.model_validate(delta.message)
                    yield AgentResponseOutputText(
                        type="reasoning" if output_msg.reasoning else "output_text",
                        end_of_turn=False,
                        role="assistant",
                        content=output_msg.content,
                    )
                    continue

                if delta.finish_reason == "tool_calls":
                    tool_call_message = AIToolCallMessage.model_validate(delta.message)
                    self._messages.append(tool_call_message)

                    for tool_call in tool_call_message.tool_calls:
                        yield AgentResponseToolCall(
                            type="tool_call",
                            end_of_turn=True,
                            role="assistant",
                            content=tool_call,
                        )

                    tool_call_results: List[ToolCallResultMessage] = []

                    async def run_tool(tool_call: ToolCall):
                        tool_ = next(
                            (t for t in self._tools if t.desc.name == tool_call.function.name),
                            None,
                        )
                        if not tool_:
                            raise RuntimeError("Tool not found")
                        resp = await tool_.call(self._runtime, tool_call.function.arguments)
                        return ToolCallResultMessage(
                            role="tool",
                            name=tool_call.function.name,
                            tool_call_id=tool_call.id,
                            content=json.dumps(resp),
                        )

                    tool_call_results = await asyncio.gather(*(run_tool(tc) for tc in tool_call_message.tool_calls))

                    for result_msg in tool_call_results:
                        self._messages.append(result_msg)
                        yield AgentResponseToolCallResult(
                            type="tool_call_result",
                            end_of_turn=True,
                            role="tool",
                            content=result_msg,
                        )

                    # Run infer again with new messages
                    break

                if delta.finish_reason in ["stop", "length", "error"]:
                    output_msg = AIOutputTextMessage.model_validate(delta.message)
                    yield AgentResponseOutputText(
                        type="reasoning" if output_msg.reasoning else "output_text",
                        end_of_turn=False,
                        role="assistant",
                        content=output_msg.content,
                    )

                    # finish this AsyncGenerator
                    return

    def add_tool(self, tool: Tool) -> None:
        if any(t.desc.name == tool.desc.name for t in self._tools):
            warnings.warn(f'Tool "{tool.desc.name}" is already added.')
            return
        self._tools.append(tool)

    def add_py_function_tool(self, desc: ToolDescription, f: Callable[[AsyncRuntime, Any], Awaitable[Any]]):
        self.add_tool(Tool(desc=desc, call_fn=f))

    def add_universal_tool(self, tool_def: UniversalToolDefinition) -> bool:
        if tool_def.type != "universal":
            raise ValueError('Tool type is not "universal"')

        async def call(runtime: AsyncRuntime, inputs: Dict[str, Any]) -> Any:
            required = tool_def.description.parameters.required or []
            for param_name in required:
                if param_name not in inputs:
                    raise ValueError(f'Parameter "{param_name}" is required but not provided')

            output = await runtime.call(tool_def.description.name, inputs)
            if tool_def.behavior.output_path is not None:
                output = jmespath.search(tool_def.behavior.output_path, output)

            return output

        return self.add_tool(Tool(desc=tool_def.description, call_fn=call))

    def add_restapi_tool(
        self,
        tool_def: RESTAPIToolDefinition,
        authenticator: Optional[ToolAuthenticator] = None,
    ) -> bool:
        if tool_def.type != "restapi":
            raise ValueError('Tool type is not "restapi"')

        behavior = tool_def.behavior

        async def call(_: AsyncRuntime, inputs: Dict[str, Any]) -> Any:
            def render_template(template: str, context: Dict[str, Any]) -> Tuple[str, List[str]]:
                import re

                variables = set()

                def replacer(match: re.Match):
                    key = match.group(1).strip()
                    variables.add(key)
                    return str(context.get(key, f"{{{key}}}"))

                rendered_url = re.sub(r"\$\{\s*([^}\s]+)\s*\}", replacer, template)
                return rendered_url, list(variables)

            # Handle path parameters
            url, path_vars = render_template(behavior.base_url, inputs)

            # Handle body
            if behavior.body is not None:
                body, body_vars = render_template(behavior.body, inputs)
            else:
                body, body_vars = None, []

            # Handle query parameters
            query_params = {k: v for k, v in inputs.items() if k not in set(path_vars + body_vars)}

            # Construct a full URL
            full_url = urlunparse(urlparse(url)._replace(query=urlencode(query_params)))

            # Construct a request payload
            request = {
                "url": full_url,
                "method": behavior.method,
                "headers": behavior.headers,
            }
            if body:
                request["body"] = body

            # Apply authentication
            if authenticator is not None:
                request = authenticator.apply(request)

            # Call HTTP request
            output = None
            resp = await self._runtime.call("http_request", request)
            output = json.loads(resp["body"])

            # Parse output path if defined
            if behavior.output_path is not None:
                output = jmespath.search(tool_def.behavior.output_path, output)

            return output

        return self.add_tool(Tool(desc=tool_def.description, call_fn=call))

    def add_tools_from_preset(self, preset_name: str, authenticator: Optional[ToolAuthenticator] = None):
        tool_presets_path = Path(__file__).parent / "presets" / "tools"
        preset_json = tool_presets_path / f"{preset_name}.json"
        if not preset_json.exists():
            raise ValueError(f'Tool preset "{preset_name}" does not exist')

        data: Dict[str, Dict[str, Any]] = json.loads(preset_json.read_text())
        for tool_name, tool_def in data.items():
            tool_type = tool_def.get("type", None)
            if tool_type == "universal":
                self.add_universal_tool(UniversalToolDefinition.model_validate(tool_def))
            elif tool_type == "restapi":
                self.add_restapi_tool(RESTAPIToolDefinition.model_validate(tool_def), authenticator=authenticator)
            else:
                warnings.warn(f'Tool type "{tool_type}" is not supported. Skip adding tool "{tool_name}".')

    def add_mcp_tool(self, params: mcp.StdioServerParameters, tool: mcp_types.Tool):
        from mcp.client.stdio import stdio_client

        async def call(_: AsyncRuntime, inputs: Dict[str, Any]) -> Any:
            async with stdio_client(params, errlog=subprocess.STDOUT) as streams:
                async with mcp.ClientSession(*streams) as session:
                    await session.initialize()

                    result = await session.call_tool(tool.name, inputs)
                    contents: List[str] = []
                    for item in result.content:
                        if isinstance(item, mcp_types.TextContent):
                            contents.append(item.text)
                        elif isinstance(item, mcp_types.ImageContent):
                            contents.append(item.data)
                        elif isinstance(item, mcp_types.EmbeddedResource):
                            if isinstance(item.resource, mcp_types.TextResourceContents):
                                contents.append(item.resource.text)
                            else:
                                contents.append(item.resource.blob)

                    return contents

        desc = ToolDescription(name=tool.name, description=tool.description, parameters=tool.inputSchema)
        return self.add_tool(Tool(desc=desc, call_fn=call))

    async def add_tools_from_mcp_server(
        self, params: mcp.StdioServerParameters, tools_to_add: Optional[List[str]] = None
    ):
        from mcp.client.stdio import stdio_client

        async with stdio_client(params, errlog=subprocess.STDOUT) as streams:
            async with mcp.ClientSession(*streams) as session:
                await session.initialize()
                resp = await session.list_tools()
                for tool in resp.tools:
                    if tools_to_add is None or tool.name in tools_to_add:
                        self.add_mcp_tool(params, tool)
                return resp.tools
