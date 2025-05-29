import json
import subprocess
import warnings
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Generator
from pathlib import Path
from typing import (
    Any,
    Literal,
    Optional,
    TypeVar,
    Union,
)
from urllib.parse import urlencode, urlparse, urlunparse

import jmespath
import mcp
import mcp.types as mcp_types
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.panel import Panel

from ailoy.ailoy_py import generate_uuid
from ailoy.runtime import Runtime

__all__ = ["Agent"]

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
    tool_calls: list["ToolCall"]


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: "ToolCallFunction"


class ToolCallFunction(BaseModel):
    name: str
    arguments: dict[str, Any]


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

TVMModelName = Literal["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
OpenAIModelName = Literal["gpt-4o"]
ModelName = Union[TVMModelName, OpenAIModelName]


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


model_descriptions: dict[ModelName, ModelDescription] = {
    "Qwen/Qwen3-0.6B": ModelDescription(
        model_id="Qwen/Qwen3-0.6B",
        component_type="tvm_language_model",
        default_system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    ),
    "Qwen/Qwen3-1.7B": ModelDescription(
        model_id="Qwen/Qwen3-1.7B",
        component_type="tvm_language_model",
        default_system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    ),
    "Qwen/Qwen3-4B": ModelDescription(
        model_id="Qwen/Qwen3-4B",
        component_type="tvm_language_model",
        default_system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    ),
    "Qwen/Qwen3-8B": ModelDescription(
        model_id="Qwen/Qwen3-8B",
        component_type="tvm_language_model",
        default_system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    ),
    "gpt-4o": ModelDescription(
        model_id="gpt-4o",
        component_type="openai",
    ),
}


class ComponentState(BaseModel):
    name: str
    valid: bool


## Types for agent's responses

_console = Console(highlight=False)


class AgentResponseBase(BaseModel):
    type: Literal["output_text", "tool_call", "tool_call_result", "reasoning", "error"]
    end_of_turn: bool
    role: Literal["assistant", "tool"]
    content: Any

    def print(self):
        raise NotImplementedError


class AgentResponseOutputText(AgentResponseBase):
    type: Literal["output_text", "reasoning"]
    role: Literal["assistant"]
    content: str

    def print(self):
        _console.print(self.content, end="", style=("yellow" if self.type == "reasoning" else None))
        if self.end_of_turn:
            _console.print()


class AgentResponseToolCall(AgentResponseBase):
    type: Literal["tool_call"]
    role: Literal["assistant"]
    content: ToolCall

    def print(self):
        panel = Panel(
            json.dumps(self.content.function.arguments, indent=2),
            title=f"[magenta]Tool Call[/magenta]: [bold]{self.content.function.name}[/bold] ({self.content.id})",
            title_align="left",
        )
        _console.print(panel)


class AgentResponseToolCallResult(AgentResponseBase):
    type: Literal["tool_call_result"]
    role: Literal["tool"]
    content: ToolCallResultMessage

    def print(self):
        try:
            # Try to parse as json
            content = json.dumps(json.loads(self.content.content), indent=2)
        except json.JSONDecodeError:
            # Use original content if not json deserializable
            content = self.content.content
        # Truncate long contents
        if len(content) > 500:
            content = content[:500] + "...(truncated)"

        panel = Panel(
            content,
            title=f"[green]Tool Result[/green]: [bold]{self.content.name}[/bold] ({self.content.tool_call_id})",
            title_align="left",
        )
        _console.print(panel)


class AgentResponseError(AgentResponseBase):
    type: Literal["error"]
    role: Literal["assistant"]
    content: str

    def print(self):
        panel = Panel(
            self.content,
            title="[bold red]Error[/bold red]",
        )
        _console.print(panel)


AgentResponse = Union[
    AgentResponseOutputText,
    AgentResponseToolCall,
    AgentResponseToolCallResult,
    AgentResponseError,
]

## Types and functions related to Tools

ToolDefinition = Union["BuiltinToolDefinition", "RESTAPIToolDefinition"]


class ToolDescription(BaseModel):
    name: str
    description: str
    parameters: "ToolParameters"
    return_type: Optional[dict[str, Any]] = Field(default=None, alias="return")
    model_config = ConfigDict(populate_by_name=True)


class ToolParameters(BaseModel):
    type: Literal["object"]
    properties: dict[str, "ToolParametersProperty"]
    required: Optional[list[str]] = []


class ToolParametersProperty(BaseModel):
    type: Literal["string", "number", "boolean", "object", "array", "null"]
    description: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class BuiltinToolDefinition(BaseModel):
    type: Literal["builtin"]
    description: ToolDescription
    behavior: "BuiltinToolBehavior"


class BuiltinToolBehavior(BaseModel):
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
    headers: Optional[dict[str, str]] = None
    body: Optional[str] = None
    output_path: Optional[str] = Field(default=None, alias="outputPath")
    model_config = ConfigDict(populate_by_name=True)


class Tool:
    def __init__(
        self,
        desc: ToolDescription,
        call_fn: Callable[..., Any],
    ):
        self.desc = desc
        self.call = call_fn


class ToolAuthenticator(ABC):
    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.apply(request)

    @abstractmethod
    def apply(self, request: dict[str, Any]) -> dict[str, Any]:
        pass


class BearerAuthenticator(ToolAuthenticator):
    def __init__(self, token: str, bearer_format: str = "Bearer"):
        self.token = token
        self.bearer_format = bearer_format

    def apply(self, request: dict[str, Any]) -> dict[str, Any]:
        headers = request.get("headers", {})
        headers["Authorization"] = f"{self.bearer_format} {self.token}"
        return {**request, "headers": headers}


T_Retval = TypeVar("T_Retval")


def run_async(coro: Callable[..., Awaitable[T_Retval]]) -> T_Retval:
    try:
        import anyio

        # Running outside async loop
        return anyio.run(lambda: coro)
    except RuntimeError:
        import anyio.from_thread

        # Already in a running event loop: use anyio from_thread
        return anyio.from_thread.run(coro)


class Agent:
    """
    The `Agent` class provides a high-level interface for interacting with large language models (LLMs) in Ailoy.
    It abstracts the underlying runtime and VM logic, allowing users to easily send queries and receive streaming
    responses.

    Agents can be extended with external tools or APIs to provide real-time or domain-specific knowledge, enabling
    more powerful and context-aware interactions.
    """

    def __init__(
        self,
        runtime: Runtime,
        model_name: ModelName,
        system_message: Optional[str] = None,
        api_key: Optional[str] = None,
        **attrs,
    ):
        """
        Create an instance.

        :param runtime: The runtime environment associated with the agent.
        :param model_name: The name of the LLM model to use.
        :param system_message: Optional system message to set the initial assistant context.
        :param api_key: (web agent only) The API key for AI API.
        :param attrs: Additional initialization parameters (for `define_component` runtime call)
        :raises ValueError: If model name is not supported or validation fails.
        """
        self._runtime = runtime

        # Initialize component state
        self._component_state = ComponentState(
            name=generate_uuid(),
            valid=False,
        )

        # Initialize messages
        self._messages: list[Message] = []
        if system_message:
            self._messages.append(SystemMessage(role="system", content=system_message))

        # Initialize tools
        self._tools: list[Tool] = []

        # Define the component
        self.define(model_name, api_key=api_key, **attrs)

    def __del__(self):
        self.delete()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.delete()

    def define(self, model_name: ModelName, api_key: Optional[str] = None, **attrs) -> None:
        """
        Initializes the agent by defining its model in the runtime.
        This must be called before running the agent. If already initialized, this is a no-op.
        :param model_name: The name of the LLM model to use.
        :param api_key: (web agent only) The API key for AI API.
        :param attrs: Additional initialization parameters (for `define_component` runtime call)
        """
        if self._component_state.valid:
            return

        if model_name not in model_descriptions:
            raise ValueError(f"Model `{model_name}` not supported")

        model_desc = model_descriptions[model_name]

        # Add model name into attrs
        if "model" not in attrs:
            attrs["model"] = model_desc.model_id

        # Set default system message
        if len(self._messages) == 0 and model_desc.default_system_message:
            self._messages.append(SystemMessage(role="system", content=model_desc.default_system_message))

        # Add API key
        if api_key:
            attrs["api_key"] = api_key

        # Call runtime's define
        self._runtime.define(
            model_descriptions[model_name].component_type,
            self._component_state.name,
            attrs,
        )

        # Mark as defined
        self._component_state.valid = True

    def delete(self) -> None:
        """
        Deinitializes the agent and releases resources in the runtime.
        This should be called when the agent is no longer needed. If already deinitialized, this is a no-op.
        """
        if not self._component_state.valid:
            return
        self._runtime.delete(self._component_state.name)
        if len(self._messages) > 0 and self._messages[0].role == "system":
            self._messages = [self._messages[0]]
        else:
            self._messages = []
        self._component_state.valid = False

    def query(
        self,
        message: str,
        enable_reasoning: bool = False,
        ignore_reasoning_messages: bool = False,
    ) -> Generator[AgentResponse, None, None]:
        """
        Runs the agent with a new user message and yields streamed responses.

        :param message: The user message to send to the model.
        :param enable_reasoning: If True, enables reasoning capabilities. (default: False)
        :param ignore_reasoning_messages: If True, reasoning steps are not included in the response stream. (default: False)
        :yield: AgentResponse output of the LLM inference or tool calls
        """  # noqa: E501
        self._messages.append(UserMessage(role="user", content=message))

        while True:
            infer_args = {
                "messages": [msg.model_dump() for msg in self._messages],
                "tools": [{"type": "function", "function": t.desc.model_dump()} for t in self._tools],
            }
            if enable_reasoning:
                infer_args["enable_reasoning"] = enable_reasoning
            if ignore_reasoning_messages:
                infer_args["ignore_reasoning_messages"] = ignore_reasoning_messages

            for resp in self._runtime.call_iter_method(self._component_state.name, "infer", infer_args):
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

                    tool_call_results: list[ToolCallResultMessage] = []

                    def run_tool(tool_call: ToolCall):
                        tool_ = next(
                            (t for t in self._tools if t.desc.name == tool_call.function.name),
                            None,
                        )
                        if not tool_:
                            raise RuntimeError("Tool not found")
                        resp = tool_.call(**tool_call.function.arguments)
                        return ToolCallResultMessage(
                            role="tool",
                            name=tool_call.function.name,
                            tool_call_id=tool_call.id,
                            content=json.dumps(resp),
                        )

                    tool_call_results = [run_tool(tc) for tc in tool_call_message.tool_calls]

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
                        end_of_turn=True,
                        role="assistant",
                        content=output_msg.content,
                    )

                    # finish this Generator
                    return

    def print(self, resp: AgentResponse):
        resp.print()

    def add_tool(self, tool: Tool) -> None:
        """
        Adds a custom tool to the agent.

        :param tool: Tool instance to be added.
        """
        if any(t.desc.name == tool.desc.name for t in self._tools):
            warnings.warn(f'Tool "{tool.desc.name}" is already added.')
            return
        self._tools.append(tool)

    def add_py_function_tool(self, desc: dict, f: Callable[..., Any]):
        """
        Adds a Python function as a tool using callable.

        :param desc: Tool descriotion.
        :param f: Function will be called when the tool invocation occured.
        """
        self.add_tool(Tool(desc=ToolDescription.model_validate(desc), call_fn=f))

    def add_builtin_tool(self, tool_def: BuiltinToolDefinition) -> bool:
        """
        Adds a built-in tool.

        :param tool_def: The built-in tool definition.
        :returns: True if the tool was successfully added.
        :raises ValueError: If the tool type is not "builtin" or required inputs are missing.
        """
        if tool_def.type != "builtin":
            raise ValueError('Tool type is not "builtin"')

        def call(**inputs: dict[str, Any]) -> Any:
            required = tool_def.description.parameters.required or []
            for param_name in required:
                if param_name not in inputs:
                    raise ValueError(f'Parameter "{param_name}" is required but not provided')

            output = self._runtime.call(tool_def.description.name, inputs)
            if tool_def.behavior.output_path is not None:
                output = jmespath.search(tool_def.behavior.output_path, output)

            return output

        return self.add_tool(Tool(desc=tool_def.description, call_fn=call))

    def add_restapi_tool(
        self,
        tool_def: RESTAPIToolDefinition,
        authenticator: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> bool:
        """
        Adds a REST API tool that performs external HTTP requests.

        :param tool_def: REST API tool definition.
        :param authenticator: Optional authenticator to inject into the request.
        :returns: True if the tool was successfully added.
        :raises ValueError: If the tool type is not "restapi".
        """
        if tool_def.type != "restapi":
            raise ValueError('Tool type is not "restapi"')

        behavior = tool_def.behavior

        def call(**inputs: dict[str, Any]) -> Any:
            def render_template(template: str, context: dict[str, Any]) -> tuple[str, list[str]]:
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
            if callable(authenticator):
                request = authenticator(request)

            # Call HTTP request
            output = None
            resp = self._runtime.call("http_request", request)
            output = json.loads(resp["body"])

            # Parse output path if defined
            if behavior.output_path is not None:
                output = jmespath.search(tool_def.behavior.output_path, output)

            return output

        return self.add_tool(Tool(desc=tool_def.description, call_fn=call))

    def add_tools_from_preset(
        self, preset_name: str, authenticator: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None
    ):
        """
        Loads tools from a predefined JSON preset file.

        :param preset_name: Name of the tool preset.
        :param authenticator: Optional authenticator to use for REST API tools.
        :raises ValueError: If the preset file is not found.
        """
        tool_presets_path = Path(__file__).parent / "presets" / "tools"
        preset_json = tool_presets_path / f"{preset_name}.json"
        if not preset_json.exists():
            raise ValueError(f'Tool preset "{preset_name}" does not exist')

        data: dict[str, dict[str, Any]] = json.loads(preset_json.read_text())
        for tool_name, tool_def in data.items():
            tool_type = tool_def.get("type", None)
            if tool_type == "builtin":
                self.add_builtin_tool(BuiltinToolDefinition.model_validate(tool_def))
            elif tool_type == "restapi":
                self.add_restapi_tool(RESTAPIToolDefinition.model_validate(tool_def), authenticator=authenticator)
            else:
                warnings.warn(f'Tool type "{tool_type}" is not supported. Skip adding tool "{tool_name}".')

    def add_mcp_tool(self, params: mcp.StdioServerParameters, tool: mcp_types.Tool):
        """
        Adds a tool from an MCP (Model Context Protocol) server.

        :param params: Parameters for connecting to the MCP stdio server.
        :param tool: Tool metadata as defined by MCP.
        :returns: True if the tool was successfully added.
        """
        from mcp.client.stdio import stdio_client

        def call(**inputs: dict[str, Any]) -> Any:
            async def _inner():
                async with stdio_client(params, errlog=subprocess.STDOUT) as streams:
                    async with mcp.ClientSession(*streams) as session:
                        await session.initialize()

                        result = await session.call_tool(tool.name, inputs)
                        contents: list[str] = []
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

            return run_async(_inner())

        desc = ToolDescription(name=tool.name, description=tool.description, parameters=tool.inputSchema)
        return self.add_tool(Tool(desc=desc, call_fn=call))

    def add_tools_from_mcp_server(self, params: mcp.StdioServerParameters, tools_to_add: Optional[list[str]] = None):
        """
        Fetches tools from an MCP stdio server and registers them with the agent.

        :param params: Parameters for connecting to the MCP stdio server.
        :param tools_to_add: Optional list of tool names to add. If None, all tools are added.
        :returns: list of all tools returned by the server.
        """
        from mcp.client.stdio import stdio_client

        async def _inner():
            async with stdio_client(params, errlog=subprocess.STDOUT) as streams:
                async with mcp.ClientSession(*streams) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    for tool in resp.tools:
                        if tools_to_add is None or tool.name in tools_to_add:
                            self.add_mcp_tool(params, tool)
                    return resp.tools

        tools = run_async(_inner())
        return tools
