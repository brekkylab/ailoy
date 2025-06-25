import base64
import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from functools import partial
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    Union,
)
from urllib.parse import urlencode, urlparse, urlunparse

import jmespath
from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from rich.console import Console
from rich.panel import Panel

from ailoy.ailoy_py import generate_uuid
from ailoy.mcp import MCPServer, MCPTool, StdioServerParameters
from ailoy.models import APIModel, LocalModel
from ailoy.runtime import Runtime
from ailoy.tools import DocstringParsingException, TypeHintParsingException, get_json_schema
from ailoy.utils.image import pillow_image_to_base64

## Types for internal data structures


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    class UrlData(BaseModel):
        url: str

    type: Literal["image_url"] = "image_url"
    image_url: UrlData

    @staticmethod
    def from_url(url: str):
        return ImageContent(image_url={"url": url})

    @staticmethod
    def from_pillow(image: Image):
        return ImageContent(image_url={"url": pillow_image_to_base64(image)})


class AudioContent(BaseModel):
    class AudioData(BaseModel):
        data: str
        format: Literal["mp3", "wav"]

    type: Literal["input_audio"] = "input_audio"
    input_audio: AudioData

    @staticmethod
    def from_bytes(data: bytes, format: Literal["mp3", "wav"]):
        return AudioContent(input_audio={"data": base64.b64encode(data).decode("utf-8"), "format": format})


class FunctionData(BaseModel):
    class FunctionBody(BaseModel):
        name: str
        arguments: Any

    type: Literal["function"] = "function"
    id: Optional[str] = None
    function: FunctionBody


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str | list[TextContent]


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[TextContent | ImageContent | AudioContent]


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str | list[TextContent]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[FunctionData]] = None

    # Non-OpenAI fields
    reasoning: Optional[list[TextContent]] = None


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: str | list[TextContent]
    tool_call_id: Optional[str] = None


Message = Union[
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
]


class MessageOutput(BaseModel):
    message: AssistantMessage
    finish_reason: Optional[Literal["stop", "tool_calls", "invalid_tool_call", "length", "error"]] = None


## Types for agent's responses

_console = Console(highlight=False, force_jupyter=False, force_terminal=True)


class AgentResponseOutputText(BaseModel):
    type: Literal["output_text", "reasoning"]
    role: Literal["assistant"] = "assistant"
    is_type_switched: bool = False
    content: str

    def print(self):
        if self.is_type_switched:
            _console.print()  # add newline if type has been switched
        _console.print(self.content, end="", style=("yellow" if self.type == "reasoning" else None))


class AgentResponseToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    role: Literal["assistant"] = "assistant"
    is_type_switched: bool = False
    content: FunctionData

    def print(self):
        title = f"[magenta]Tool Call[/magenta]: [bold]{self.content.function.name}[/bold]"
        if self.content.id is not None and len(self.content.id) > 0:
            title += f" ({self.content.id})"
        panel = Panel(
            json.dumps(self.content.function.arguments, indent=2),
            title=title,
            title_align="left",
        )
        _console.print(panel)


class AgentResponseToolResult(BaseModel):
    type: Literal["tool_call_result"] = "tool_call_result"
    role: Literal["tool"] = "tool"
    is_type_switched: bool = False
    content: ToolMessage

    def print(self):
        try:
            # Try to parse as json
            content = json.dumps(json.loads(self.content.content[0].text), indent=2)
        except json.JSONDecodeError:
            # Use original content if not json deserializable
            content = self.content.content[0].text
        # Truncate long contents
        if len(content) > 500:
            content = content[:500] + "...(truncated)"

        title = "[green]Tool Result[/green]"
        if self.content.tool_call_id is not None and len(self.content.tool_call_id) > 0:
            title += f" ({self.content.tool_call_id})"
        panel = Panel(
            content,
            title=title,
            title_align="left",
        )
        _console.print(panel)


class AgentResponseError(BaseModel):
    type: Literal["error"] = "error"
    role: Literal["assistant"] = "assistant"
    is_type_switched: bool = False
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
    AgentResponseToolResult,
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
        model: APIModel | LocalModel,
        system_message: Optional[str] = None,
    ):
        """
        Create an instance.

        :param runtime: The runtime environment associated with the agent.
        :param model: The model instance.
        :param system_message: Optional system message to set the initial assistant context.
        :raises ValueError: If model name is not supported or validation fails.
        """
        self._runtime = runtime

        # Initialize component state
        self._component_name = generate_uuid()
        self._component_ready = False

        # Initialize messages
        self._messages: list[Message] = []

        # Initialize system message
        self._system_message = system_message

        # Initialize tools
        self._tools: list[Tool] = []

        # Initialize MCP servers
        self._mcp_servers: list[MCPServer] = []

        # Define the component
        self.define(model)

    def __del__(self):
        self.delete()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.delete()

    def define(self, model: APIModel | LocalModel) -> None:
        """
        Initializes the agent by defining its model in the runtime.
        This must be called before running the agent. If already initialized, this is a no-op.
        :param model: The model instance.
        """
        if self._component_ready:
            return

        if not self._runtime.is_alive():
            raise ValueError("Runtime is currently stopped.")

        # Set default system message if not given; still can be None
        if self._system_message is None:
            self._system_message = getattr(model, "default_system_message", None)

        self.clear_messages()

        # Call runtime's define
        self._runtime.define(
            model.component_type,
            self._component_name,
            model.to_attrs(),
        )

        # Mark as defined
        self._component_ready = True

    def delete(self) -> None:
        """
        Deinitializes the agent and releases resources in the runtime.
        This should be called when the agent is no longer needed. If already deinitialized, this is a no-op.
        """
        if not self._component_ready:
            return

        if self._runtime.is_alive():
            self._runtime.delete(self._component_name)

        self.clear_messages()

        for mcp_server in self._mcp_servers:
            mcp_server.cleanup()

        self._component_ready = False

    def query(
        self,
        message: str | list[str | Image | dict | TextContent | ImageContent | AudioContent],
        reasoning: bool = False,
    ) -> Generator[AgentResponse, None, None]:
        """
        Runs the agent with a new user message and yields streamed responses.

        :param message: The user message to send to the model.
        :param reasoning: If True, enables reasoning capabilities. (Default: False)
        :return: An iterator over the output, where each item represents either a generated token from the assistant or a tool call.
        :rtype: Iterator[:class:`AgentResponse`]
        """  # noqa: E501
        if not self._component_ready:
            raise ValueError("Agent is not valid. Create one or define newly.")

        if not self._runtime.is_alive():
            raise ValueError("Runtime is currently stopped.")

        if isinstance(message, str):
            self._messages.append(UserMessage(content=[TextContent(text=message)]))
        elif isinstance(message, list):
            if len(message) == 0:
                raise ValueError("Message is empty")

            contents = []
            for content in message:
                if isinstance(content, str):
                    contents.append(TextContent(text=content))
                elif isinstance(content, Image):
                    contents.append(ImageContent.from_pillow(image=content))
                elif isinstance(content, dict):
                    ta: TypeAdapter[TextContent | ImageContent | AudioContent] = TypeAdapter(
                        Annotated[TextContent | ImageContent | AudioContent, Field(discriminator="type")]
                    )
                    validated_content = ta.validate_python(content)
                    contents.append(validated_content)
                else:
                    contents.append(content)

            self._messages.append(UserMessage(content=contents))
        else:
            raise ValueError(f"Invalid message type: {type(message)}")

        prev_resp_type = None

        while True:
            infer_args = {
                "messages": [msg.model_dump(exclude_none=True) for msg in self._messages],
                "tools": [{"type": "function", "function": t.desc.model_dump(exclude_none=True)} for t in self._tools],
            }
            if reasoning:
                infer_args["reasoning"] = reasoning

            assistant_reasoning = None
            assistant_content = None
            assistant_tool_calls = None
            finish_reason = ""
            for result in self._runtime.call_iter_method(self._component_name, "infer", infer_args):
                msg = MessageOutput.model_validate(result)

                if msg.message.reasoning:
                    for v in msg.message.reasoning:
                        if not assistant_reasoning:
                            assistant_reasoning = [v]
                        else:
                            assistant_reasoning[0].text += v.text
                        resp = AgentResponseOutputText(
                            type="reasoning",
                            is_type_switched=(prev_resp_type != "reasoning"),
                            content=v.text,
                        )
                        prev_resp_type = resp.type
                        yield resp
                if msg.message.content is not None:
                    # Canonicalize message content to the array of TextContent
                    if isinstance(msg.message.content, str):
                        msg.message.content = [TextContent(text=msg.message.content)]

                    for v in msg.message.content:
                        if not assistant_content:
                            assistant_content = [v]
                        else:
                            assistant_content[0].text += v.text
                        resp = AgentResponseOutputText(
                            type="output_text",
                            is_type_switched=(prev_resp_type != "output_text"),
                            content=v.text,
                        )
                        prev_resp_type = resp.type
                        yield resp
                if msg.message.tool_calls:
                    for v in msg.message.tool_calls:
                        if not assistant_tool_calls:
                            assistant_tool_calls = [v]
                        else:
                            assistant_tool_calls.append(v)
                        resp = AgentResponseToolCall(
                            is_type_switched=True,
                            content=v,
                        )
                        prev_resp_type = resp.type
                        yield resp
                if msg.finish_reason:
                    finish_reason = msg.finish_reason
                    break

            # Append output
            self._messages.append(
                AssistantMessage(
                    reasoning=assistant_reasoning,
                    content=assistant_content,
                    tool_calls=assistant_tool_calls,
                )
            )

            if finish_reason == "tool_calls":

                def run_tool(tool_call: FunctionData) -> ToolMessage:
                    tool_ = next(
                        (t for t in self._tools if t.desc.name == tool_call.function.name),
                        None,
                    )
                    if not tool_:
                        raise RuntimeError("Tool not found")
                    tool_result = tool_.call(**tool_call.function.arguments)
                    return ToolMessage(
                        content=[TextContent(text=json.dumps(tool_result))],
                        tool_call_id=tool_call.id,
                    )

                tool_call_results = [run_tool(tc) for tc in assistant_tool_calls]
                for result_msg in tool_call_results:
                    self._messages.append(result_msg)
                    resp = AgentResponseToolResult(
                        is_type_switched=True,
                        content=result_msg,
                    )
                    prev_resp_type = resp.type
                    yield resp
                # Infer again if tool calls happened
                continue

            # Finish this generator
            yield AgentResponseOutputText(type="output_text", content="\n")
            break

    def get_messages(self) -> list[Message]:
        """
        Get the current conversation history.
        Each item in the list represents a message from either the user or the assistant.

        :return: The conversation history so far in the form of a list.
        :rtype: list[Message]
        """
        return self._messages

    def clear_messages(self):
        """
        Clear the history of conversation messages.
        """
        self._messages.clear()
        if self._system_message is not None:
            self._messages.append(SystemMessage(role="system", content=[TextContent(text=self._system_message)]))

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

    def add_py_function_tool(self, f: Callable[..., Any], desc: Optional[dict] = None):
        """
        Adds a Python function as a tool using callable.

        :param f: Function will be called when the tool invocation occured.
        :param desc: Tool description to override. If not given, parsed from docstring of function `f`.

        :raises ValueError: Docstring parsing is failed.
        :raises ValidationError: Given or parsed description is not a valid `ToolDescription`.
        """
        tool_description = None
        if desc is not None:
            tool_description = ToolDescription.model_validate(desc)

        if tool_description is None:
            try:
                json_schema = get_json_schema(f)
            except (TypeHintParsingException, DocstringParsingException) as e:
                raise ValueError("Failed to parse docstring", e)

            tool_description = ToolDescription.model_validate(json_schema.get("function", {}))

        self.add_tool(Tool(desc=tool_description, call_fn=f))

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

    def add_tools_from_mcp_server(
        self, name: str, params: StdioServerParameters, tools_to_add: Optional[list[str]] = None
    ):
        """
        Create a MCP server and register its tools to agent.

        :param name: The unique name of the MCP server.
                     If there's already a MCP server with the same name, it raises RuntimeError.
        :param params: Parameters for connecting to the MCP stdio server.
        :param tools_to_add: Optional list of tool names to add. If None, all tools are added.
        """
        if any([s.name == name for s in self._mcp_servers]):
            raise RuntimeError(f"MCP server with name '{name}' is already registered")

        # Create and register MCP server
        mcp_server = MCPServer(name, params)
        self._mcp_servers.append(mcp_server)

        # Register tools
        for tool in mcp_server.list_tools():
            # Skip if this tool is not in the whitelist
            if tools_to_add is not None and tool.name not in tools_to_add:
                continue

            desc = ToolDescription(
                name=f"{name}-{tool.name}", description=tool.description, parameters=tool.inputSchema
            )

            def call(tool: MCPTool, **inputs: dict[str, Any]) -> list[str]:
                return mcp_server.call_tool(tool, inputs)

            self.add_tool(Tool(desc=desc, call_fn=partial(call, tool)))

    def remove_mcp_server(self, name: str):
        """
        Removes the MCP server and its tools from the agent, with terminating the MCP server process.

        :param name: The unique name of the MCP server.
                     If there's no MCP server matches the name, it raises RuntimeError.
        """
        if all([s.name != name for s in self._mcp_servers]):
            raise RuntimeError(f"MCP server with name '{name}' does not exist")

        # Remove the MCP server
        mcp_server = next(filter(lambda s: s.name == name, self._mcp_servers))
        self._mcp_servers.remove(mcp_server)
        mcp_server.cleanup()

        # Remove tools registered from the MCP server
        self._tools = list(filter(lambda t: not t.desc.name.startswith(f"{mcp_server.name}-"), self._tools))

    def get_tools(self):
        """
        Get the list of registered tools.
        """
        return self._tools

    def clear_tools(self):
        """
        Clear the registered tools.
        """
        self._tools.clear()
