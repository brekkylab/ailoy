import asyncio
import json
import multiprocessing
import platform
import tempfile
from multiprocessing.connection import Connection
from typing import Annotated, Any, Literal, Union

import mcp.types as mcp_types
from mcp import Tool as MCPTool
from mcp.client.session import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    stdio_client,
)
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, Field, TypeAdapter

__all__ = ["MCPServer"]


class ListToolsRequest(BaseModel):
    type: Literal["list_tools"] = "list_tools"


class CallToolRequest(BaseModel):
    type: Literal["call_tool"] = "call_tool"
    tool: MCPTool
    arguments: dict[str, Any]


class ShutdownRequest(BaseModel):
    type: Literal["shutdown"] = "shutdown"


# Requests (main -> subprocess)
RequestMessage = Annotated[Union[ListToolsRequest, CallToolRequest, ShutdownRequest], Field(discriminator="type")]


class ResultMessage(BaseModel):
    type: Literal["result"] = "result"
    result: Any


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    error: str


# Response (subprocess -> main)
ResponseMessage = Annotated[Union[ResultMessage, ErrorMessage], Field(discriminator="type")]


class MCPServer:
    """
    MCPServer manages a subprocess that acts as a bridge between an MCP stdio server and the main process.

    - The subprocess communicates with the MCP stdio server using the official MCP Python SDK.
    - Communication between the main process and the subprocess is handled through a multiprocessing Pipe.
      Messages sent over this Pipe are serialized and deserialized using structured Pydantic models:
        - `RequestMessage` for requests from the main process to the subprocess.
        - `ResponseMessage` for responses from the subprocess to the main process.

    This design ensures:
    - Type-safe, structured inter-process communication.
    - Synchronous interaction with an asynchronous MCP session (via message passing).
    - Subprocess lifecycle control (including initialization and shutdown).
    """

    def __init__(self, name: str, params: StdioServerParameters):
        self.name = name
        self.params = params

        self._parent_conn, self._child_conn = multiprocessing.Pipe()

        ctx = multiprocessing.get_context("fork" if platform.system() != "Windows" else "spawn")
        self._proc: multiprocessing.Process = ctx.Process(target=self._run_process, args=(self._child_conn,))
        self._proc.start()

        # Wait for subprocess to signal initialization complete
        try:
            self._recv_response()
        except RuntimeError as e:
            self.cleanup()
            raise e

    def __del__(self):
        self.cleanup()

    def _run_process(self, conn: Connection):
        asyncio.run(self._process_main(conn))

    async def _process_main(self, conn: Connection):
        with tempfile.TemporaryFile(mode="w+t") as _errlog:
            async with stdio_client(self.params, errlog=_errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    # Notify to main process that the initialization has been finished and ready to receive requests
                    try:
                        await session.initialize()
                        conn.send(ResultMessage(result=True).model_dump())
                    except McpError:
                        _errlog.seek(0)
                        error = _errlog.read()
                        conn.send(
                            ErrorMessage(
                                error=f"Failed to initialize MCP subprocess. Check the error output below.\n\n{error}"
                            ).model_dump()
                        )

                    while True:
                        if not conn.poll(0.1):
                            await asyncio.sleep(0.1)
                            continue

                        try:
                            raw = conn.recv()
                            req = TypeAdapter(RequestMessage).validate_python(raw)

                            if isinstance(req, ListToolsRequest):
                                result = await session.list_tools()
                                conn.send(ResultMessage(result=result.tools).model_dump())

                            elif isinstance(req, CallToolRequest):
                                result = await session.call_tool(req.tool.name, req.arguments)
                                contents: list[str] = []
                                for item in result.content:
                                    if isinstance(item, mcp_types.TextContent):
                                        try:
                                            content = json.loads(item.text)
                                            contents.append(json.dumps(content))
                                        except json.JSONDecodeError:
                                            contents.append(item.text)
                                    elif isinstance(item, mcp_types.ImageContent):
                                        contents.append(item.data)
                                    elif isinstance(item, mcp_types.EmbeddedResource):
                                        if isinstance(item.resource, mcp_types.TextResourceContents):
                                            contents.append(item.resource.text)
                                        else:
                                            contents.append(item.resource.blob)
                                conn.send(ResultMessage(result=contents).model_dump())

                            elif isinstance(req, ShutdownRequest):
                                break

                        except Exception as e:
                            conn.send(ErrorMessage(error=str(e)).model_dump())

    def _send_request(self, msg: RequestMessage):
        self._parent_conn.send(msg.model_dump())

    def _recv_response(self) -> ResultMessage:
        raw = self._parent_conn.recv()
        msg = TypeAdapter(ResponseMessage).validate_python(raw)
        if isinstance(msg, ErrorMessage):
            raise RuntimeError(msg.error)
        return msg

    def list_tools(self) -> list[MCPTool]:
        self._send_request(ListToolsRequest())
        msg = self._recv_response()
        return [MCPTool.model_validate(tool) for tool in msg.result]

    def call_tool(self, tool: MCPTool, arguments: dict[str, Any]) -> list[str]:
        self._send_request(CallToolRequest(tool=tool, arguments=arguments))
        msg = self._recv_response()
        return msg.result

    def cleanup(self) -> None:
        if self._proc.is_alive():
            self._send_request(ShutdownRequest())
            self._proc.join()
