import asyncio
import json
import threading
from concurrent.futures import Future
from contextlib import AsyncExitStack
from typing import Any, Coroutine

import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp import Tool as MCPTool
from mcp.client.stdio import stdio_client


class MCPServer:
    def __init__(self, name: str, params: StdioServerParameters):
        self.name = name
        self.params = params

        self._session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._exit_stack: AsyncExitStack = AsyncExitStack()

        self._loop = asyncio.new_event_loop()
        self._queue = asyncio.Queue[tuple[Coroutine, Future]]()
        self._ready = threading.Event()
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _start_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._thread_loop())

    async def _thread_loop(self):
        try:
            await self._initialize()
            self._ready.set()

            while not self._stop.is_set():
                try:
                    coro, fut = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                try:
                    result = await coro()
                    fut.set_result(result)
                except Exception as e:
                    fut.set_exception(e)

        finally:
            await self._cleanup()

    def _run(self, coro):
        self._ready.wait()
        fut = Future()
        self._loop.call_soon_threadsafe(lambda: self._queue.put_nowait((coro, fut)))
        return fut.result()

    async def _initialize(self):
        try:
            stdio_transport = await self._exit_stack.enter_async_context(stdio_client(self.params))
            read, write = stdio_transport
            session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session
        except Exception:
            await self._cleanup()
            raise

    async def _list_tools(self):
        if not self._session:
            raise RuntimeError("MCP server not initialized")

        resp = await self._session.list_tools()
        return resp.tools

    async def _call_tool(self, tool: MCPTool, arguments: dict[str, Any]):
        if not self._session:
            raise RuntimeError("MCP server not initialized")

        try:
            result = await self._session.call_tool(tool.name, arguments)
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
            return contents
        except Exception as e:
            print(f"Error executing tool {tool.name}: {e}")
            raise

    async def _cleanup(self):
        async with self._cleanup_lock:
            try:
                await self._exit_stack.aclose()
                self._session = None
            except Exception as e:
                print(f"Error during cleanup of MCP server {self.name}: {e}")

    def list_tools(self) -> list[MCPTool]:
        return self._run(lambda: self._list_tools())

    def call_tool(self, tool: MCPTool, arguments: dict[str, Any]) -> list[str]:
        return self._run(lambda: self._call_tool(tool, arguments))

    def cleanup(self) -> None:
        self._stop.set()
        self._thread.join()
