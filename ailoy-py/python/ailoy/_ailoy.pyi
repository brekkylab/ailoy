from collections.abc import AsyncIterable, Awaitable, Callable
from types import TracebackType
from typing import Any, Generic, Literal, Protocol, Self, TypeVar

from virtx import Client

from .types import (
    AgentSpec,
    Backend,
    LangModelOptions,
    LangModelServer,
    Message,
    MessageDeltaOutput,
    MessageOutput,
    SessionSandbox,
    ToolDesc,
    WebSearchProvider,
)

_T = TypeVar("_T")

class AiloyError(Exception): ...

class AgentRun(Generic[_T]):
    """One turn, iterated with ``async for``. Nothing runs until it is iterated."""

    def __aiter__(self) -> Self: ...
    async def __anext__(self) -> _T: ...
    async def aclose(self) -> None: ...

class Agent:
    @staticmethod
    async def from_spec(
        spec: AgentSpec,
        *,
        history: list[Message] | None = None,
        sandbox: SessionSandbox | None = None,
    ) -> Agent:
        """An agent of ``spec``, starting from ``history``. A spec with ``use_sandbox`` works in
        a sandbox made as ``sandbox`` says with the client given to ``set_virtx``, which goes
        with the agent; one with ``use_browser`` drives its ``browser`` abin, which the sandbox
        has to be made with."""
        ...
    @property
    def sandbox_id(self) -> str | None:
        """The id of the sandbox made for the agent, where the client given to ``set_virtx``
        reaches its files."""
        ...
    async def history(self) -> list[Message]:
        """The conversation so far, as the server keeps it."""
        ...
    def run(self, query: str | Message) -> AgentRun[MessageOutput]: ...
    def run_stream(self, query: str | Message) -> AgentRun[MessageDeltaOutput]: ...
    async def close(self) -> None:
        """Let go of the agent and its session, and remove its sandbox."""
        ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

class LangModel:
    """A model built from its desc, as an ``Agent`` builds one, run directly."""

    @staticmethod
    async def create(desc: str) -> LangModel: ...
    async def infer(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDesc] | None = None,
        options: LangModelOptions | None = None,
    ) -> MessageOutput: ...
    def infer_stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDesc] | None = None,
        options: LangModelOptions | None = None,
    ) -> AgentRun[MessageDeltaOutput]: ...

class CustomLangModel(Protocol):
    """A model of your own, as a factory given to ``set_lang_model_provider`` returns it.
    ``infer_stream`` is optional: without it, streaming gives ``infer``'s output as one delta."""

    def infer(
        self, messages: list[Message], tools: list[ToolDesc], options: LangModelOptions
    ) -> MessageOutput | Awaitable[MessageOutput]: ...

class StreamingCustomLangModel(CustomLangModel, Protocol):
    def infer_stream(
        self, messages: list[Message], tools: list[ToolDesc], options: LangModelOptions
    ) -> AsyncIterable[MessageDeltaOutput]: ...

LangModelFactory = Callable[[str, str | None], LangModel | CustomLangModel]

def set_lang_model_provider(
    org: str, provider: Backend | LangModelServer | LangModelFactory
) -> None: ...
def get_lang_model_provider(desc: str) -> Backend | Literal["custom"] | None: ...
def remove_lang_model_provider(org: str) -> bool: ...
def set_virtx(client: Client | None) -> None:
    """Make each agent's sandbox with ``client`` from now on; with ``None``, make none."""
    ...

def set_web_search_provider(provider: WebSearchProvider) -> None: ...
def get_web_search_provider() -> WebSearchProvider: ...
def remove_tool_provider(name: str) -> bool: ...
def register_tool(desc: ToolDesc, func: Callable[..., Any | Awaitable[Any]]) -> str: ...
async def register_mcp_stdio(prefix: str, command: str, args: list[str] = ...) -> list[str]: ...
async def register_mcp_streamable_http(prefix: str, url: str) -> list[str]: ...
def unregister_mcp(prefix: str) -> int: ...
