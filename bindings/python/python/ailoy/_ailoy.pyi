from collections.abc import AsyncIterable, Awaitable, Callable
from types import TracebackType
from typing import Any, Generic, Literal, Protocol, Self, TypeVar

from virtx import ConsoleClient

from .types import (
    AgentSpec,
    Backend,
    LangModelOptions,
    Message,
    MessageDeltaOutput,
    MessageOutput,
    ThinkingEffort,
    ToolDesc,
    Value,
    WebSearchEngine,
)

_T = TypeVar("_T")

class AiloyError(Exception): ...

class AgentBuilder:
    """Filled in place and emptied by ``build()``. Every setter returns this builder."""

    def __init__(self, model: str) -> None: ...
    def agent_provider(self, name: str) -> Self: ...
    def instruction(self, instruction: str) -> Self: ...
    def tool(self, desc: ToolDesc) -> Self: ...
    def tools(self, descs: list[ToolDesc]) -> Self: ...
    def system_tools(self) -> Self: ...
    def shell_tool(self) -> Self: ...
    def web_search_tool(self, engines: list[WebSearchEngine] | None = None) -> Self: ...
    def web_fetch_tool(self) -> Self: ...
    def history(self, history: list[Message]) -> Self: ...
    def console(self, console: ConsoleClient) -> Self: ...
    def browser(
        self,
        *,
        ask_user: Literal["end_turn", "wait"] = "end_turn",
        user_data_dir: _Path | None = None,
        on_ask: Callable[[str], str | None | Awaitable[str | None]] | None = None,
        ask_timeout: float | None = None,
        careful: bool = True,
        start_url: str | None = None,
    ) -> Self:
        """The person's own Chrome, which the agent drives with its ``browser_*`` tools from
        this process rather than the console.

        It needs Chrome 144 or later with remote debugging on at
        ``chrome://inspect/#remote-debugging``, and asks the person before it lets the agent
        in; ``user_data_dir`` picks another Chrome's.

        ``ask_user`` is how the agent waits on the person: ``"end_turn"`` tells them in its
        reply and ends the turn, ``"wait"`` waits for ``on_ask``. ``on_ask`` is given the
        request and returns ``"done"``, ``"declined"``, or any other string, a note for the
        agent; a raise is a decline. ``ask_timeout`` bounds it, in seconds.

        ``careful`` tells the agent to use sites as a person would, so sites checking for bots
        are less likely to turn it away.

        ``start_url`` is the page the agent's tab opens on.
        """
        ...
    def memory(self, memfile: str) -> Self: ...
    def skill(self, dir: str) -> Self: ...
    def context_manager(
        self,
        max_input_tokens: int | None = None,
        preserve_recent_turns: int | None = None,
    ) -> Self: ...
    def thinking_effort(self, effort: ThinkingEffort) -> Self: ...
    def output_schema(self, schema: dict[str, Value]) -> Self: ...
    async def build(self) -> Agent: ...

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
        agent_provider: str = "default",
        history: list[Message] | None = None,
        console: ConsoleClient | None = None,
        memory: str | None = None,
    ) -> Agent: ...
    @property
    def history(self) -> list[Message]: ...
    async def start_browser(self) -> None:
        """Connect the browser now rather than on the agent's first use.

        This is when Chrome asks the person, so call it while they are at their screen; a
        refusal raises. A no-op once connected. Raises during a turn.
        """
        ...
    def run(self, query: str | Message) -> AgentRun[MessageOutput]: ...
    def run_stream(self, query: str | Message) -> AgentRun[MessageDeltaOutput]: ...
    async def close(self) -> None: ...
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

def set_lang_model_provider(org: str, provider: Backend | LangModelFactory) -> None: ...
def remove_lang_model_provider(org: str) -> bool: ...
def add_tool_provider(name: str, *, builtins: bool = True) -> None: ...
def register_tool(
    desc: ToolDesc,
    func: Callable[..., Any | Awaitable[Any]],
    *,
    provider: str = "default",
) -> ToolDesc: ...
async def register_mcp_stdio(
    prefix: str, command: str, args: list[str] = ..., *, provider: str = "default"
) -> list[ToolDesc]: ...
async def register_mcp_streamable_http(
    prefix: str, url: str, *, provider: str = "default"
) -> list[ToolDesc]: ...
def unregister_mcp(prefix: str, *, provider: str = "default") -> int: ...
async def register_a2a(
    name: str, url: str, *, provider: str = "default"
) -> ToolDesc: ...
def add_agent_provider(name: str, *, tool_provider: str = "default") -> None: ...
