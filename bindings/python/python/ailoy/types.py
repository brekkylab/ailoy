"""The shapes of the dicts ailoy's Python API takes and gives.

Each is the JSON form of the Rust type of the same name, as ``serde`` writes it — a
``TypedDict`` here only so that an editor can say what the keys are. Nothing checks a dict
against these at runtime; the Rust side does, when the dict arrives.
"""

from __future__ import annotations

import sys
from typing import Any, Literal, TypedDict, Union

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:  # pragma: no cover
    from typing_extensions import NotRequired

Role = Literal["system", "user", "assistant", "tool"]

# Anything JSON can hold.
Value = Any


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class PartFunction(TypedDict):
    name: str
    arguments: Value


class FunctionPart(TypedDict):
    type: Literal["function"]
    id: str
    function: PartFunction


class ValuePart(TypedDict):
    type: Literal["value"]
    value: Value


class EmbeddedImage(TypedDict):
    type: Literal["embedded"]
    mime_type: str
    data: bytes


class UrlImage(TypedDict):
    type: Literal["url"]
    url: str


class ImagePart(TypedDict):
    type: Literal["image"]
    image: Union[EmbeddedImage, UrlImage]


Part = Union[TextPart, FunctionPart, ValuePart, ImagePart]


class Message(TypedDict):
    role: Role
    contents: list[Part]
    thinking: NotRequired[str]
    tool_calls: NotRequired[list[FunctionPart]]
    id: NotRequired[str]
    signature: NotRequired[str]


class FinishReason(TypedDict):
    """``{"type": "stop"}``, ``"length"``, ``"tool_call"``, or ``"refusal"`` with a
    ``reason``."""

    type: Literal["stop", "length", "tool_call", "refusal"]
    reason: NotRequired[str]


class TokenUsage(TypedDict):
    """``input_tokens`` is every token of input the call processed, cached or not;
    ``cache_creation_input_tokens`` and ``cache_read_input_tokens`` are the parts of it
    written to and read from a prompt cache."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: NotRequired[int]
    cache_read_input_tokens: NotRequired[int]


class MessageOutput(TypedDict):
    """What ``Agent.run`` yields: one complete message."""

    message: Message
    finish_reason: FinishReason
    usage: NotRequired[TokenUsage]


class PartDelta(TypedDict):
    """A piece of a part: ``text``, ``function``, ``value``, ``image`` or ``null``, with the
    fields of that part as far as they have arrived."""

    type: Literal["text", "function", "value", "image", "null"]
    text: NotRequired[str]


class MessageDelta(TypedDict):
    role: NotRequired[Role]
    contents: list[PartDelta]
    id: NotRequired[str]
    thinking: NotRequired[str]
    tool_calls: list[PartDelta]
    signature: NotRequired[str]


class MessageDeltaOutput(TypedDict):
    """What ``Agent.run_stream`` yields. A ``finish_reason`` marks the end of a message."""

    delta: MessageDelta
    finish_reason: FinishReason | None
    usage: TokenUsage | None


class ToolDesc(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: Value
    returns: NotRequired[Value]


ThinkingEffort = Literal["low", "medium", "high"]


class LangModelOptions(TypedDict):
    thinking_effort: NotRequired[ThinkingEffort]
    output_schema: NotRequired[Value]


class AgentSpec(TypedDict):
    model: str
    instruction: NotRequired[str]
    tools: NotRequired[list[ToolDesc]]
    model_options: NotRequired[LangModelOptions]
    web_search_engines: NotRequired[list[str]]
    skills: NotRequired[list[str]]


Backend = Literal["cli", "api", "bedrock", "openrouter", "vllm", "openai-compatible"]


class LangModelServer(TypedDict):
    """A backend on a server of your own, given to ``set_lang_model_provider`` in place of a
    backend's name. ``vllm`` names the model ``<org>/<name>``, the Hugging Face repo id vLLM
    serves it as; ``openai-compatible`` names it ``<name>`` as it is, the org being your name
    for the server."""

    backend: Literal["vllm", "openai-compatible"]
    base_url: str
    api_key: NotRequired[str]

WebSearchEngine = Literal[
    "Bing", "Brave", "DuckDuckGo", "Google", "Mojeek", "Naver", "Startpage", "Yahoo", "Yandex"
]

__all__ = [
    "AgentSpec",
    "Backend",
    "EmbeddedImage",
    "FinishReason",
    "FunctionPart",
    "ImagePart",
    "LangModelOptions",
    "LangModelServer",
    "Message",
    "MessageDelta",
    "MessageDeltaOutput",
    "MessageOutput",
    "Part",
    "PartDelta",
    "PartFunction",
    "Role",
    "TextPart",
    "ThinkingEffort",
    "TokenUsage",
    "ToolDesc",
    "UrlImage",
    "Value",
    "ValuePart",
    "WebSearchEngine",
]
