---
title: Types
description: "Python reference for the dict shapes in ailoy.types of messages, tools and specs."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `ailoy-py`.*

## `types`

The shapes of the dicts ailoy's Python API takes and gives.

Each is the JSON form of the Rust type of the same name, as ``serde`` writes it, a
``TypedDict`` here only so that an editor can say what the keys are. Nothing checks a dict
against these at runtime; the Rust side does, when the dict arrives.

### `AgentSpec`

#### `model`

```python
model: str
```

#### `instruction`

```python
instruction: NotRequired[str]
```

#### `tools`

```python
tools: NotRequired[list[ToolDesc]]
```

#### `model_options`

```python
model_options: NotRequired[LangModelOptions]
```

#### `web_search_engines`

```python
web_search_engines: NotRequired[list[str]]
```

#### `skills`

```python
skills: NotRequired[list[str]]
```

### `Backend`

```python
Backend = Literal['cli', 'api', 'bedrock', 'openrouter', 'vllm']
```

### `EmbeddedImage`

#### `type`

```python
type: Literal['embedded']
```

#### `mime_type`

```python
mime_type: str
```

#### `data`

```python
data: bytes
```

### `FinishReason`

``{"type": "stop"}``, ``"length"``, ``"tool_call"``, or ``"refusal"`` with a
``reason``.

#### `type`

```python
type: Literal['stop', 'length', 'tool_call', 'refusal']
```

#### `reason`

```python
reason: NotRequired[str]
```

### `FunctionPart`

#### `type`

```python
type: Literal['function']
```

#### `id`

```python
id: str
```

#### `function`

```python
function: PartFunction
```

### `ImagePart`

#### `type`

```python
type: Literal['image']
```

#### `image`

```python
image: Union[EmbeddedImage, UrlImage]
```

### `LangModelOptions`

#### `thinking_effort`

```python
thinking_effort: NotRequired[ThinkingEffort]
```

#### `output_schema`

```python
output_schema: NotRequired[Value]
```

### `Message`

#### `role`

```python
role: Role
```

#### `contents`

```python
contents: list[Part]
```

#### `thinking`

```python
thinking: NotRequired[str]
```

#### `tool_calls`

```python
tool_calls: NotRequired[list[FunctionPart]]
```

#### `id`

```python
id: NotRequired[str]
```

#### `signature`

```python
signature: NotRequired[str]
```

### `MessageDelta`

#### `role`

```python
role: NotRequired[Role]
```

#### `contents`

```python
contents: list[PartDelta]
```

#### `id`

```python
id: NotRequired[str]
```

#### `thinking`

```python
thinking: NotRequired[str]
```

#### `tool_calls`

```python
tool_calls: list[PartDelta]
```

#### `signature`

```python
signature: NotRequired[str]
```

### `MessageDeltaOutput`

What ``Agent.run_stream`` yields. A ``finish_reason`` marks the end of a message.

#### `delta`

```python
delta: MessageDelta
```

#### `finish_reason`

```python
finish_reason: FinishReason | None
```

#### `usage`

```python
usage: TokenUsage | None
```

### `MessageOutput`

What ``Agent.run`` yields: one complete message.

#### `message`

```python
message: Message
```

#### `finish_reason`

```python
finish_reason: FinishReason
```

#### `usage`

```python
usage: NotRequired[TokenUsage]
```

### `Part`

```python
Part = Union[TextPart, FunctionPart, ValuePart, ImagePart]
```

### `PartDelta`

A piece of a part: ``text``, ``function``, ``value``, ``image`` or ``null``, with the
fields of that part as far as they have arrived.

#### `type`

```python
type: Literal['text', 'function', 'value', 'image', 'null']
```

#### `text`

```python
text: NotRequired[str]
```

### `PartFunction`

#### `name`

```python
name: str
```

#### `arguments`

```python
arguments: Value
```

### `Role`

```python
Role = Literal['system', 'user', 'assistant', 'tool']
```

### `TextPart`

#### `type`

```python
type: Literal['text']
```

#### `text`

```python
text: str
```

### `ThinkingEffort`

```python
ThinkingEffort = Literal['low', 'medium', 'high']
```

### `TokenUsage`

``input_tokens`` is every token of input the call processed, cached or not;
``cache_creation_input_tokens`` and ``cache_read_input_tokens`` are the parts of it
written to and read from a prompt cache.

#### `input_tokens`

```python
input_tokens: int
```

#### `output_tokens`

```python
output_tokens: int
```

#### `cache_creation_input_tokens`

```python
cache_creation_input_tokens: NotRequired[int]
```

#### `cache_read_input_tokens`

```python
cache_read_input_tokens: NotRequired[int]
```

### `ToolDesc`

#### `name`

```python
name: str
```

#### `description`

```python
description: NotRequired[str]
```

#### `parameters`

```python
parameters: Value
```

#### `returns`

```python
returns: NotRequired[Value]
```

### `UrlImage`

#### `type`

```python
type: Literal['url']
```

#### `url`

```python
url: str
```

### `Value`

```python
Value = Any
```

### `ValuePart`

#### `type`

```python
type: Literal['value']
```

#### `value`

```python
value: Value
```

### `WebSearchEngine`

```python
WebSearchEngine = Literal[
    "Bing",
    "Brave",
    "DuckDuckGo",
    "Google",
    "Mojeek",
    "Naver",
    "Startpage",
    "Yahoo",
    "Yandex",
]

```
