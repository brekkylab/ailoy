---
title: "Messages"
description: "The shape of a message and its parts: text, tool calls, values and images."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

A conversation is a list of [`Message`](#message)s. Each has a `role` and `contents`, a list of parts, each tagged by its `type`. A message is a plain dict in its JSON form. The `TypedDict`s in `ailoy.types` only spell out the keys for an editor or a type checker. Nothing checks a dict against them at runtime: the Rust side does, when the dict arrives.

A query can be a string, taken as the user's text, or a whole message. `Agent.history`, `AgentBuilder.history()` and `LangModel.infer()` all use this shape.

## Example

```python
import asyncio

from ailoy import AgentBuilder
from ailoy.types import Message

question: Message = {
    "role": "user",
    "contents": [
        {"type": "text", "text": "What is in this picture?"},
        {"type": "image", "image": {"type": "url", "url": "https://example.com/cat.png"}},
    ],
}


async def main() -> None:
    async with await AgentBuilder("anthropic/claude-sonnet-5-5").build() as agent:
        async for out in agent.run(question):
            message = out["message"]
            for call in message.get("tool_calls", []):
                print("calls", call["function"]["name"], call["function"]["arguments"])
            for part in message["contents"]:
                if part["type"] == "text":
                    print(part["text"])


asyncio.run(main())
```

Guides: [Message format](/docs/guides/message-format/).

## On this page

| Name | Kind | Description |
| --- | --- | --- |
| [`Message`](#message) | type | One message of a conversation. |
| [`Role`](#role) | type | Who wrote a message: the system instruction, the user, the model (`assistant`) or a tool's result (`tool`). |
| [`Part`](#part) | type | One part of a message, tagged by `type`. |
| [`TextPart`](#textpart) | type | Text. |
| [`FunctionPart`](#functionpart) | type | A tool call the model asks for. |
| [`PartFunction`](#partfunction) | type | The tool a call names and its arguments. |
| [`ValuePart`](#valuepart) | type | A structured value, such as a tool's result. |
| [`ImagePart`](#imagepart) | type | An image, embedded or by URL. |
| [`EmbeddedImage`](#embeddedimage) | type | An image carried in the message. |
| [`UrlImage`](#urlimage) | type | An image by its URL. |
| [`Value`](#value) | type | Anything JSON can hold. |

## `Message`

One message of a conversation.

```python
class Message(TypedDict):
    role: Role
    contents: list[Part]
    thinking: NotRequired[str]
    tool_calls: NotRequired[list[FunctionPart]]
    id: NotRequired[str]
    signature: NotRequired[str]
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `role` | [`Role`](/docs/reference/python/messages/#role) | yes | Who wrote it. |
| `contents` | `list[`[`Part`](/docs/reference/python/messages/#part)`]` | yes | Its parts: text, values, images. |
| `thinking` | `str` | no | The model's thinking before it answered, for models that give it. |
| `tool_calls` | `list[`[`FunctionPart`](/docs/reference/python/messages/#functionpart)`]` | no | The tools the model asks to run, one [`FunctionPart`](#functionpart) each. |
| `id` | `str` | no | For a tool result, the id of the call it answers. |
| `signature` | `str` | no | A signature of `thinking`, for the APIs that need it sent back with the thinking. |

## `Role`

Who wrote a message: the system instruction, the user, the model (`assistant`) or a tool's result (`tool`).

```python
Role = Literal["system", "user", "assistant", "tool"]
```

## `Part`

One part of a message, tagged by `type`.

```python
Part = Union[TextPart, FunctionPart, ValuePart, ImagePart]
```

## `TextPart`

Text.

```python
class TextPart(TypedDict):
    type: Literal['text']
    text: str
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal['text']` | yes | `"text"`. |
| `text` | `str` | yes | The text. |

## `FunctionPart`

A tool call the model asks for.

```python
class FunctionPart(TypedDict):
    type: Literal['function']
    id: str
    function: PartFunction
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal['function']` | yes | `"function"`. |
| `id` | `str` | yes | Pairs the call with its result, a tool message with the same `id`. |
| `function` | [`PartFunction`](/docs/reference/python/messages/#partfunction) | yes | The tool and its arguments. |

## `PartFunction`

The tool a call names and its arguments.

```python
class PartFunction(TypedDict):
    name: str
    arguments: Value
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | `str` | yes | The tool's name. |
| `arguments` | [`Value`](/docs/reference/python/messages/#value) | yes | The arguments, usually an object the tool's parameter schema describes. |

## `ValuePart`

A structured value, such as a tool's result.

```python
class ValuePart(TypedDict):
    type: Literal['value']
    value: Value
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal['value']` | yes | `"value"`. |
| `value` | [`Value`](/docs/reference/python/messages/#value) | yes | Any JSON value. |

## `ImagePart`

An image, embedded or by URL.

```python
class ImagePart(TypedDict):
    type: Literal['image']
    image: Union[EmbeddedImage, UrlImage]
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal['image']` | yes | `"image"`. |
| `image` | `Union[`[`EmbeddedImage`](/docs/reference/python/messages/#embeddedimage)`, `[`UrlImage`](/docs/reference/python/messages/#urlimage)`]` | yes | The image. |

## `EmbeddedImage`

An image carried in the message.

```python
class EmbeddedImage(TypedDict):
    type: Literal['embedded']
    mime_type: str
    data: bytes
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal['embedded']` | yes | `"embedded"`. |
| `mime_type` | `str` | yes | Its type, such as `"image/png"`. |
| `data` | `bytes` | yes | The image's bytes. |

## `UrlImage`

An image by its URL.

```python
class UrlImage(TypedDict):
    type: Literal['url']
    url: str
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal['url']` | yes | `"url"`. |
| `url` | `str` | yes | Where the image is. |

## `Value`

Anything JSON can hold.

```python
Value = Any
```
