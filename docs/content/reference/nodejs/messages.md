---
title: "Messages"
description: "The shape of a message and its parts: text, tool calls, values and images."
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

A conversation is a list of [`Message`](#message)s. Each has a `role` and `contents`, a list of parts, each tagged by its `type`. A message is a plain object in its JSON form. These types only spell out the keys for an editor. Nothing checks an object against them at runtime: the Rust side does, when the object arrives.

A query can be a string, taken as the user's text, or a whole message. `agent.history`, `AgentBuilder.history()` and `LangModel.infer()` all use this shape.

## Example

```js
const { AgentBuilder } = require('@brekkylab/ailoy')

/** @type {import('@brekkylab/ailoy').Message} */
const question = {
  role: 'user',
  contents: [
    { type: 'text', text: 'What is in this picture?' },
    { type: 'image', image: { type: 'url', url: 'https://example.com/cat.png' } },
  ],
}

async function main() {
  const agent = await new AgentBuilder('anthropic/claude-sonnet-5-5').build()
  for await (const { message } of agent.run(question)) {
    for (const call of message.tool_calls ?? []) {
      console.log('calls', call.function.name, call.function.arguments)
    }
    for (const part of message.contents) {
      if (part.type === 'text') console.log(part.text)
    }
  }
  await agent.close()
}

main()
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

```ts
interface Message {
  role: Role;
  contents: Part[];
  thinking?: string;
  tool_calls?: FunctionPart[];
  id?: string;
  signature?: string;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `role` | [`Role`](/docs/reference/nodejs/messages/#role) | yes | Who wrote it. |
| `contents` | [`Part`](/docs/reference/nodejs/messages/#part)`[]` | yes | Its parts: text, values, images. |
| `thinking` | `string` | no | The model's thinking before it answered, for models that give it. |
| `tool_calls` | [`FunctionPart`](/docs/reference/nodejs/messages/#functionpart)`[]` | no | The tools the model asks to run, one [`FunctionPart`](#functionpart) each. |
| `id` | `string` | no | For a tool result, the id of the call it answers. |
| `signature` | `string` | no | A signature of `thinking`, for the APIs that need it sent back with the thinking. |

## `Role`

Who wrote a message: the system instruction, the user, the model (`assistant`) or a tool's result (`tool`).

```ts
type Role = "system" | "user" | "assistant" | "tool";
```

## `Part`

One part of a message, tagged by `type`.

```ts
type Part = TextPart | FunctionPart | ValuePart | ImagePart;
```

## `TextPart`

Text.

```ts
interface TextPart {
  type: "text";
  text: string;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `"text"` | yes | `"text"`. |
| `text` | `string` | yes | The text. |

## `FunctionPart`

A tool call the model asks for.

```ts
interface FunctionPart {
  type: "function";
  id: string;
  function: PartFunction;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `"function"` | yes | `"function"`. |
| `id` | `string` | yes | Pairs the call with its result, a tool message with the same `id`. |
| `function` | [`PartFunction`](/docs/reference/nodejs/messages/#partfunction) | yes | The tool and its arguments. |

## `PartFunction`

The tool a call names and its arguments.

```ts
interface PartFunction {
  name: string;
  arguments: any;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | `string` | yes | The tool's name. |
| `arguments` | `any` | yes | The arguments, usually an object the tool's parameter schema describes. |

## `ValuePart`

A structured value, such as a tool's result.

```ts
interface ValuePart {
  type: "value";
  value: any;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `"value"` | yes | `"value"`. |
| `value` | `any` | yes | Any JSON value. |

## `ImagePart`

An image, embedded or by URL.

```ts
interface ImagePart {
  type: "image";
  image: EmbeddedImage | UrlImage;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `"image"` | yes | `"image"`. |
| `image` | [`EmbeddedImage`](/docs/reference/nodejs/messages/#embeddedimage)` \| `[`UrlImage`](/docs/reference/nodejs/messages/#urlimage) | yes | The image. |

## `EmbeddedImage`

An image carried in the message.

```ts
interface EmbeddedImage {
  type: "embedded";
  mime_type: string;
  data: Buffer;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `"embedded"` | yes | `"embedded"`. |
| `mime_type` | `string` | yes | Its type, such as `"image/png"`. |
| `data` | `Buffer` | yes | The image's bytes. |

## `UrlImage`

An image by its URL.

```ts
interface UrlImage {
  type: "url";
  url: string;
}
```

**Fields**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `"url"` | yes | `"url"`. |
| `url` | `string` | yes | Where the image is. |

## `Value`

Anything JSON can hold.

```ts
type Value = any;
```
