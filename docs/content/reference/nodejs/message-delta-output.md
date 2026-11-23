---
title: MessageDeltaOutput
description: What Agent.runStream yields.
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [v0.3.2](https://github.com/brekkylab/ailoy/tree/v0.3.2) release of `@brekkylab/ailoy`.*

What `Agent.runStream` yields. A `finish_reason` marks the end of a message.

## Properties

| Property | Type |
| ------ | ------ |
| <a id="property-delta"></a> `delta` | [`MessageDelta`](/docs/reference/nodejs/message-delta/) |
| <a id="property-finish_reason"></a> `finish_reason` | [`FinishReason`](/docs/reference/nodejs/finish-reason/) \| `null` |
| <a id="property-usage"></a> `usage` | [`TokenUsage`](/docs/reference/nodejs/token-usage/) \| `null` |
