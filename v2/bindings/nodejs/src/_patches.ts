import { Agent as _Agent, LangModel as _LangModel } from "./ailoy_core";
import type {
  AgentRunIterator,
  Document,
  InferenceConfig,
  LangModelRunIterator,
  Message,
  Part,
  Role,
  ToolDesc,
} from "./ailoy_core";

interface SimpleMessage {
  role: Role;
  contents: string;
  id?: string;
  thinking?: string;
  tool_calls?: Array<Part>;
  signature?: string;
}

export class Agent extends _Agent {
  run(
    messages: Array<SimpleMessage | Message>,
    config?: InferenceConfig | undefined | null
  ): AgentRunIterator {
    const new_messages = messages.map((message) => ({
      ...message,
      contents: Array.isArray(message.contents)
        ? message.contents
        : [{ type: "text", text: message.contents } as Part],
    }));
    return super.run(new_messages, config);
  }
}

export class LangModel extends _LangModel {
  run(
    messages: Array<SimpleMessage | Message>,
    tools?: Array<ToolDesc> | undefined | null,
    docs?: Array<Document> | undefined | null
  ): LangModelRunIterator {
    const new_messages = messages.map((message) => ({
      ...message,
      contents: Array.isArray(message.contents)
        ? message.contents
        : [{ type: "text", text: message.contents } as Part],
    }));
    return super.run(new_messages, tools, docs);
  }
}
