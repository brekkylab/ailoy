import util from "util";

import { Message, MessageOutput, Part, Role } from "./ailoy_core";

// Add custom inspect symbol to Part
if (typeof Part !== "undefined" && Part.prototype) {
  (Part.prototype as any)[Symbol.for("nodejs.util.inspect.custom")] = function (
    _depth: number,
    _opts: any
  ) {
    return `Part ${util.inspect(this.toJSON())}`;
  };
}

// Add custom inspect symbol to Message
if (typeof Message !== "undefined" && Message.prototype) {
  (Message.prototype as any)[Symbol.for("nodejs.util.inspect.custom")] =
    function (_depth: number, _opts: any) {
      return `Message ${util.inspect(this.toJSON())}`;
    };
}

// Add custom inspect symbol to MessageOutput
if (typeof MessageOutput !== "undefined" && MessageOutput.prototype) {
  (MessageOutput.prototype as any)[Symbol.for("nodejs.util.inspect.custom")] =
    function (_depth: number, _opts: any) {
      return `MessageOutput ${util.inspect(this.toJSON())}`;
    };
}

export { Message, MessageOutput, Part, Role };
export {
  Agent,
  AgentRunIterator,
  AnthropicLanguageModel,
  ChromaVectorStore,
  FaissVectorStore,
  GeminiLanguageModel,
  LanguageModelRunIterator,
  LocalEmbeddingModel,
  LocalLanguageModel,
  OpenAILanguageModel,
  XAILanguageModel,
} from "./ailoy_core";
export type {
  AgentRunIteratorResult,
  CacheProgress,
  Embedding,
  EmbeddingModel,
  LanguageModel,
  LanguageModelIteratorResult,
  VectorStore,
  VectorStoreAddInput,
  VectorStoreGetResult,
  VectorStoreRetrieveResult,
} from "./ailoy_core";
