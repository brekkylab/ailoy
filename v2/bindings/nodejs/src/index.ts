import {
  Message as Message,
  Part,
  Role,
  MessageOutput,
  CacheProgress,
  LanguageModelRunIterator,
  LocalLanguageModel,
  OpenAILanguageModel,
  GeminiLanguageModel,
  AnthropicLanguageModel,
  XAILanguageModel,
} from "./ailoy_core";
import util from "util";

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

export {
  Message,
  MessageOutput,
  Part,
  Role,
  CacheProgress,
  LanguageModelRunIterator,
  LocalLanguageModel,
  OpenAILanguageModel,
  GeminiLanguageModel,
  AnthropicLanguageModel,
  XAILanguageModel,
};
