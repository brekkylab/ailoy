import ClaudeModel, { ClaudeModel as _ClaudeModel } from "./anthropic";
import GeminiModel, { GeminiModel as _GeminiModel } from "./google";
import OpenAIModel, { OpenAIModel as _OpenAIModel } from "./openai";
import TVMModel, { TVMModel as _TVMModel } from "./tvm";

export type AiloyModel = _ClaudeModel | _GeminiModel | _OpenAIModel | _TVMModel;
export { ClaudeModel, GeminiModel, OpenAIModel, TVMModel };
