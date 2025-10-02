export declare class LangModel {
  static createLocal(
    modelName: string,
    progressCallback?: ((arg: CacheProgress) => void) | undefined | null
  ): Promise<LangModel>;
  static createStreamApi(
    spec: APISpecification,
    modelName: string,
    apiKey: string
  ): LangModel;
  run(
    messages: Array<Message>,
    tools?: Array<ToolDesc> | undefined | null
  ): LangModelRunIterator;
}

export declare class LangModelRunIterator {
  [Symbol.asyncIterator](): this;
  next(): Promise<LanguageModelIteratorResult>;
}

export declare const enum APISpecification {
  ChatCompletion = "ChatCompletion",
  OpenAI = "OpenAI",
  Gemini = "Gemini",
  Claude = "Claude",
  Responses = "Responses",
  Grok = "Grok",
}

export interface CacheProgress {
  comment: string;
  current: number;
  total: number;
}

export type FinishReason =
  | { type: "Stop" }
  | { type: "Length" }
  | { type: "ToolCall" }
  | { type: "Refusal"; field0: string };

export type Grammar =
  | { type: "Plain" }
  | { type: "JSON" }
  | { type: "JSONSchema"; field0: string }
  | { type: "Regex"; field0: string }
  | { type: "CFG"; field0: string };

export interface InferenceConfig {
  thinkEffort: ThinkEffort;
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  grammar: Grammar;
}

export interface LanguageModelIteratorResult {
  value: MessageOutput;
  done: boolean;
}

export interface Message {
  role: Role;
  id?: string;
  thinking: string;
  contents: Array<Part>;
  toolCalls: Array<Part>;
  signature?: string;
}

export interface MessageDelta {
  role?: Role;
  id?: string;
  thinking: string;
  contents: Array<PartDelta>;
  toolCalls: Array<PartDelta>;
  signature?: string;
}

export interface MessageOutput {
  delta: MessageDelta;
  finishReason?: FinishReason;
}

export type Part =
  | { type: "Text"; text: string }
  | { type: "Function"; id?: string; f: PartFunction }
  | { type: "Value"; value: any }
  | { type: "Image"; image: PartImage };

export type PartDelta =
  | { type: "Text"; text: string }
  | { type: "Function"; id?: string; f: PartDeltaFunction }
  | { type: "Value"; value: any }
  | { type: "Null" };

export type PartDeltaFunction =
  | { type: "Verbatim"; field0: string }
  | { type: "WithStringArgs"; name: string; args: string }
  | { type: "WithParsedArgs"; name: string; args: any };

export interface PartFunction {
  name: string;
  args: any;
}

export type PartImage = {
  type: "Binary";
  h: number;
  w: number;
  c: PartImageColorspace;
  data: Buffer;
};

export declare const enum PartImageColorspace {
  Grayscale = "Grayscale",
  RGB = "RGB",
  RGBA = "RGBA",
}

/** The author of a message (or streaming delta) in a chat. */
export declare const enum Role {
  /** System instructions and constraints provided to the assistant. */
  System = "System",
  /** Content authored by the end user. */
  User = "User",
  /** Content authored by the assistant/model. */
  Assistant = "Assistant",
  /** Outputs produced by external tools/functions */
  Tool = "Tool",
}

export declare const enum ThinkEffort {
  Disable = "Disable",
  Enable = "Enable",
  Low = "Low",
  Medium = "Medium",
  High = "High",
}

export interface ToolDesc {
  name: string;
  description?: string;
  parameters: any;
  returns?: any;
}
