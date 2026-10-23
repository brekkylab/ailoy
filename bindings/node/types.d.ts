// The shapes of the objects ailoy's JavaScript API takes and gives.
//
// Each is the JSON form of the Rust type of the same name, as `serde` writes it — declared
// here only so that an editor can say what the keys are. Nothing checks an object against
// these at runtime; the Rust side does, when the object arrives.

export type Role = 'system' | 'user' | 'assistant' | 'tool'

/** Anything JSON can hold. */
export type Value = any

export interface TextPart {
  type: 'text'
  text: string
}

export interface PartFunction {
  name: string
  arguments: Value
}

export interface FunctionPart {
  type: 'function'
  id: string
  function: PartFunction
}

export interface ValuePart {
  type: 'value'
  value: Value
}

export interface EmbeddedImage {
  type: 'embedded'
  mime_type: string
  data: Buffer
}

export interface UrlImage {
  type: 'url'
  url: string
}

export interface ImagePart {
  type: 'image'
  image: EmbeddedImage | UrlImage
}

export type Part = TextPart | FunctionPart | ValuePart | ImagePart

export interface Message {
  role: Role
  contents: Array<Part>
  thinking?: string
  tool_calls?: Array<FunctionPart>
  id?: string
  signature?: string
}

/** `{ type: 'stop' }`, `'length'`, `'tool_call'`, or `'refusal'` with a `reason`. */
export interface FinishReason {
  type: 'stop' | 'length' | 'tool_call' | 'refusal'
  reason?: string
}

export interface TokenUsage {
  /** Every token of input the call processed, cached or not. */
  input_tokens: number
  output_tokens: number
  /** The part of `input_tokens` written to a prompt cache. */
  cache_creation_input_tokens?: number
  /** The part of `input_tokens` read from a prompt cache. */
  cache_read_input_tokens?: number
}

/** What `Agent.run` yields: one complete message. */
export interface MessageOutput {
  message: Message
  finish_reason: FinishReason
  usage?: TokenUsage
}

/**
 * A piece of a part: `text`, `function`, `value`, `image` or `null`, with the fields of that
 * part as far as they have arrived.
 */
export interface PartDelta {
  type: 'text' | 'function' | 'value' | 'image' | 'null'
  text?: string
  [field: string]: Value
}

export interface MessageDelta {
  role?: Role
  contents: Array<PartDelta>
  id?: string
  thinking?: string
  tool_calls: Array<PartDelta>
  signature?: string
}

/** What `Agent.runStream` yields. A `finish_reason` marks the end of a message. */
export interface MessageDeltaOutput {
  delta: MessageDelta
  finish_reason: FinishReason | null
  usage: TokenUsage | null
}

export interface ToolDesc {
  name: string
  description?: string
  parameters: Value
  returns?: Value
}

export type ThinkingEffort = 'low' | 'medium' | 'high'

export interface LangModelOptions {
  thinking_effort?: ThinkingEffort
  output_schema?: Value
}

export interface AgentSpec {
  model: string
  instruction?: string
  tools?: Array<ToolDesc>
  model_options?: LangModelOptions
  web_search_engines?: Array<WebSearchEngine>
  skills?: Array<string>
}

export type Backend = 'cli' | 'api' | 'bedrock' | 'openrouter' | 'vllm'

/**
 * A model of your own, as a factory given to `setLangModelProvider` returns it. Its methods are
 * called with `this` as the model. Without `inferStream`, streaming gives `infer`'s output as
 * one delta.
 */
export interface CustomLangModel {
  infer(
    messages: Array<Message>,
    tools: Array<ToolDesc>,
    options: LangModelOptions,
  ): MessageOutput | Promise<MessageOutput>
  inferStream?(
    messages: Array<Message>,
    tools: Array<ToolDesc>,
    options: LangModelOptions,
  ): AsyncIterable<MessageDeltaOutput>
}

export type WebSearchEngine =
  | 'Bing'
  | 'Brave'
  | 'DuckDuckGo'
  | 'Google'
  | 'Mojeek'
  | 'Naver'
  | 'Startpage'
  | 'Yahoo'
  | 'Yandex'
