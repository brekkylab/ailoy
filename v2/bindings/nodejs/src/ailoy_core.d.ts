// prettier-ignore
/* @ts-ignore */
export type LanguageModel = LocalLanguageModel | OpenAILanguageModel | GeminiLanguageModel | AnthropicLanguageModel | XAILanguageModel;
/* @ts-ignore */
export type EmbeddingModel = LocalEmbeddingModel;
/* @ts-ignore */
export type VectorStore = FaissVectorStore | ChromaVectorStore;
export declare class Agent {
  constructor(lm: LanguageModel);
  run(parts: Array<JsPart>): AgentRunIterator;
}
export type JsAgent = Agent;

export declare class AgentRunIterator {
  [Symbol.asyncIterator](): this;
  next(): Promise<AgentRunIteratorResult>;
}

export declare class AnthropicLanguageModel {
  constructor(modelName: string, apiKey: string);
  run(messages: Array<JsMessage>): LanguageModelRunIterator;
}
export type JsAnthropicLanguageModel = AnthropicLanguageModel;

export declare class ChromaVectorStore {
  static create(
    chromaUrl: string,
    collectionName: string
  ): Promise<ChromaVectorStore>;
  addVector(input: VectorStoreAddInput): Promise<string>;
  addVectors(inputs: Array<VectorStoreAddInput>): Promise<Array<string>>;
  getById(id: string): Promise<VectorStoreGetResult | null>;
  getByIds(ids: Array<string>): Promise<Array<VectorStoreGetResult>>;
  retrieve(
    queryEmbedding: Embedding,
    topK: number
  ): Promise<Array<VectorStoreRetrieveResult>>;
  batchRetrieve(
    queryEmbeddings: Array<Embedding>,
    topK: number
  ): Promise<Array<Array<VectorStoreRetrieveResult>>>;
  removeVector(id: string): Promise<void>;
  removeVectors(ids: Array<string>): Promise<void>;
  clear(): Promise<void>;
  count(): Promise<number>;
}
export type JsChromaVectorStore = ChromaVectorStore;

export declare class FaissVectorStore {
  static create(dim: number): Promise<FaissVectorStore>;
  addVector(input: VectorStoreAddInput): Promise<string>;
  addVectors(inputs: Array<VectorStoreAddInput>): Promise<Array<string>>;
  getById(id: string): Promise<VectorStoreGetResult | null>;
  getByIds(ids: Array<string>): Promise<Array<VectorStoreGetResult>>;
  retrieve(
    queryEmbedding: Embedding,
    topK: number
  ): Promise<Array<VectorStoreRetrieveResult>>;
  batchRetrieve(
    queryEmbeddings: Array<Embedding>,
    topK: number
  ): Promise<Array<Array<VectorStoreRetrieveResult>>>;
  removeVector(id: string): Promise<void>;
  removeVectors(ids: Array<string>): Promise<void>;
  clear(): Promise<void>;
  count(): Promise<number>;
}
export type JsFaissVectorStore = FaissVectorStore;

export declare class GeminiLanguageModel {
  constructor(modelName: string, apiKey: string);
  run(messages: Array<JsMessage>): LanguageModelRunIterator;
}
export type JsGeminiLanguageModel = GeminiLanguageModel;

export declare class LanguageModelRunIterator {
  [Symbol.asyncIterator](): this;
  next(): Promise<LanguageModelIteratorResult>;
}

export declare class LocalEmbeddingModel {
  static create(
    modelName: string,
    progressCallback?: ((arg: CacheProgress) => void) | undefined | null
  ): Promise<LocalEmbeddingModel>;
  run(message: string): Promise<Embedding>;
}
export type JsLocalEmbeddingModel = LocalEmbeddingModel;

export declare class LocalLanguageModel {
  static create(
    modelName: string,
    progressCallback?: ((arg: CacheProgress) => void) | undefined | null
  ): Promise<LocalLanguageModel>;
  run(messages: Array<JsMessage>): LanguageModelRunIterator;
}
export type JsLocalLanguageModel = LocalLanguageModel;

/** Message /// */
export declare class Message {
  constructor();
  get role(): Role | null;
  set role(role: Role);
  get contents(): Array<Part>;
  set contents(contents: Array<Part>);
  get reasoning(): string;
  set reasoning(reasoning: string);
  get toolCalls(): Array<Part>;
  set toolCalls(toolCalls: Array<Part>);
  get toolCallId(): string | null;
  set toolCallId(toolCallId: string);
  toJSON(): object;
  toString(): string;
}
export type JsMessage = Message;

/** MessageOutput /// */
export declare class MessageOutput {
  get delta(): Message;
  get finishReason(): FinishReason | null;
  toJSON(): object;
  toString(): string;
}
export type JsMessageOutput = MessageOutput;

export declare class OpenAILanguageModel {
  constructor(modelName: string, apiKey: string);
  run(messages: Array<JsMessage>): LanguageModelRunIterator;
}
export type JsOpenAILanguageModel = OpenAILanguageModel;

/** Part /// */
export declare class Part {
  static newText(text: string): Part;
  static newFunction(id: string, name: string, arguments: object): Part;
  static newImageUrl(url: string): Part;
  static newImageData(data: string, mimeType: string): Part;
  get partType(): string;
  get text(): string | null;
  set text(text: string);
  get id(): string | null;
  set id(id: string);
  get name(): string | null;
  set name(name: string);
  get arguments(): object | null;
  set arguments(arguments: object);
  get url(): string | null;
  set url(url: string);
  get data(): string | null;
  set data(data: string);
  get mimeType(): string | null;
  set mimeType(mimeType: string);
  toJSON(): object;
  toString(): string;
}
export type JsPart = Part;

export declare class XAILanguageModel {
  constructor(modelName: string, apiKey: string);
  run(messages: Array<JsMessage>): LanguageModelRunIterator;
}
export type JsXAILanguageModel = XAILanguageModel;

export interface AgentRunIteratorResult {
  value?: JsMessageOutput;
  done: boolean;
}

export interface CacheProgress {
  comment: string;
  current: number;
  total: number;
}

export type Embedding = Array<number>;

export declare const enum FinishReason {
  Stop = "Stop",
  Length = "Length",
  ContentFilter = "ContentFilter",
  ToolCalls = "ToolCalls",
}

export interface LanguageModelIteratorResult {
  value?: JsMessageOutput;
  done: boolean;
}

export type Metadata = Record<string, any> | undefined | null;

/** The author of a message (or streaming delta) in a chat. */
export declare const enum Role {
  /** System instructions and constraints provided to the assistant. */
  System = "System",
  /** Content authored by the end user. */
  User = "User",
  /** Content authored by the assistant/model. */
  Assistant = "Assistant",
  /**
   * Outputs produced by external tools/functions, typically in
   * response to an assistant tool call (and often correlated via `tool_call_id`).
   */
  Tool = "Tool",
}

export interface VectorStoreAddInput {
  embedding: Embedding;
  document: string;
  metadata: Metadata;
}

export interface VectorStoreGetResult {
  id: string;
  document: string;
  metadata: Metadata;
  embedding: Embedding;
}

export interface VectorStoreRetrieveResult {
  id: string;
  document: string;
  metadata: Metadata;
  distance: number;
}
