/* tslint:disable */
/* eslint-disable */
/**
 * The `ReadableStreamType` enum.
 *
 * *This API requires the following crate features to be activated: `ReadableStreamType`*
 */
type ReadableStreamType = "bytes";
export interface CacheProgress {
    comment: string;
    current: number;
    total: number;
}

export interface ToolDesc {
    name: string;
    description?: string;
    parameters: Value;
    returns?: Value;
}

export type Bytes = Uint8Array;

export type Value = undefined | boolean | number | number | number | string | Record<string, any> | Value[];

export type APISpecification = "ChatCompletion" | "OpenAI" | "Gemini" | "Claude" | "Responses" | "Grok";

export interface VectorStoreRetrieveResult {
    id: string;
    document: string;
    metadata?: Metadata;
    distance: number;
}

export interface VectorStoreGetResult {
    id: string;
    document: string;
    metadata?: Metadata;
    embedding: Float32Array;
}

export interface VectorStoreAddInput {
    embedding: Float32Array;
    document: string;
    metadata?: Metadata;
}

export type Metadata = Map<string, Value>;

export type Embedding = number[];

export interface KnowledgeRetrieveResult {
    document: string;
    metadata?: Metadata;
}

/**
 * The yielded value from agent.run().
 */
export interface AgentResponse {
    /**
     * The message delta per iteration.
     */
    delta: MessageDelta;
    /**
     * Optional finish reason. If this is Some, the message aggregation is finalized and stored in `aggregated`.
     */
    finish_reason: FinishReason | undefined;
    /**
     * Optional aggregated message.
     */
    aggregated: Message | undefined;
}

export type PartDelta = { type: "text"; text: string } | { type: "function"; id: string | undefined; function: PartDeltaFunction } | { type: "value"; value: Value } | { type: "null" };

export type PartDeltaFunction = { type: "verbatim"; text: string } | { type: "with_string_args"; name: string; arguments: string } | { type: "with_parsed_args"; name: string; arguments: Value };

export type Part = { type: "text"; text: string } | { type: "function"; id: string | undefined; function: PartFunction } | { type: "value"; value: Value } | { type: "image"; image: PartImage };

export type PartImage = { type: "binary"; height: number; width: number; colorspace: PartImageColorspace; data: Bytes };

export type PartImageColorspace = "grayscale" | "rgb" | "rgba";

export interface PartFunction {
    name: string;
    arguments: Value;
}

export interface MessageOutput {
    delta: MessageDelta;
    finish_reason: FinishReason | undefined;
}

export type FinishReason = { type: "stop" } | { type: "length" } | { type: "tool_call" } | { type: "refusal"; reason: string };

export interface MessageDelta {
    role: Role | undefined;
    id: string | undefined;
    thinking: string | undefined;
    contents: PartDelta[];
    tool_calls: PartDelta[];
    signature: string | undefined;
}

export interface Message {
    role: Role;
    contents: Part[];
    id?: string;
    thinking?: string;
    tool_calls?: Part[];
    signature?: string;
}

/**
 * The author of a message (or streaming delta) in a chat.
 */
export type Role = "system" | "user" | "assistant" | "tool";

export interface InferenceConfig {
    thinkEffort?: ThinkEffort;
    temperature?: number;
    topP?: number;
    maxTokens?: number;
    grammar?: Grammar;
}

export type Grammar = { type: "plain" } | { type: "json" } | { type: "jsonschema"; schema: string } | { type: "regex"; regex: string } | { type: "cfg"; cfg: string };

export type ThinkEffort = "disable" | "enable" | "low" | "medium" | "high";

export class Agent {
  free(): void;
  [Symbol.dispose](): void;
  constructor(lm: LangModel, tools: Tool[]);
  run(contents: Part[]): AsyncIterable<AgentResponse>;
}
export class EmbeddingModel {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  static newLocal(modelName: string, progressCallback?: (progress: CacheProgress) => void | null): Promise<EmbeddingModel>;
  infer(text: string): Promise<Float32Array>;
}
export class IntoUnderlyingByteSource {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  start(controller: ReadableByteStreamController): void;
  pull(controller: ReadableByteStreamController): Promise<any>;
  cancel(): void;
  readonly type: ReadableStreamType;
  readonly autoAllocateChunkSize: number;
}
export class IntoUnderlyingSink {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  write(chunk: any): Promise<any>;
  close(): Promise<any>;
  abort(reason: any): Promise<any>;
}
export class IntoUnderlyingSource {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  pull(controller: ReadableStreamDefaultController): Promise<any>;
  cancel(): void;
}
export class Knowledge {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  static newVectorStore(store: VectorStore, embedding_model: EmbeddingModel): Knowledge;
  retrieve(query: string, top_k: number): Promise<KnowledgeRetrieveResult[]>;
}
export class LangModel {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  static create_local(modelName: string, progressCallback?: (progress: CacheProgress) => void | null): Promise<LangModel>;
  static create_stream_api(spec: APISpecification, modelName: string, apiKey: string): Promise<LangModel>;
  infer(msgs: Message[], tools?: ToolDesc[] | null, config?: InferenceConfig | null): AsyncIterable<MessageOutput>;
}
export class MCPClient {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  static streamableHttp(url: string): Promise<MCPClient>;
  readonly tools: Tool[];
}
export class Tool {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  static newFunction(desc: ToolDesc, func: (args: any) => Promise<any>): Tool;
  run(args: any): Promise<any>;
  readonly description: ToolDesc;
}
export class VectorStore {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  static newFaiss(dim: number): Promise<VectorStore>;
  static newChroma(url: string, collectionName?: string | null): Promise<VectorStore>;
  addVector(input: VectorStoreAddInput): Promise<string>;
  addVectors(inputs: VectorStoreAddInput[]): Promise<string[]>;
  getById(id: string): Promise<VectorStoreGetResult | undefined>;
  getByIds(ids: string[]): Promise<VectorStoreGetResult[]>;
  retrieve(query_embedding: Float32Array, top_k: number): Promise<VectorStoreRetrieveResult[]>;
  removeVector(id: string): Promise<void>;
  removeVectors(ids: string[]): Promise<void>;
  clear(): Promise<void>;
  count(): Promise<number>;
}
