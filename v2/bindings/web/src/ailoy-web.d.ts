/* tslint:disable */
/* eslint-disable */
/**
 * The `ReadableStreamType` enum.
 *
 * *This API requires the following crate features to be activated: `ReadableStreamType`*
 */
type ReadableStreamType = "bytes";
export type APISpecification = "ChatCompletion" | "OpenAI" | "Gemini" | "Claude" | "Responses" | "Grok";

export interface ToolDesc {
    name: string;
    description: string | undefined;
    parameters: any;
    returns: any | undefined;
}

export type Bytes = Uint8Array;

export type Value = undefined | boolean | number | number | number | string | Record<string, any> | Value[];

export type PartDelta = { text: string } | { id: string | undefined; function: PartDeltaFunction } | { value: Value } | [];

export type PartDeltaFunction = string | { name: string; arguments: string } | { name: string; arguments: Value };

export type Part = { type: "text"; text: string } | { type: "function"; id: string | undefined; function: PartFunction } | { type: "value"; value: Value } | { type: "image"; image: PartImage };

export type PartImage = { media_type: "image/x-binary"; height: number; width: number; colorspace: PartImageColorspace; data: Bytes };

export type PartImageColorspace = "grayscale" | "rgb" | "rgba";

export interface PartFunction {
    name: string;
    arguments: Value;
}

export interface MessageOutput {
    delta: MessageDelta;
    finishReason: FinishReason | undefined;
}

export type FinishReason = { type: "stop" } | { type: "length" } | { type: "tool_call" } | { type: "refusal"; reason: string };

export interface MessageDelta {
    role: Role | undefined;
    id: string | undefined;
    thinking: string;
    contents: PartDelta[];
    toolCalls: PartDelta[];
    signature: string | undefined;
}

export interface Message {
    role: Role;
    id: string | undefined;
    thinking: string;
    contents: Part[];
    toolCalls: Part[];
    signature: string | undefined;
}

/**
 * The author of a message (or streaming delta) in a chat.
 */
export type Role = "system" | "user" | "assistant" | "tool";

export interface InferenceConfig {
    thinkEffort: ThinkEffort;
    temperature: number | undefined;
    topP: number | undefined;
    maxTokens: number | undefined;
    grammar: Grammar;
}

export type Grammar = { type: "plain" } | { type: "json" } | { type: "jsonschema"; schema: string } | { type: "regex"; regex: string } | { type: "cfg"; cfg: string };

export type ThinkEffort = "disable" | "enable" | "low" | "medium" | "high";

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
