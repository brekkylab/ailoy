import { Message, MessageOutput, Tool } from "../agent";
import { Runtime } from "../runtime";

export const openAIModelIds = [
  "gpt-5",
  "gpt-5-mini",
  "gpt-5-nano",
  "gpt-5-chat-latest",
  "o4-mini",
  "o3",
  "o3-pro",
  "o3-mini",
  "gpt-4o",
  "gpt-4o-mini",
  "gpt-4.1",
  "gpt-4.1-mini",
  "gpt-4.1-nano",
] as const;

export const geminiModelIds = [
  "gemini-2.5-flash",
  "gemini-2.5-pro",
  "gemini-2.0-flash",
  "gemini-1.5-flash",
  "gemini-1.5-pro",
] as const;

export const claudeModelIds = [
  "claude-sonnet-4-20250514",
  "claude-3-7-sonnet-20250219",
  "claude-3-5-sonnet-20241022",
  "claude-3-5-sonnet-20240620",
  "claude-opus-4-1-20250805",
  "claude-opus-4-20250514",
  "claude-3-opus-20240229",
  "claude-3-5-haiku-20241022",
  "claude-3-haiku-20240307",
] as const;

export const grokModelIds = [
  "grok-4",
  "grok-4-0709",
  "grok-3",
  "grok-3-fast",
  "grok-3-mini",
  "grok-3-mini-fast",
  "grok-2",
  "grok-2-1212",
  "grok-2-vision-1212",
  "grok-2-image-1212",
] as const;

export type APIModelId =
  | (typeof openAIModelIds)[number]
  | (typeof geminiModelIds)[number]
  | (typeof claudeModelIds)[number]
  | (typeof grokModelIds)[number]
  | (string & {});

export type APIModelProvider = "openai" | "gemini" | "claude" | "grok";

export interface APIModelArgs {
  id: APIModelId;
  provider?: APIModelProvider;
  apiKey: string;
}

export class _APIModel {
  id: APIModelId;
  provider: APIModelProvider;
  apiKey: string;
  readonly componentType: string;

  // Internal states required for infer
  #initialized: boolean = false;
  private componentName: string | undefined;
  private runtime: Runtime | undefined;

  constructor(args: APIModelArgs) {
    this.id = args.id;
    this.apiKey = args.apiKey;
    if (args.provider === undefined) {
      if (openAIModelIds.includes(this.id as any)) {
        args.provider = "openai";
      } else if (geminiModelIds.includes(this.id as any)) {
        args.provider = "gemini";
      } else if (claudeModelIds.includes(this.id as any)) {
        args.provider = "claude";
      } else if (grokModelIds.includes(this.id as any)) {
        args.provider = "grok";
      } else {
        throw Error(
          `Failed to infer the model provider based on the model id "${this.id}". Please provide an explicit model provider.`
        );
      }
    }
    this.provider = args.provider;

    this.componentType = args.provider;
  }

  defaultSystemMessage(): string | undefined {
    return undefined;
  }

  async init(args: { runtime: Runtime }) {
    if (this.#initialized) return;

    this.runtime = args.runtime;
    this.componentName = this.runtime.generateUUID();

    const result = await this.runtime.define(
      this.componentType,
      this.componentName,
      {
        model: this.id,
        api_key: this.apiKey,
      }
    );
    if (!result) {
      throw Error(`Failed to define API model component`);
    }

    this.#initialized = true;
  }

  async infer(args: {
    messages: Message[];
    tools: Tool[];
    reasoning?: boolean;
  }): Promise<AsyncIterable<MessageOutput>> {
    if (!this.#initialized) {
      throw Error(`The model is not initialized yet`);
    }
    return this.runtime?.callIterMethod(this.componentName!, "infer", {
      messages: args.messages,
      tools: args.tools.map((tool) => ({
        type: "function",
        function: tool.desc,
      })),
      reasoning: args.reasoning,
    })!;
  }

  async dispose() {
    if (this.#initialized) {
      await this.runtime!.delete(this.componentName!);
      this.runtime = undefined;
      this.componentName = undefined;
    }
  }
}

export function APIModel(args: APIModelArgs): _APIModel {
  return new _APIModel(args);
}
