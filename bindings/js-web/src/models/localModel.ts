import { Message, MessageOutput, Tool } from "../agent";
import { ChatManager } from "../llm/chat_manager";
import { GenerationConfig } from "../llm/config";
import { Engine } from "../llm/engine";
import { Tokenizer } from "../llm/tokenizer";
import { Runtime } from "../runtime";

export type LocalModelId =
  | "Qwen/Qwen3-0.6B"
  | "Qwen/Qwen3-1.7B"
  | "Qwen/Qwen3-4B"
  | "Qwen/Qwen3-8B"
  | "Qwen/Qwen3-14B"
  | "Qwen/Qwen3-32B"
  | "Qwen/Qwen3-30B-A3B";
export type LocalModelBackend = "tvm";
export type Quantization = "q4f16_1";

export interface LocalModelArgs {
  id: LocalModelId;
  backend?: LocalModelBackend;
  quantization?: Quantization;
  device?: number;
}

export class _LocalModel {
  id: LocalModelId;
  backend: LocalModelBackend;
  quantization: Quantization;
  readonly componentType: string;

  // Internal states required for infer
  #initialized: boolean = false;
  private engine: Engine | undefined;
  private chatManager: ChatManager | undefined;
  private genConfig: GenerationConfig | undefined;

  constructor(args: LocalModelArgs) {
    this.id = args.id;
    this.backend = args.backend ?? "tvm";
    this.quantization = args.quantization ?? "q4f16_1";

    if (this.backend === "tvm") {
      this.componentType = "tvm_language_model";
    } else {
      throw Error(`Unknown local model backend: ${this.backend}`);
    }
  }

  defaultSystemMessage(): string | undefined {
    if (this.id.startsWith("Qwen")) {
      return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    }
    return undefined;
  }

  async init(args: { runtime: Runtime; genConfig?: GenerationConfig }) {
    if (this.#initialized) return;

    const { results }: { results: Array<{ model_id: string }> } =
      await args.runtime.call("list_local_models");

    if (!results.some(({ model_id }) => model_id === this.id)) {
      await args.runtime.call("download_model", {
        model_id: this.id,
        quantization: this.quantization,
        device: "webgpu",
      });
    }

    const tokenizer = new Tokenizer(args.runtime, this.id, this.quantization);
    await tokenizer.init();

    this.chatManager = new ChatManager(
      args.runtime,
      this.id,
      this.quantization
    );
    await this.chatManager.init();

    this.engine = new Engine(this.id, tokenizer);
    await this.engine.loadModel();

    this.genConfig = args.genConfig;

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
    const input = await this.chatManager!.applyChatTemplate(
      args.messages,
      args.tools.map((tool) => ({ type: "function", function: tool.desc })),
      args.reasoning
    );
    return (await this.engine!.inferLM(input, this.genConfig ?? {}))!;
  }

  async dispose() {
    if (this.#initialized) {
      await this.engine!.dispose();
      this.engine = undefined;

      await this.chatManager!.dispose();
      this.chatManager = undefined;

      this.genConfig = undefined;
    }
  }
}

export function LocalModel(args: LocalModelArgs): _LocalModel {
  return new _LocalModel(args);
}
