export type LocalModelId =
  | "Qwen/Qwen3-8B"
  | "Qwen/Qwen3-4B"
  | "Qwen/Qwen3-1.7B"
  | "Qwen/Qwen3-0.6B";
export type LocalModelProvider = "tvm";
export type Quantization = "q4f16_1";

interface LocalModelArgs {
  id: LocalModelId;
  provider?: LocalModelProvider;
  quantization?: Quantization;
  device?: number;
}

export class LocalModel {
  id: LocalModelId;
  provider: LocalModelProvider;
  quantization: Quantization;
  device: number;
  readonly componentType: string;

  constructor(args: LocalModelArgs) {
    this.id = args.id;
    this.provider = args.provider ?? "tvm";
    this.quantization = args.quantization ?? "q4f16_1";
    this.device = args.device ?? 0;

    if (this.provider === "tvm") {
      this.componentType = "tvm_language_model";
    } else {
      throw Error(`Unknown local model provider: ${this.provider}`);
    }
  }

  defaultSystemMessage(): string | undefined {
    if (this.id.startsWith("Qwen")) {
      return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    }
    return undefined;
  }

  toAttrs() {
    if (this.provider === "tvm") {
      return {
        model: this.id,
        quantization: this.quantization,
        device: this.device,
      };
    }
    throw Error(`Unknown local model provider: ${this.provider}`);
  }
}

export default function (args: LocalModelArgs): LocalModel {
  return new LocalModel(args);
}
