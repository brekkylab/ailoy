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
  device: number;
  readonly componentType: string;

  constructor(args: LocalModelArgs) {
    this.id = args.id;
    this.backend = args.backend ?? "tvm";
    this.quantization = args.quantization ?? "q4f16_1";
    this.device = args.device ?? 0;

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

  toAttrs() {
    if (this.backend === "tvm") {
      return {
        model: this.id,
        quantization: this.quantization,
        device: this.device,
      };
    }
    throw Error(`Unknown local model backend: ${this.backend}`);
  }
}

export function LocalModel(args: LocalModelArgs): _LocalModel {
  return new _LocalModel(args);
}
