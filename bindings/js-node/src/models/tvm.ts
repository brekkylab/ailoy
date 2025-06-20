export type TVMModelId =
  | "Qwen/Qwen3-8B"
  | "Qwen/Qwen3-4B"
  | "Qwen/Qwen3-1.7B"
  | "Qwen/Qwen3-0.6B";

export type Quantization = "q4f16_1";

interface TVMModelArgs {
  id: TVMModelId;
  quantization?: Quantization;
  device?: number;
}

export class TVMModel {
  id: TVMModelId;
  quantization: Quantization;
  device: number;
  readonly componentType: string = "tvm_language_model";

  constructor(args: TVMModelArgs) {
    this.id = args.id;
    this.quantization = args.quantization ?? "q4f16_1";
    this.device = args.device ?? 0;
  }

  defaultSystemMessage(): string | undefined {
    if (this.id.startsWith("Qwen")) {
      return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    }
    return undefined;
  }

  toAttrs() {
    return {
      model: this.id,
      quantization: this.quantization,
      device: this.device,
    };
  }
}

export default function (args: TVMModelArgs): TVMModel {
  return new TVMModel(args);
}
