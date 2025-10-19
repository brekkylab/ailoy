import { EmbeddingExceedContextWindowSizeError } from "./error";
import { Instance, PackedFunc } from "./tvmjs";
import { init as init_tvm_runtime, TVMRuntime } from "./tvm_runtime";

export class TVMEmbeddingModel {
  private rt: TVMRuntime;
  private tvm: Instance;
  private contextWindowSize: number;

  private fPrefill: PackedFunc;

  constructor(rt: TVMRuntime) {
    this.rt = rt;
    this.tvm = this.rt.tvm;

    this.tvm.beginScope();

    this.contextWindowSize = this.rt.metadata.context_window_size || 8192;
    this.fPrefill = this.rt.get_function("prefill");

    this.tvm.endScope();
  }

  async infer(tokens: Uint32Array): Promise<Float32Array | Uint16Array> {
    if (!tokens || tokens.length === 0) {
      throw new Error("Token must not be empty");
    }

    this.tvm.beginScope();

    const inputSize = tokens.length;
    if (inputSize > this.contextWindowSize) {
      throw new EmbeddingExceedContextWindowSizeError(
        this.contextWindowSize,
        inputSize
      );
    }

    const input_cpu = this.tvm.withNewScope(() => {
      return this.tvm.detachFromCurrentScope(
        this.tvm.empty([1, inputSize], "int32", this.tvm.cpu())
      );
    });
    input_cpu.copyFrom(new Int32Array(tokens.map((value) => value)));
    // await this.device.sync();
    const mask_cpu = Array(inputSize).fill(1);
    const input = this.tvm.withNewScope(() => {
      return this.tvm.detachFromCurrentScope(
        this.tvm.empty([1, inputSize], "int32", this.rt.device)
      );
    });
    input.copyFrom(input_cpu);
    const mask = this.tvm.withNewScope(() => {
      return this.tvm.detachFromCurrentScope(
        this.tvm.empty([1, inputSize], "int32", this.rt.device)
      );
    });
    mask.copyFrom(mask_cpu);
    await this.rt.device.sync();

    const logitsCurBatchOnGPU = this.fPrefill(input, mask, this.rt.params);
    await this.rt.device.sync();

    const hidden_size = logitsCurBatchOnGPU.shape[2];
    let logitsCurBatchOnCPU = this.tvm.empty(
      logitsCurBatchOnGPU.shape,
      logitsCurBatchOnGPU.dtype,
      this.tvm.cpu()
    );
    logitsCurBatchOnCPU.copyFrom(logitsCurBatchOnGPU);
    logitsCurBatchOnCPU = logitsCurBatchOnCPU.view([inputSize * hidden_size]);
    await this.rt.device.sync();

    let logitsCurBatchOnCPUArray: Float32Array | Uint16Array;
    if (logitsCurBatchOnCPU.dtype === "float16") {
      logitsCurBatchOnCPUArray = <Uint16Array>logitsCurBatchOnCPU.toArray();
    } else {
      logitsCurBatchOnCPUArray = <Float32Array>logitsCurBatchOnCPU.toArray();
    }

    this.tvm.endScope();

    return logitsCurBatchOnCPUArray.slice(0, hidden_size);
  }
}

export async function init(
  cache_contents: Record<string, ArrayBuffer>
): Promise<TVMEmbeddingModel> {
  const rt = await init_tvm_runtime(cache_contents);
  const rv = new TVMEmbeddingModel(rt);
  return rv;
}
