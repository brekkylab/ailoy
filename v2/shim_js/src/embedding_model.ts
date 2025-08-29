import { FeatureSupportError, WebGPUNotAvailableError } from "./error";
import {
  EmbeddingChunkingUnsupportedError,
  EmbeddingExceedContextWindowSizeError,
  EmbeddingInputEmptyError,
  EmbeddingSlidingWindowError,
  MinValueError,
} from "./error";
import {
  createPolyfillWASI,
  detectGPUDevice,
  DLDevice,
  Instance,
  instantiate,
  NDArray,
  PackedFunc,
  Scalar,
  TVMObject,
  VirtualMachine,
} from "./tvmjs";

export async function getGPUDevice(
  requiredFeatures: GPUFeatureName[] = ["shader-f16"]
) {
  const gpuDetectOutput = await detectGPUDevice();
  if (gpuDetectOutput == undefined) {
    throw new WebGPUNotAvailableError();
  }
  for (const feature of requiredFeatures) {
    if (!gpuDetectOutput.device.features.has(feature)) {
      throw new FeatureSupportError(feature);
    }
  }
  return gpuDetectOutput.device;
}

export interface NDArrayCacheEntry {
  name: string;
  shape: Array<number>;
  dtype: string;
  format: "f32-to-bf16" | "raw";
  byteOffset: number;
  nbytes: number;
}

export interface NDArrayShardEntry {
  dataPath: string;
  format: "raw-shard";
  nbytes: number;
  records: Array<NDArrayCacheEntry>;
}

const PAGE_SIZE = 16;

export class EmbeddingModel {
  private tvm: Instance;
  private vm: VirtualMachine;
  private params: TVMObject;
  private contextWindowSize: number;

  private fPrefill: PackedFunc;
  private device: DLDevice;

  constructor(
    tvm: Instance,
    device: DLDevice,
    vm: VirtualMachine,
    params: TVMObject,
    contextWindowSize: number
  ) {
    this.tvm = tvm;
    this.vm = vm;
    this.params = params;

    this.fPrefill = tvm.detachFromCurrentScope(vm.getFunction("prefill"));
    this.contextWindowSize = contextWindowSize;

    this.device = device;
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
        this.tvm.empty([1, inputSize], "int32", this.device)
      );
    });
    input.copyFrom(input_cpu);
    const mask = this.tvm.withNewScope(() => {
      return this.tvm.detachFromCurrentScope(
        this.tvm.empty([1, inputSize], "int32", this.device)
      );
    });
    mask.copyFrom(mask_cpu);
    await this.device.sync();

    const logitsCurBatchOnGPU = this.fPrefill(input, mask, this.params);
    await this.device.sync();

    const hidden_size = logitsCurBatchOnGPU.shape[2];
    let logitsCurBatchOnCPU = this.tvm.empty(
      logitsCurBatchOnGPU.shape,
      logitsCurBatchOnGPU.dtype,
      this.tvm.cpu()
    );
    logitsCurBatchOnCPU.copyFrom(logitsCurBatchOnGPU);
    logitsCurBatchOnCPU = logitsCurBatchOnCPU.view([inputSize * hidden_size]);
    await this.device.sync();

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
): Promise<EmbeddingModel> {
  // Load WASM
  const tvmBytes = cache_contents["rt.wasm"];
  delete cache_contents["rt.wasm"];
  const tvm = await instantiate(tvmBytes, createPolyfillWASI(), undefined);

  // Initialize GPU
  const gpu: GPUDevice = await getGPUDevice();
  tvm.initWebGPU(gpu);
  const device = tvm.webgpu(0);

  tvm.beginScope();

  // Load ndarray cache
  const ndarray_cache = JSON.parse(
    new TextDecoder().decode(cache_contents["ndarray-cache.json"])
  );
  delete cache_contents["ndarray-cache.json"];
  const entries: Array<NDArrayShardEntry> = ndarray_cache.records;

  // Register parameters
  for (const entry of entries) {
    const buffer = cache_contents[entry.dataPath];
    delete cache_contents[entry.dataPath];
    for (const record of entry.records) {
      const bufferPart = buffer.slice(
        record.byteOffset,
        record.byteOffset + record.nbytes
      );
      await tvm.ndarrayCacheUpdateBuffer(device, record, bufferPart);
    }
  }

  // Load VM
  const vm = tvm.detachFromCurrentScope(tvm.createVirtualMachine(device));

  // Load parameters
  const fgetMetadata = vm.getFunction("_metadata");
  const metadataStr = fgetMetadata().toString();
  const metadata = JSON.parse(metadataStr);

  const contextWindowSize = metadata.context_window_size || 8192;

  const paramNames: string[] = [];
  metadata.params.forEach((param: any) => {
    paramNames.push(param.name);
  });
  const params = tvm.detachFromCurrentScope(
    tvm.getParamsFromCacheByName(paramNames)
  );

  // Return
  const rv = new EmbeddingModel(tvm, device, vm, params, contextWindowSize);
  tvm.endScope();
  return rv;
}
