import { FeatureSupportError, WebGPUNotAvailableError } from "./error";
import {
  Instance,
  instantiate,
  createPolyfillWASI,
  detectGPUDevice,
} from "./tvmjs";

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

export async function init(
  cache_contents: Record<string, ArrayBuffer>
): Promise<Instance> {
  // Initialize GPU
  const gpu: GPUDevice = await getGPUDevice();

  // Load wasm
  const wasm = cache_contents["rt.wasm"];
  delete cache_contents["rt.wasm"];
  const inst = await instantiate(wasm, createPolyfillWASI(), undefined);
  inst.initWebGPU(gpu);
  const device = inst.webgpu();

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
    console.log(buffer.byteLength);
    for (const record of entry.records) {
      const bufferPart = buffer.slice(
        record.byteOffset,
        record.byteOffset + record.nbytes
      );
      await inst.addParameter(device, record, bufferPart);
    }
  }

  // Load parameters
  inst.beginScope();
  const vm = inst.detachFromCurrentScope(inst.createVirtualMachine(device));
  const fgetMetadata = vm.getFunction("_metadata");
  const metadataStr = fgetMetadata().toString();
  const metadata = JSON.parse(metadataStr);

  const paramNames: string[] = [];
  metadata.params.forEach((param: any) => {
    paramNames.push(param.name);
  });
  const params = inst.detachFromCurrentScope(
    inst.getParamsFromCacheByName(paramNames)
  );
  inst.endScope();
  console.log(params);

  // 4. Read in compilation configurations from metadata
  const prefillChunkSize = metadata.prefill_chunk_size;
  // log.info("Using prefillChunkSize: ", this.prefillChunkSize);
  if (prefillChunkSize <= 0) {
    throw new Error("prefill_chunk_size < 0");
  }

  // // 5. Consolidate KVCache settings: context window, sliding window, attention sink
  // this.slidingWindowSize = config.sliding_window_size;
  // this.contextWindowSize = config.context_window_size;
  // this.attentionSinkSize = config.attention_sink_size;
  // if (this.contextWindowSize !== -1 && this.slidingWindowSize !== -1) {
  //   throw new WindowSizeConfigurationError(
  //     this.contextWindowSize,
  //     this.slidingWindowSize
  //   );
  // } else if (this.slidingWindowSize != -1) {
  //   // Use sliding window and attention sink
  //   // log.info("Using slidingWindowSize: ", this.slidingWindowSize);
  //   if (this.attentionSinkSize >= 0) {
  //     // log.info("Using attentionSinkSize: ", this.attentionSinkSize);
  //   } else {
  //     throw new AttentionSinkSizeError();
  //   }
  // } else if (this.contextWindowSize != -1) {
  //   // Use default kv cache without sliding window
  //   // log.info("Using contextWindowSize: ", this.contextWindowSize);
  // } else {
  //   throw new WindowSizeSpecificationError();
  // }

  // 5. Create cache
  // Load cache functions and instantiate KVCache
  const fclearKVCaches = inst.detachFromCurrentScope(
    inst.getGlobalFunc("vm.builtin.kv_state_clear")
  );
  const fKVCacheAddSequence = inst.detachFromCurrentScope(
    inst.getGlobalFunc("vm.builtin.kv_state_add_sequence")
  );
  const fKVCacheRemoveSequence = inst.detachFromCurrentScope(
    inst.getGlobalFunc("vm.builtin.kv_state_remove_sequence")
  );
  const fKVCacheBeginForward = inst.detachFromCurrentScope(
    inst.getGlobalFunc("vm.builtin.kv_state_begin_forward")
  );
  const fKVCacheEndForward = inst.detachFromCurrentScope(
    inst.getGlobalFunc("vm.builtin.kv_state_end_forward")
  );
  const fKVCacheEnableSlidingWindowForSeq = inst.detachFromCurrentScope(
    inst.getGlobalFunc(
      "vm.builtin.attention_kv_cache_enable_sliding_window_for_seq"
    )
  );

  // Create PagedKVCache; we do not expose KVCache config for now
  const fcreateCache = vm.getFunction("create_tir_paged_kv_cache");
  const defaultPageSize = 16;
  const defaultMaxNumSequence = 1;
  const maxTotalSeqLen =
    this.slidingWindowSize != -1
      ? this.slidingWindowSize
      : this.contextWindowSize;
  this.kvCache = this.tvm.detachFromCurrentScope(
    fcreateCache(
      this.tvm.makeShapeTuple([defaultMaxNumSequence]), // max_num_sequence
      this.tvm.makeShapeTuple([maxTotalSeqLen]), // max_total_sequence_length
      this.tvm.makeShapeTuple([prefillChunkSize]), // prefill_chunk_size
      this.tvm.makeShapeTuple([defaultPageSize]), // page_size, hard coded for now
      this.tvm.makeShapeTuple([this.slidingWindowSize != -1 ? 1 : 0])
    )
  );

  this.filledKVCacheLength = 0;
  inst.endScope();

  return inst;
}

// export function loadParams(inst: Instance) {
//   const cpu_arr = inst.withNewScope(() => {
//     return inst.detachFromCurrentScope(
//       inst.empty(rec.shape, rec.dtype, inst.cpu())
//     );
//   });
// }
