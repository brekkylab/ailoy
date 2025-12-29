import { FeatureSupportError, WebGPUNotAvailableError } from "./error";
import {
  createPolyfillWASI,
  detectGPUDevice,
  DLDevice,
  Instance,
  instantiate,
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

export interface TensorCacheEntry {
  name: string;
  shape: Array<number>;
  dtype: string;
  format: "f32-to-bf16" | "raw";
  byteOffset: number;
  nbytes: number;
}

export interface TensorShardEntry {
  dataPath: string;
  format: "raw-shard";
  nbytes: number;
  records: Array<TensorCacheEntry>;
}

export class TVMRuntime {
  private _tvm: Instance;
  private _vm: VirtualMachine;
  private _device: DLDevice;
  private _metadata: any;
  private _params: TVMObject;

  constructor(
    tvm: Instance,
    device: DLDevice,
    vm: VirtualMachine,
    metadata: any,
    params: TVMObject
  ) {
    this._tvm = tvm;
    this._device = device;
    this._vm = vm;
    this._metadata = metadata;
    this._params = params;
  }

  public get tvm(): Instance {
    return this._tvm;
  }

  public get vm(): VirtualMachine {
    return this._vm;
  }

  public get device(): DLDevice {
    return this._device;
  }

  public get metadata(): any {
    return this._metadata;
  }

  public get params(): TVMObject {
    return this._params;
  }

  public get_function(fname: string) {
    this._tvm.beginScope();
    const func = this._tvm.detachFromCurrentScope(this._vm.getFunction(fname));
    this._tvm.endScope();
    return func;
  }
}

export type CacheEntries = Record<
  string,
  { fullPath: string; eagerData?: ArrayBuffer }
>;

/**
 * Helper to read a file from OPFS lazily.
 * First checks if the file is available in eagerFiles (already loaded in Rust).
 * If not, reads from OPFS. This avoids double-reading small files.
 */
async function readOPFSFile(
  filename: string,
  cacheEntries: CacheEntries
): Promise<ArrayBuffer> {
  if (!(filename in cacheEntries)) {
    throw new Error(`${filename} is not in the cache entries`);
  }

  const entry = cacheEntries[filename];

  // Check if file is already loaded eagerly
  if (entry.eagerData) {
    return entry.eagerData;
  }

  // Navigate to OPFS root
  const root = await navigator.storage.getDirectory();

  // Parse path and navigate to parent directory
  const parts = entry.fullPath.split("/").filter((p) => p.length > 0);
  let dirHandle = root;

  for (let i = 0; i < parts.length - 1; i++) {
    dirHandle = await dirHandle.getDirectoryHandle(parts[i]);
  }

  // Get file handle and read
  const fileHandle = await dirHandle.getFileHandle(parts[parts.length - 1]);
  const file = await fileHandle.getFile();
  return await file.arrayBuffer();
}

export async function init(cacheEntries: CacheEntries): Promise<TVMRuntime> {
  // Load runtime WASM
  const rtBytes = await readOPFSFile("rt.wasm", cacheEntries);
  const tvm = await instantiate(rtBytes, createPolyfillWASI(), undefined);

  // Initialize GPU
  const gpu: GPUDevice = await getGPUDevice();
  tvm.initWebGPU(gpu);
  const device = tvm.webgpu(0);

  tvm.beginScope();

  // Load tensor cache JSON
  let tensor_cache_json: ArrayBuffer;
  if ("tensor-cache.json" in cacheEntries) {
    tensor_cache_json = await readOPFSFile("tensor-cache.json", cacheEntries);
  } else if ("ndarray-cache.json" in cacheEntries) {
    // Fallback to ndarray-cache.json for backward compatibility
    tensor_cache_json = await readOPFSFile("ndarray-cache.json", cacheEntries);
  } else {
    throw new Error(
      "Cannot find either tensor-cache.json or ndarray-cache.json in file entries"
    );
  }

  const tensor_cache = JSON.parse(new TextDecoder().decode(tensor_cache_json));

  // Load parameters with pipelined parallelism (4 concurrent operations)
  const MAX_CONCURRENT = 4;

  const processShard = async (entry: TensorShardEntry) => {
    const buffer = await readOPFSFile(entry.dataPath, cacheEntries);

    for (const record of entry.records) {
      const bufferPart = buffer.slice(
        record.byteOffset,
        record.byteOffset + record.nbytes
      );
      await tvm.tensorCacheUpdateBuffer(device, record, bufferPart);
    }
    // buffer can be GC'd after this function returns
  };

  const pending = new Set<Promise<void>>();

  // Process shards with a sliding window of concurrency,
  // up to MAX_CONCURRENT concurrent operations
  for (const entry of tensor_cache.records as Array<TensorShardEntry>) {
    // Start processing this shard
    const promise = processShard(entry).then(() => {
      // Remove this promise from pending set when it completes
      pending.delete(promise);
    });
    pending.add(promise);

    // If we've reached max concurrency, wait for at least one to complete
    if (pending.size >= MAX_CONCURRENT) {
      await Promise.race(pending);
    }
  }

  // Wait for all remaining operations to complete
  await Promise.all(Array.from(pending));

  // Load VM
  const vm = tvm.detachFromCurrentScope(tvm.createVirtualMachine(device));

  // Load parameters
  const fgetMetadata = vm.getFunction("_metadata");
  const metadataStr = fgetMetadata().toString();
  const metadata = JSON.parse(metadataStr);

  const paramNames: string[] = [];
  metadata.params.forEach((param: any) => {
    paramNames.push(param.name);
  });
  const params = tvm.detachFromCurrentScope(
    tvm.getParamsFromCacheByName(paramNames)
  );

  // Return
  const rv = new TVMRuntime(tvm, device, vm, metadata, params);
  tvm.endScope();
  return rv;
}
