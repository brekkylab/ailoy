import { FeatureSupportError, WebGPUNotAvailableError } from "./error";
import {
  createPolyfillWASI,
  detectGPUDevice,
  DLDevice,
  Instance,
  instantiate,
  PackedFunc,
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

export async function init(
  cache_contents: Record<string, ArrayBuffer>
): Promise<TVMRuntime> {
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
