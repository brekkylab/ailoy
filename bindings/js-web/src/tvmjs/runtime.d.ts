/// <reference types="@webgpu/types" />
/**
 * TVM JS Wasm Runtime library.
 */
import { Pointer, PtrOffset } from "./ctypes";
import { Disposable } from "./types";
import { Memory, CachedCallStack } from "./memory";
import { Environment } from "./environment";
import { WebGPUContext } from "./webgpu";
/**
 * Type for PackedFunc in the TVMRuntime.
 */
export type PackedFunc = ((...args: any) => any) & Disposable & {
    _tvmPackedCell: PackedFuncCell;
};
/**
 * Type for AyncPackedFunc in TVMRuntime
 * possibly may contain stack unwinding through Asynctify
 */
export type AsyncPackedFunc = ((...args: any) => Promise<any>) & Disposable & {
    _tvmPackedCell: PackedFuncCell;
};
/**
 * @internal
 * FFI Library wrapper, maintains most runtime states.
 */
declare class FFILibrary implements Disposable {
    wasm32: boolean;
    memory: Memory;
    exports: Record<string, Function>;
    webGPUContext?: WebGPUContext;
    private wasmInstance;
    private recycledCallStacks;
    constructor(wasmInstance: WebAssembly.Instance, imports: Record<string, any>);
    dispose(): void;
    sizeofPtr(): number;
    checkCall(code: number): void;
    getOrAllocCallStack(): CachedCallStack;
    recycleCallStack(callstack: CachedCallStack): void;
    private validateInstance;
    private checkExports;
    private detectWasmMemory;
}
/**
 * @internal
 * Manages extra runtime context for the runtime.
 */
declare class RuntimeContext implements Disposable {
    functionListGlobalNamesFunctor: PackedFunc;
    moduleGetFunction: PackedFunc;
    moduleImport: PackedFunc;
    ndarrayEmpty: PackedFunc;
    ndarrayCopyFromTo: PackedFunc;
    ndarrayCopyFromJSBytes: PackedFunc;
    ndarrayCopyToJSBytes: PackedFunc;
    arrayGetItem: PackedFunc;
    arrayGetSize: PackedFunc;
    arrayMake: PackedFunc;
    arrayConcat: PackedFunc;
    getSysLib: PackedFunc;
    arrayCacheGet: PackedFunc;
    arrayCacheUpdate: PackedFunc;
    arrayCacheRemove: PackedFunc;
    arrayCacheClear: PackedFunc;
    arrayDecodeStorage: PackedFunc;
    paramModuleFromCache: PackedFunc;
    paramModuleFromCacheByName: PackedFunc;
    makeShapeTuple: PackedFunc;
    ndarrayCreateView: PackedFunc;
    sampleTopPFromLogits: PackedFunc;
    sampleTopPFromProb: PackedFunc;
    applyRepetitionPenalty: PackedFunc;
    applyPresenceAndFrequencyPenalty: PackedFunc;
    applySoftmaxWithTemperature: PackedFunc;
    concatEmbeddings: PackedFunc | undefined;
    bool: PackedFunc;
    private autoDisposeScope;
    constructor(getGlobalFunc: (name: string) => PackedFunc);
    dispose(): void;
    beginScope(): void;
    endScope(): void;
    /**
     * Track object for dispose in current scope.
     *
     * @param obj The object to be tracked.
     * @returns the same object.
     * @note This function only needs to be called for raw system C API values.
     *       The return value of PackedFunc will be automatically tracked.
     */
    attachToCurrentScope<T extends Disposable>(obj: T): T;
    moveToParentScope<T extends Disposable>(obj: T): T;
    detachFromCurrentScope<T extends Disposable>(obj: T): T;
}
/**
 * A typed scalar constant used to represent a typed number
 * argument to PackedFunc calls.
 */
export declare class Scalar {
    /** The value. */
    value: number;
    /** The data type of the scalar. */
    dtype: string;
    constructor(value: number, dtype: string);
}
/**
 * Represent a runtime context where a NDArray can reside.
 */
export declare class DLDevice {
    /** The device type code of the device. */
    deviceType: number;
    /** The device index. */
    deviceId: number;
    private lib;
    constructor(deviceType: number | string, deviceId: number, lib: FFILibrary);
    /**
     * Synchronize the device
     */
    sync(): Promise<void>;
    toString(): string;
}
/**
 * The data type code in DLDataType
 */
export declare enum DLDataTypeCode {
    Int = 0,
    UInt = 1,
    Float = 2,
    OpaqueHandle = 3
}
/**
 * Runtime data type of NDArray.
 */
export declare class DLDataType {
    /** The type code */
    code: number;
    /** Number of bits in the data type. */
    bits: number;
    /** Number of vector lanes. */
    lanes: number;
    constructor(code: number, bits: number, lanes: number);
    toString(): string;
    numStorageBytes(): number;
}
/**
 * Generic object base
 */
export declare class TVMObject implements Disposable {
    protected handle: Pointer;
    protected lib: FFILibrary;
    protected ctx: RuntimeContext;
    constructor(handle: Pointer, lib: FFILibrary, ctx: RuntimeContext);
    dispose(): void;
    /**
     * Get handle of module, check it is not null.
     *
     * @param requireNotNull require handle is not null.
     * @returns The handle.
     */
    getHandle(requireNotNull?: boolean): Pointer;
    /** get the type index of the object */
    typeIndex(): number;
    /** get the type key of the object */
    typeKey(): string;
}
/**
 * Cell holds the PackedFunc object.
 */
declare class PackedFuncCell extends TVMObject {
    constructor(handle: Pointer, lib: FFILibrary, ctx: RuntimeContext);
}
/**
 * n-dimnesional array.
 */
export declare class NDArray extends TVMObject {
    /** Number of dimensions. */
    ndim: number;
    /** Data type of the array. */
    dtype: string;
    /** Shape of the array. */
    shape: Array<number>;
    /** Device of the array. */
    device: DLDevice;
    /** Whether it is a temporary view that can become invalid after the call. */
    isView: boolean;
    private byteOffset;
    private dltensor;
    private dataPtr;
    private dlDataType;
    constructor(handle: Pointer, lib: FFILibrary, ctx: RuntimeContext, isView: boolean);
    /**
     * Create a view of the array.
     * @param shape The shape of the view.
     * @param dtype The data type of the new array.
     * @returns The new sliced ndarray.
     */
    view(shape: Array<number>, dtype?: string): NDArray;
    /**
     * Get dataPtr of NDarray
     *
     * @returns The handle.
     */
    getDataPtr(): Pointer;
    /**
     * Copy data from another NDArray or javascript array.
     * The number of elements must match.
     *
     * @param data The source data array.
     * @returns this
     */
    copyFrom(data: NDArray | Array<number> | Float32Array | Float64Array | Int32Array | Int8Array | Uint8Array | Uint8ClampedArray): this;
    /**
     * Copy data from raw bytes.
     * @param data Uint8Array of bytes.
     * @returns this
     */
    copyFromRawBytes(data: Uint8Array): this;
    /**
     * Return a copied Uint8Array of the raw bytes in the NDArray.
     * @returns The result array.
     */
    toRawBytes(): Uint8Array;
    /**
     * Return a TypedArray copy of the NDArray, the specific type depends on
     * the dtype of the NDArray.
     * @returns The result array.
     */
    toArray(): Float32Array | Float64Array | Int32Array | Int8Array | Uint8Array;
    private getDLTensorFromArrayHandle;
}
/**
 * Runtime Module.
 */
export declare class Module extends TVMObject {
    constructor(handle: Pointer, lib: FFILibrary, ctx: RuntimeContext);
    /**
     * Get a function in the module.
     * @param name The name of the function.
     * @param queryImports Whether to also query imports
     * @returns The result function.
     */
    getFunction(name: string, queryImports?: boolean): PackedFunc;
    /**
     * Import another module into the current runtime module.
     * @param mod The module to be imported.
     */
    importModule(mod: Module): void;
}
/** Objectconstructor */
type FObjectConstructor = (handle: Pointer, lib: FFILibrary, ctx: RuntimeContext) => TVMObject;
/** All possible object types. */
type TVMObjectBase = TVMObject | PackedFunc;
/** Runtime array object. */
export declare class TVMArray extends TVMObject {
    constructor(handle: Pointer, lib: FFILibrary, ctx: RuntimeContext);
    /**
     * @returns the size of the array.
     */
    size(): number;
    /**
     * Get index-th element of the array
     * @param index the array index.
     * @returns The element.
     */
    get(index: number): TVMObjectBase;
}
export declare enum VMAllocatorKind {
    NAIVE_ALLOCATOR = 1,
    POOLED_ALLOCATOR = 2
}
/**
 *  VirtualMachine Executor.
 *
 *  This is a thin wrapper of the underlying TVM module.
 *  you can also directly call set_input, run, and get_output
 *  of underlying module functions
 */
export declare class VirtualMachine implements Disposable {
    private mod;
    /**
     * Constructor
     * @param mod The underlying module, need to be detached.
     * @param device The main device ro run VM on.
     */
    constructor(mod: Module, device: DLDevice);
    dispose(): void;
    /**
     * Get a function in the VM module.
     * @param name The name of the function.
     * @returns The result function.
     */
    getFunction(name: string): PackedFunc;
    /**
     * Get the internal module.
     */
    getInternalModule(): Module;
}
export interface InitProgressReport {
    progress: number;
    timeElapsed: number;
    text: string;
}
export type InitProgressCallback = (report: InitProgressReport) => void;
/**
 * TVM runtime instance.
 *
 * All objects(NDArray, Module, PackedFunc) returned by TVM runtim function call
 * and PackedFunc instance are tracked through a scope mechanism that will get
 * auto-released when we call EndScope.
 *
 * This is necessarily to be able to release the underlying WASM and WebGPU memory that
 * are not tracked through JS native garbage collection mechanism.
 *
 * This does mean that we have to get familar with the following functions:
 * - {@link beginScope}
 * - {@link endScope}
 * - {@link withNewScope}
 * - {@link attachToCurrentScope}
 * - {@link detachFromCurrentScope}
 */
export declare class Instance implements Disposable {
    memory: Memory;
    exports: Record<string, Function>;
    cacheMetadata: Record<string, any>;
    private lib;
    private env;
    private objFactory;
    private ctx;
    private asyncifyHandler;
    private initProgressCallback;
    private rng;
    private deviceLostIsError;
    /**
     * Internal function(registered by the runtime)
     */
    private wasmCreateLibraryModule?;
    /**
     * Constructor
     *
     * importObject can also be a {@link LibraryProvider} object,
     * a WASI object, or an object containing wasmLibraryProvider field.
     *
     * @param wasmModule The input module or instance.
     * @param importObject The imports to initialize the wasmInstance if it is not provided.
     * @param wasmInstance Additional wasm instance argument for deferred construction.
     * @param env Directly specified environment module.
     *
     * @see Please use the async version {@link instantiate} when targeting browsers.
     */
    constructor(wasmModule: WebAssembly.Module, importObject?: Record<string, any>, wasmInstance?: WebAssembly.Instance, env?: Environment);
    /**
     * Benchmark stable execution of the run function.
     *
     * @params run The run function
     * @params dev The device to sync during each run.
     * @number The number of times to compute the average.
     * @repeat The number of times to repeat the run.
     */
    benchmark(run: () => void, dev: DLDevice, number?: number, repeat?: number): Promise<number[]>;
    /**
     * Check whether we enabled asyncify mode
     * @returns The asynctify mode toggle
     */
    asyncifyEnabled(): boolean;
    dispose(): void;
    /**
     * Obtain the runtime information in readable format.
     */
    runtimeStatsText(): string;
    /**
     * Begin a new scope for tracking object disposal.
     */
    beginScope(): void;
    /**
     * End a scope and release all created TVM objects
     * under the current scope.
     *
     * Exception: one can call {@link moveToParentScope} to move
     * a value to parent scope.
     */
    endScope(): void;
    /**
     * Perform action under a new scope.
     *
     * @param action The action function.
     * @returns The result value.
     *
     * @note For action to return a valid value,
     *       we will need to call {@link moveToParentScope}
     *       for the objects that are created in the scope.
     */
    withNewScope<T>(action: () => T): T;
    /**
     * Attach a detached obj to the auto-release pool of the current scope.
     *
     * @param obj The input obj.
     * @note Normally user do not need to call this function explicitly, as
     *       all library call return values are explicitly attached to
     *       the current scope. You only need to do so when you call
     *       {@link detachFromCurrentScope} to create a detached object.
     */
    attachToCurrentScope<T extends Disposable>(obj: T): T;
    /**
     * Move obj's attachment to the parent scope.
     *
     * This function is useful to make sure objects are still
     * alive when exit the current scope.
     *
     * @param obj The object to be moved.
     * @returns The input obj.
     */
    moveToParentScope<T extends Disposable>(obj: T): T;
    /**
     * Detach the object from the current scope
     * so it won't be released via auto-release during endscope.
     *
     * User needs to either explicitly call obj.dispose(), or
     * {@link attachToCurrentScope} to re-attach to the current scope.
     *
     * This function can be used to return values to the parent scope.
     * @param obj The object.
     */
    detachFromCurrentScope<T extends Disposable>(obj: T): T;
    /**
     * Get system-wide library module in the wasm.
     * System lib is a global module that contains self register functions in startup.
     * @returns The system library module.
     */
    systemLib(): Module;
    /**
     * List all the global function names registered in the runtime.
     * @returns The name list.
     */
    listGlobalFuncNames(): Array<string>;
    /**
     * Register function to be global function in tvm runtime.
     * @param name The name of the function.
     * @param f function to be registered.
     * @param override Whether overwrite function in existing registry.
     */
    registerFunc(name: string, func: PackedFunc | Function, override?: boolean): void;
    /**
     * Get global PackedFunc from the runtime.
     * @param name The name of the function.
     * @param autoAttachToScope Whether to track it via autoDispose
     * @returns The result function.
     */
    getGlobalFunc(name: string): PackedFunc;
    private getGlobalFuncInternal;
    /**
     * Check if func is PackedFunc.
     *
     * @param func The input.
     * @returns The check result.
     */
    isPackedFunc(func: unknown): boolean;
    /**
     * Convert func to PackedFunc
     *
     * @param func Input function.
     * @returns The converted function.
     */
    toPackedFunc(func: Function): PackedFunc;
    private toPackedFuncInternal;
    /**
    * Setup a virtual machine module with given device.
    *
    * @param dev DLDevice the device.
    * @returns The created virtual machime.
    */
    createVirtualMachine(dev: DLDevice): VirtualMachine;
    /**
     * Register a call back for fetch progress.
    *
     * @param cb the fetch progress callback.
     */
    registerInitProgressCallback(cb: InitProgressCallback): void;
    /**
     * Get parameters in the form of prefix_i
     *
     * @param prefix The parameter prefix.
     * @param numParams  Number of parameters.
     * @returns
     */
    getParamsFromCache(prefix: string, numParams: number): TVMObject;
    /**
     * Get parameters based on parameter names provided
     *
     * @param paramNames Names of the parameters.
     * @returns Parameters read.
     */
    getParamsFromCacheByName(paramNames: Array<string>): TVMObject;
    /**
     * Get NDArray from cache.
     * @param name  The name of array.
     * @returns  The result.
     */
    ndarrayCacheGet(name: string): NDArray | undefined;
    /**
     * Get NDArray from cache.
     * @param name  The name of array.
     * @returns  The result.
     */
    ndarrayCacheRemove(name: string): NDArray | undefined;
    /**
     * Update the ndarray cache.
     * @param name The name of the array.
     * @param arr The content.
     */
    ndarrayCacheUpdate(name: string, arr: NDArray, override?: boolean): void;
    /**
     * Update the ndarray cache.
     * @param name The name of the array.
     * @param arr The content.
     */
    ndarrayCacheClear(): void;
    /**
     * Given cacheUrl, search up items to fetch based on cacheUrl/ndarray-cache.json
     *
     * @param ndarrayCacheUrl The cache url.
     * @param device The device to be fetched to.
     * @param cacheScope The scope identifier of the cache
     * @param cacheType The type of the cache: "cache" or "indexedDB"
     * @param signal An optional AbortSignal to abort the fetch
     * @returns The meta data
     */
    fetchNDArrayCache(ndarrayCacheUrl: string, device: DLDevice, cacheScope?: string, cacheType?: string, signal?: AbortSignal): Promise<any>;
    /**
     * Fetch list of NDArray into the NDArrayCache.
     *
     * @param ndarrayCacheUrl The cache url.
     * @param list The list of array data.
     * @param device The device to store the data to.
     * @param artifactCache The artifact cache
     * @param signal An optional AbortSignal to abort the fetch
     */
    private fetchNDArrayCacheInternal;
    /**
     * Create a new {@link Scalar} that can be passed to a PackedFunc.
     * @param value The number value.
     * @param dtype The dtype string.
     * @returns The created scalar.
     */
    scalar(value: number, dtype: string): Scalar;
    /**
     * Create a new {@link DLDevice}
     * @param deviceType The device type.
     * @param deviceId The device index.
     * @returns The created device.
     */
    device(deviceType: number | string, deviceId?: number): DLDevice;
    /**
     * Create a new cpu {@link DLDevice}
     * @param deviceId The device index.
     */
    cpu(deviceId?: number): DLDevice;
    /**
     * Create a new webgpu {@link DLDevice}
     * @param deviceId The device index.
     */
    webgpu(deviceId?: number): DLDevice;
    /**
     * Create an empty {@link NDArray} with given shape and dtype.
     *
     * @param shape The shape of the array.
     * @param dtype The data type of the array.
     * @param dev The device of the ndarray.
     * @returns The created ndarray.
     */
    empty(shape: Array<number> | number, dtype?: string | DLDataType, dev?: DLDevice): NDArray;
    /**
     * Create am uniform {@link NDArray} with given shape.
     *
     * @param shape The shape of the array.
     * @param low The low value.
     * @param high The high value.
     * @param dev The device of the ndarray.
     * @returns The created ndarray.
     */
    uniform(shape: Array<number>, low: number, high: number, dev: DLDevice): NDArray;
    /**
     * Set the seed of the internal LinearCongruentialGenerator.
     */
    setSeed(seed: number): void;
    /**
     * Sample index via top-p sampling.
     *
     * @param logits The input logits before normalization.
     * @param temperature  The temperature factor, will take argmax if temperature = 0.0
     * @param top_p The top_p
     * @returns The sampled index.
     */
    sampleTopPFromLogits(logits: NDArray, temperature: number, top_p: number): number;
    /**
     * Sample index via top-p sampling.
     *
     * @param prob The distribution, i.e. logits after `applySoftmaxWithTemperature()` is performed.
     * @param top_p The top_p
     * @returns The sampled index.
     */
    sampleTopPFromProb(prob: NDArray, top_p: number): number;
    /**
     * Apply repetition penalty to the logits.
     * @param logits The input logits before penalty.
     * @param token_ids The appeared token ids.
     * @param penalty The penalty factor.
     */
    applyRepetitionPenalty(logits: NDArray, token_ids: NDArray, penalty: number): any;
    /**
     * Apply presence and frequency penalty. This is an inplace operation.
     * @param logits The input logits before penalty.
     * @param token_ids The appeared token ids.
     * @param token_freqs The number of times each token has appeared since last PrefillStep.
     * token_freqs[i] is the frequency of token_ids[i], for all i. And all token_freqs should be >= 1.
     * @param presence_penalty The penalty factor.
     * @param frequency_penalty The penalty factor.
     */
    applyPresenceAndFrequencyPenalty(logits: NDArray, token_ids: NDArray, token_freqs: NDArray, presence_penalty: number, frequency_penalty: number): any;
    /**
     * Apply softmax with temperature to the logits.
     * @param logits The input logits before softmax w/ temperature.
     * @param temperature The temperature factor.
     */
    applySoftmaxWithTemperature(logits: NDArray, temperature: number): any;
    /**
     * Bind canvas to the current WebGPU context
     * @param canvas The canvas.
     */
    bindCanvas(canvas: HTMLCanvasElement): void;
    /**
     * Show image in canvas.
     *
     * @param dataRGBA Image array in height x width uint32 NDArray RGBA format on GPU.
     */
    showImage(dataRGBA: NDArray): void;
    /**
     * Clear canvas
     */
    clearCanvas(): void;
    /**
     * Create an tuple {@link TVMArray} input array.
     *
     * The input array can be passed to tvm runtime function
     * and needs to b explicitly disposed.
     *
     * @param inputs The input array
     * @returns The result array.
     */
    makeTVMArray(inputs: Array<any>): TVMArray;
    /**
     * Join a sequence of NDArrays that represent embeddings.
     * @param inputs A list of embeddings in NDArrays, each array i has shape (m_i, hidden_size).
     * @returns An NDArray of shape (\sum_{i} {m}, hidden_size)
     */
    concatEmbeddings(embeddings: Array<NDArray>): NDArray;
    /**
     * Create a shape tuple to pass to runtime.
     * @param shape The shape .
     * @returns The created shape tuple.
     */
    makeShapeTuple(shape: Array<number>): TVMObject;
    /**
     * Get type index from type key.
     * @param typeKey The type key.
     * @returns The corresponding type index.
     */
    typeKey2Index(typeKey: string): number;
    /**
     * Register an object constructor.
     * @param typeKey The name of the function.
     * @param func Function to be registered.
     * @param override Whether overwrite function in existing registry.
     */
    registerObjectConstructor(typeKey: string, func: FObjectConstructor, override?: boolean): void;
    /**
     * Wrap a function obtained from tvm runtime as AsyncPackedFunc
     * through the asyncify mechanism
     *
     * You only need to call it if the function may contain callback into async
     * JS function via asynctify. A common one can be GPU synchronize.
     *
     * It is always safe to wrap any function as Asynctify, however you do need
     * to make sure you use await when calling the funciton.
     *
     * @param func The PackedFunc.
     * @returns The wrapped AsyncPackedFunc
     */
    wrapAsyncifyPackedFunc(func: PackedFunc): AsyncPackedFunc;
    /**
     * Register async function as asynctify callable in global environment.
     *
     * @param name The name of the function.
     * @param func function to be registered.
     * @param override Whether overwrite function in existing registry.
     *
     * @note This function is handled via asynctify mechanism
     * The wasm needs to be compiled with Asynctify
     */
    registerAsyncifyFunc(name: string, func: (...args: Array<any>) => Promise<any>, override?: boolean): void;
    /**
     * Register an asyncfunction to be global function in the server.
     *
     * @param name The name of the function.
     * @param func function to be registered.
     * @param override Whether overwrite function in existing registry.
     *
     * @note The async function will only be used for serving remote calls in the rpc
     * These functions contains explicit continuation
     */
    registerAsyncServerFunc(name: string, func: Function, override?: boolean): void;
    /**
     * Asynchronously load webgpu pipelines when possible.
     * @param mod The input module.
     */
    asyncLoadWebGPUPipelines(mod: Module): Promise<void>;
    /**
     * Initialize webgpu in the runtime.
     * @param device The given GPU device.
     */
    initWebGPU(device: GPUDevice): void;
    /** Register all object factory */
    private registerObjectFactoryFuncs;
    /** Register global packed functions needed by the backend to the env. */
    private registerEnvGlobalPackedFuncs;
    private createPackedFuncFromSafeCallType;
    /**
     * Set packed function arguments into the location indicated by argsValue and argsCode.
     * Allocate new temporary space from the stack if necessary.
     *
     * @parma stack The call stack
     * @param args  The input arguments.
     * @param packedArgs The offset of packedArgs.
     */
    setPackedArguments(stack: CachedCallStack, args: Array<any>, packedArgs: PtrOffset): void;
    private wrapJSFuncAsSafeCallType;
    private makePackedFunc;
    /**
     * Creaye return value of the packed func. The value us auto-tracked for dispose.
     * @param resultAnyPtr The location of rvalue
     * @param callbackArg Whether it is being used in callbackArg.
     * @returns The JS value.
     */
    private retValueToJS;
}
/**
 * Asynchrously instantiate a new {@link Instance}.
 *
 * importObject can also be a {@link LibraryProvider} object,
 * a WASI object, or an object containing wasmLibraryProvider field.
 * We can take benefit of syslib implementations from the Emscripten
 * by passing its generated js Module as the imports.
 *
 * @param bufferSource The source to be compiled.
 * @param importObject The import objects.
 * @param logger The system logger.
 */
export declare function instantiate(bufferSource: ArrayBuffer, importObject?: Record<string, any>, logger?: (msg: string) => void): Promise<Instance>;
export {};
