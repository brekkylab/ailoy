/**
 * enums to check the current state of asynctify
 */
declare const enum AsyncifyStateKind {
    None = 0,
    Unwinding = 1,
    Rewinding = 2
}
/** Hold asynctify handler instance that runtime can use */
export declare class AsyncifyHandler {
    /** exports from wasm */
    private exports;
    /** current state kind */
    private state;
    /** The stored value before unwind */
    private storedPromiseBeforeUnwind;
    /** The stored value that is resolved */
    private storedValueBeforeRewind;
    /** The stored exception */
    private storedExceptionBeforeRewind;
    constructor(exports: Record<string, Function>, memory: WebAssembly.Memory);
    /**
     * Whether the wasm enables asynctify
     * @returns Whether the wasm enables asynctify
     */
    enabled(): boolean;
    /**
     * Get the current asynctify state
     *
     * @returns The current asynctify state
     */
    isNormalStackState(): boolean;
    /**
     * Get the current asynctify state
     *
     * @returns The current asynctify state
     */
    getState(): AsyncifyStateKind;
    /**
     * Wrap a function that can be used as import of the wasm asynctify layer
     *
     * @param func The input import function
     * @returns The wrapped function that can be registered to the system
     */
    wrapImport(func: (...args: Array<any>) => any): (...args: Array<any>) => any;
    /**
     * Warp an exported asynctify function so it can return promise
     *
     * @param func The input function
     * @returns The wrapped async function
     */
    wrapExport(func: (...args: Array<any>) => any): (...args: Array<any>) => Promise<any>;
    private startRewind;
    private stopRewind;
    private startUnwind;
    private stopUnwind;
    /**
     * Initialize the wasm memory to setup necessary meta-data
     * for asynctify handling
     * @param memory The memory ti
     */
    private initMemory;
}
export {};
