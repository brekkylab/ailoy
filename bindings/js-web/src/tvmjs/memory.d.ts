/**
 * Classes to manipulate Wasm memories.
 */
import { Pointer, PtrOffset } from "./ctypes";
import { Disposable } from "./types";
import * as ctypes from "./ctypes";
/**
 * Wasm Memory wrapper to perform JS side raw memory access.
 */
export declare class Memory {
    memory: WebAssembly.Memory;
    wasm32: boolean;
    private buffer;
    private viewU8;
    private viewU16;
    private viewI32;
    private viewU32;
    private viewF32;
    private viewF64;
    constructor(memory: WebAssembly.Memory);
    loadU8(ptr: Pointer): number;
    loadU16(ptr: Pointer): number;
    loadU32(ptr: Pointer): number;
    loadI32(ptr: Pointer): number;
    loadI64(ptr: Pointer): number;
    loadF32(ptr: Pointer): number;
    loadF64(ptr: Pointer): number;
    loadPointer(ptr: Pointer): Pointer;
    loadUSize(ptr: Pointer): Pointer;
    sizeofPtr(): number;
    /**
     * Load raw bytes from ptr.
     * @param ptr The head address
     * @param numBytes The number
     */
    loadRawBytes(ptr: Pointer, numBytes: number): Uint8Array;
    /**
     * Load null-terminated C-string from ptr.
     * @param ptr The head address
     */
    loadCString(ptr: Pointer): string;
    /**
     * Store raw bytes to the ptr.
     * @param ptr The head address.
     * @param bytes The bytes content.
     */
    storeRawBytes(ptr: Pointer, bytes: Uint8Array): void;
    /**
     * Load the object type index from the object handle.
     * @param objectHandle The handle of the object.
     * @returns The object type index.
     */
    loadObjectTypeIndex(objectHandle: Pointer): number;
    /**
     * Load the type key from the type info pointer.
     * @param typeInfoPtr The pointer to the type info.
     * @returns The type key.
     */
    loadTypeInfoTypeKey(typeInfoPtr: Pointer): string;
    /**
     * Load bytearray as string from ptr.
     * @param byteArrayPtr The head address of the bytearray.
     */
    loadByteArrayAsString(byteArrayPtr: Pointer): string;
    /**
     * Load bytearray as bytes from ptr.
     * @param byteArrayPtr The head address of the bytearray.
     */
    loadByteArrayAsBytes(byteArrayPtr: Pointer): Uint8Array;
    /**
     * Update memory view after the memory growth.
     */
    private updateViews;
}
/**
 * Auxiliary call stack for the FFI calls.
 *
 * Lifecyle of a call stack.
 * - Calls into allocXX to allocate space, mixed with storeXXX to store data.
 * - Calls into ptrFromOffset, no further allocation(as ptrFromOffset can change),
 *   can still call into storeXX
 * - Calls into commitToWasmMemory once.
 * - reset.
 */
export declare class CachedCallStack implements Disposable {
    /** List of temporay arguments that can be disposed during reset. */
    tempArgs: Array<Disposable>;
    private memory;
    private cAllocSpace;
    private cFreeSpace;
    private buffer;
    private viewU8;
    private viewI32;
    private viewU32;
    private viewF64;
    private stackTop;
    private basePtr;
    private addressToSetTargetValue;
    constructor(memory: Memory, allocSpace: ctypes.FTVMWasmAllocSpace, freeSpace: ctypes.FTVMWasmFreeSpace);
    dispose(): void;
    /**
     * Rest the call stack so that it can be reused again.
     */
    reset(): void;
    /**
     * Commit all the cached data to WasmMemory.
     * This function can only be called once.
     * No further store function should be called.
     *
     * @param nbytes Number of bytes to be stored.
     */
    commitToWasmMemory(nbytes?: number): void;
    /**
     * Allocate space by number of bytes
     * @param nbytes Number of bytes.
     * @note This function always allocate space that aligns to 64bit.
     */
    allocRawBytes(nbytes: number): PtrOffset;
    /**
     * Allocate space for pointers.
     * @param count Number of pointers.
     * @returns The allocated pointer array.
     */
    allocPtrArray(count: number): PtrOffset;
    /**
     * Get the real pointer from offset values.
     * Note that the returned value becomes obsolete if alloc is called on the stack.
     * @param offset The allocated offset.
     */
    ptrFromOffset(offset: PtrOffset): Pointer;
    storePtr(offset: PtrOffset, value: Pointer): void;
    storeUSize(offset: PtrOffset, value: Pointer): void;
    storeI32(offset: PtrOffset, value: number): void;
    storeU32(offset: PtrOffset, value: number): void;
    storeI64(offset: PtrOffset, value: number): void;
    storeF64(offset: PtrOffset, value: number): void;
    storeRawBytes(offset: PtrOffset, bytes: Uint8Array): void;
    /**
     * Allocate a byte array for a string and return the offset of the byte array.
     * @param data The string to allocate.
     * @returns The offset of the byte array.
     */
    allocByteArrayForString(data: string): PtrOffset;
    /**
     * Allocate then set C-String pointer to the offset.
     * This function will call into allocBytes to allocate necessary data.
     * The address won't be set immediately(because the possible change of basePtr)
     * and will be filled when we commit the data.
     *
     * @param offset The offset to set ot data pointer.
     * @param data The string content.
     */
    allocThenSetArgString(offset: PtrOffset, data: string): void;
    /**
     * Allocate then set the argument location with a TVMByteArray.
     * Allocate new temporary space for bytes.
     *
     * @param offset The offset to set ot data pointer.
     * @param data The string content.
     */
    allocThenSetArgBytes(offset: PtrOffset, data: Uint8Array): void;
    /**
     * Update internal cache views.
     */
    private updateViews;
}
