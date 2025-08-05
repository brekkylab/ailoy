/**
 * Check if value is a promise type
 *
 * @param value The input value
 * @returns Whether value is promise
 */
export declare function isPromise(value: any): boolean;
/**
 * Convert string to Uint8array.
 * @param str The string.
 * @returns The corresponding Uint8Array.
 */
export declare function StringToUint8Array(str: string): Uint8Array;
/**
 * Convert Uint8array to string.
 * @param array The array.
 * @returns The corresponding string.
 */
export declare function Uint8ArrayToString(arr: Uint8Array): string;
/**
 * Internal assert helper
 * @param condition The condition to fail.
 * @param msg The message.
 */
export declare function assert(condition: boolean, msg?: string): asserts condition;
/**
 * Get the path to the wasm library in nodejs.
 * @return The wasm path.
 */
export declare function wasmPath(): string;
/**
 * Linear congruential generator for random number generating that can be seeded.
 *
 * Follows the implementation of `include/tvm/support/random_engine.h`, which follows the
 * sepcification in https://en.cppreference.com/w/cpp/numeric/random/linear_congruential_engine.
 *
 * Note `Number.MAX_SAFE_INTEGER = 2^53 - 1`, and our intermediates are strictly less than 2^48.
 */
export declare class LinearCongruentialGenerator {
    readonly modulus: number;
    readonly multiplier: number;
    readonly increment: number;
    private rand_state;
    /**
     * Set modulus, multiplier, and increment. Initialize `rand_state` according to `Date.now()`.
     */
    constructor();
    /**
     * Sets `rand_state` after normalized with `modulus` to ensure that it is within range.
     * @param seed Any integer. Used to set `rand_state` after normalized with `modulus`.
     *
     * Postcondition: pass `checkRandState()`, i.e. rand_state > 0 and is an integer.
     */
    setSeed(seed: number): void;
    /**
     * Generate the next integer in the range (0, this.modulus) non-inclusive, updating `rand_state`.
     *
     * Postcondition: pass `checkRandState()`, i.e. rand_state > 0 and is an integer.
     */
    nextInt(): number;
    /**
     * Generates random float between (0, 1) non-inclusive, updating `rand_state`.
     *
     * Postcondition: pass `checkRandState()`, i.e. rand_state > 0 and is an integer.
     */
    randomFloat(): number;
    private checkRandState;
}
