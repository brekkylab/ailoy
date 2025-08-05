/** NodeJS and Web compact layer */
import { LibraryProvider } from "./types";
/**
 * Get performance measurement.
 */
export declare function getPerformance(): Performance;
/**
 * Create a new websocket for a given URL
 * @param url The url.
 */
export declare function createWebSocket(url: string): WebSocket;
/**
 * Create a WASI based on current environment.
 *
 * @return A wasi that can run on broswer or local.
 */
export declare function createPolyfillWASI(): LibraryProvider;
