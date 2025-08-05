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
/**
 *   Common Interface for the artifact cache
 */
export interface ArtifactCacheTemplate {
    /**
     * Retrieve data object that corresponds to `url` from cache. If data object does not exist in
     * cache, fetch the data and then add to cache.
     *
     * @param url: The url to the data to be cached.
     * @param storetype: This field is required so that `ArtifactIndexedDBCache` can store the
     * actual data object (see `addToCache()`), while `ArtifactCache` which uses the Cache API can
     * return the actual data object rather than the request. There are two options:
     * 1. "json": returns equivalent to `fetch(url).json()`
     * 2. "arraybuffer": returns equivalent to `fetch(url).arraybuffer()`
     * @param signal: An optional AbortSignal allowing user to abort the fetching before its completion.
     * @return The data object (i.e. users do not need to call `.json()` or `.arraybuffer()`).
     *
     * @note This is an async function.
     */
    fetchWithCache(url: string, storetype?: string, signal?: AbortSignal): Promise<any>;
    /**
     * Fetch data from url and add into cache. If already exists in cache, should return instantly.
     *
     * @param url: The url to the data to be cached.
     * @param storetype: Only applies to `ArtifactIndexedDBCache`. Since `indexedDB` stores the actual
     * @param signal: An optional AbortSignal to abort data retrival
     * data rather than a request, we specify `storagetype`. There are two options:
     * 1. "json": IndexedDB stores `fetch(url).json()`
     * 2. "arraybuffer": IndexedDB stores `fetch(url).arrayBuffer()`
     *
     * @note This is an async function.
     */
    addToCache(url: string, storetype?: string, signal?: AbortSignal): Promise<void>;
    /**
     * check if cache has all keys in Cache
     *
     * @note This is an async function.
     */
    hasAllKeys(keys: string[]): Promise<boolean>;
    /**
     * Delete url in cache if url exists
     *
     * @note This is an async function.
     */
    deleteInCache(url: string): Promise<void>;
}
/**
 * Cache to store model related data, implemented with the Cache API.
 */
export declare class ArtifactCache implements ArtifactCacheTemplate {
    private scope;
    private cache?;
    constructor(scope: string);
    /**
     * Convert the Response object to the expected storetype instead
     */
    responseTostoretype(response: Response, storetype?: string): Promise<any>;
    /**
     * fetch the corresponding url object in response or stored object format
     * @param url url
     * @param storetype the storage type for indexedDB
     * @param signal an optional abort signal to abort fetching
     * @returns response in json, arraybuffer or pure response format
     */
    fetchWithCache(url: string, storetype?: string, signal?: AbortSignal): Promise<any>;
    addToCache(url: string, storetype?: string, signal?: AbortSignal): Promise<void>;
    /**
     * Determine if all keys exist in the cache
     * @param keys the url key list of the strings
     * @returns boolean value indicate if all keys are in cache
     */
    hasAllKeys(keys: string[]): Promise<boolean>;
    /**
     * Delete the corresponding url object in cache
     * @param url the corresponding url object to be deleted
     */
    deleteInCache(url: string): Promise<void>;
}
/**
 * Cache by IndexedDB to support caching model data
 */
export declare class ArtifactIndexedDBCache implements ArtifactCacheTemplate {
    private dbName?;
    private dbVersion;
    private db;
    constructor(dbName: string);
    /**
     * Init the indexed DB database if it is not initialized.
     */
    private initDB;
    /**
     * Check if current url object is in indexedDB or not
     * @param url the url link
     * @returns boolean indicate if url object in indexedDB
     */
    private isUrlInDB;
    asyncGetHelper(url: string): Promise<any>;
    fetchWithCache(url: string, storetype?: string, signal?: AbortSignal): Promise<any>;
    addToIndexedDB(url: string, response: any, storetype?: string): Promise<void>;
    addToCache(url: string, storetype?: string, signal?: AbortSignal): Promise<void>;
    hasAllKeys(keys: string[]): Promise<boolean>;
    deleteInCache(url: string): Promise<void>;
}
/**
 * Function to check if NDarray is in Cache or not
 *
 * @param ndarrayCacheUrl The cache url which links to the NDArray
 * @param cacheScope The scope identifier of the cache
 * @param cacheType The type of the cache: "cache" or "indexedDB"
 * @returns the result if the cache has NDArray
 */
export declare function hasNDArrayInCache(ndarrayCacheUrl: string, cacheScope?: string, cacheType?: string): Promise<boolean>;
/**
 * Given cacheUrl, search up items to delete based on cacheUrl/ndarray-cache.json
 *
 * @param cacheUrl The cacheUrl for the items
 * @param cacheScope The scope identifier of the cache
 * @param cacheType The type of the cache: "cache" or "indexedDB"
 */
export declare function deleteNDArrayCache(cacheUrl: string, cacheScope?: string, cacheType?: string): Promise<void>;
