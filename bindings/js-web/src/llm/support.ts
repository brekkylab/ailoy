/** Util methods. */
import { AppConfig, ModelRecord } from "./config";
import {
  ModelNotFoundError,
  ModelNotLoadedError,
  SpecifiedModelNotFoundError,
  UnclearModelToUseError,
} from "./error";

export function findModelRecord(
  modelId: string,
  appConfig: AppConfig
): ModelRecord {
  const matchedItem = appConfig.model_list.find(
    (item) => item.model_id == modelId
  );
  if (matchedItem !== undefined) return matchedItem;
  throw new ModelNotFoundError(modelId);
}

/**
 * Return the model to use given the loaded modelIds and requestModel. Throws error when unclear
 * which model to load.
 * @param loadedModelIds Models currently loaded in the engine.
 * @param requestModel Model the user specified to load via the request. Required when multiple
 *   models are loaded
 * @param requestName The type of request or API to load the model for. Needed for error throwing.
 */
export function getModelIdToUse(
  loadedModelIds: string[],
  requestModel: string | undefined | null,
  requestName: string
): string {
  let selectedModelId: string;
  if (loadedModelIds.length === 0) {
    throw new ModelNotLoadedError(requestName);
  }
  if (requestModel) {
    // If specified model
    if (loadedModelIds.indexOf(requestModel) === -1) {
      throw new SpecifiedModelNotFoundError(
        loadedModelIds,
        requestModel,
        requestName
      );
    } else {
      selectedModelId = requestModel;
    }
  } else {
    // If not specified
    if (loadedModelIds.length > 1) {
      throw new UnclearModelToUseError(loadedModelIds, requestName);
    } else {
      selectedModelId = loadedModelIds[0];
    }
  }
  return selectedModelId;
}

/**
 * Chunk the inputData such that each chunk's total input length is smaller than prefill
 * chunk size.
 * @returns [the data chunks, the input length of each chunk]
 * @note precondition: if inputData has image in it, then prefillChunkSize >= IMAGE_EMBED_SIZE.
 */
export function getChunkedPrefillInputData(
  // inputData: Array<Array<number> | ImageURL>,
  inputData: Array<Array<number>>,
  prefillChunkSize: number
  // ): [Array<Array<number> | ImageURL>[], Array<number>] {
): [Array<Array<number>>[], Array<number>] {
  // const chunks: Array<Array<number> | ImageURL>[] = [];
  // const chunkLens: Array<number> = [];
  // let curChunk: Array<Array<number> | ImageURL> = [];
  // let curChunkLen = 0;
  const chunks: Array<Array<number>>[] = [];
  const chunkLens: Array<number> = [];
  let curChunk: Array<Array<number>> = [];
  let curChunkLen = 0;
  for (let i = 0; i < inputData.length; i++) {
    // let curData: Array<number> | ImageURL = inputData[i];
    let curData: Array<number> = inputData[i];
    // const curDataLen = Array.isArray(curData)
    //   ? curData.length
    //   : IMAGE_EMBED_SIZE;
    const curDataLen = curData.length;
    // 1. curData can fit into this chunk
    if (curChunkLen + curDataLen <= prefillChunkSize) {
      curChunk.push(curData);
      curChunkLen += curDataLen;
      if (curChunkLen === prefillChunkSize) {
        chunks.push([...curChunk]);
        chunkLens.push(curChunkLen);
        curChunk = [];
        curChunkLen = 0;
      }
      continue;
    }

    // 2. Otherwise, depends on whether it is token data or image data
    if (Array.isArray(curData)) {
      // 2.1. Token data, which itself needs to be chunked. Keep
      // chunking and finalizing until finished
      while (curData.length > 0) {
        const curDataToChunkLen = Math.min(
          curData.length,
          prefillChunkSize - curChunkLen
        );
        curChunk.push(curData.slice(0, curDataToChunkLen));
        curChunkLen += curDataToChunkLen;
        curData = curData.slice(curDataToChunkLen);
        if (curChunkLen === prefillChunkSize) {
          // curChunk is now full, so finalize to chunks
          chunks.push([...curChunk]);
          chunkLens.push(curChunkLen);
          curChunk = [];
          curChunkLen = 0;
        }
      }
    }
    // else {
    //   // 2.2. Image data, which itself cannot be chunked, so cannot fit in current chunk.
    //   // 2.2.1. Finalize curChunk
    //   if (curChunk.length === 0) {
    //     throw new Error(
    //       "InternalError: do not expect curChunk to be empty when an image does not fit."
    //     );
    //   }
    //   chunks.push([...curChunk]);
    //   chunkLens.push(curChunkLen);
    //   // 2.2.2. Then push image to the new chunk
    //   curChunk = [curData];
    //   curChunkLen = IMAGE_EMBED_SIZE;
    //   if (curChunkLen === prefillChunkSize) {
    //     chunks.push([...curChunk]);
    //     chunkLens.push(curChunkLen);
    //     curChunk = [];
    //     curChunkLen = 0;
    //   }
    // }
  }
  // Last chunk
  if (curChunk.length > 0) {
    chunks.push([...curChunk]);
    chunkLens.push(curChunkLen);
  }

  return [chunks, chunkLens];
}

type Cont = () => void;

/**
 * A lock implemented using Promise.
 *
 * Referred to:
 * - https://jackpordi.com/posts/locks-in-js-because-why-not
 * - https://www.linkedin.com/pulse/asynchronous-locking-using-promises-javascript-abdul-ahad-o7smf/
 */
export class CustomLock {
  private acquired = false;
  private readonly queue: Cont[] = [];

  public async acquire(): Promise<void> {
    if (!this.acquired) {
      // If lock is free, directly return
      this.acquired = true;
    } else {
      // Otherwise, push the request to the queue, and
      // a future release() will resolve it
      return new Promise<void>((resolve) => {
        this.queue.push(resolve);
      });
    }
  }

  public async release(): Promise<void> {
    if (!this.acquired) {
      throw Error("InternalError: expect lock is acquired upon release()");
    }
    if (this.queue.length === 0) {
      // No one is waiting for the lock, so we free it
      this.acquired = false;
      return;
    }

    // Otherwise, hand the execution to the next in queue, and
    // the lock is still acquired
    const cont = this.queue.shift();
    return new Promise((res: Cont) => {
      cont!();
      res();
    });
  }
}
