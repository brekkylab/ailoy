import FaissModule from "./faiss_bridge";
import type { FaissIndexHandle, FaissIndexSearchResult } from "./faiss_bridge";

type FaissWASM = Awaited<ReturnType<typeof FaissModule>>;
type FaissMetricType = FaissWASM["FaissMetricType"];

/**
 * By "Inner" it means it's used inside Rust's Faiss FFI wrapper.
 */
export class FaissIndexInner {
  static wasm: FaissWASM | undefined;
  private handle: FaissIndexHandle;

  constructor(handle: FaissIndexHandle) {
    this.handle = handle;
  }

  get_metric_type(): keyof FaissMetricType {
    const index_metric = this.handle.get_metric_type().value;
    const metric_types = Object.keys(
      FaissIndexInner.wasm!.FaissMetricType
    ) as (keyof FaissMetricType)[];
    return metric_types.find(
      (key) =>
        FaissIndexInner.wasm!.FaissMetricType[key]?.value === index_metric
    )!;
  }

  clear(): void {
    return this.handle.clear();
  }

  is_trained(): boolean {
    return this.handle.is_trained();
  }

  get_dimension(): number {
    return this.handle.get_dimension();
  }

  get_ntotal(): bigint {
    return this.handle.get_ntotal();
  }

  train_index(
    training_vectors: Float32Array,
    num_training_vectors: number
  ): void {
    return this.handle.train_index(training_vectors, num_training_vectors);
  }

  add_vectors_with_ids(
    vectors: Float32Array,
    num_vectors: number,
    ids: BigInt64Array
  ): void {
    return this.handle.add_vectors_with_ids(vectors, num_vectors, ids);
  }

  search_vectors(
    query_vectors: Float32Array,
    k: number
  ): FaissIndexSearchResult {
    return this.handle.search_vectors(query_vectors, k);
  }

  get_by_ids(ids: BigInt64Array): Float32Array {
    return this.handle.get_by_ids(ids);
  }

  remove_vectors(ids: BigInt64Array): number {
    return this.handle.remove_vectors(ids);
  }
}

export async function init(args: {
  dimension: number;
  description: string;
  metric: keyof FaissMetricType;
}) {
  // Load WASM
  if (FaissIndexInner.wasm === undefined) {
    FaissIndexInner.wasm = await FaissModule();
  }

  // Create Faiss Index
  const index = FaissIndexInner.wasm.create_index(
    args.dimension,
    args.description,
    FaissIndexInner.wasm.FaissMetricType[args.metric]
  );

  const vectorstore = new FaissIndexInner(index);
  return vectorstore;
}

export type { FaissIndexSearchResult };
