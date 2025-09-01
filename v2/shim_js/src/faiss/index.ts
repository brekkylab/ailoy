import FaissModule from "./faiss_bridge";
import type { FaissIndex, FaissIndexSearchResult } from "./faiss_bridge";

type FaissWASM = Awaited<ReturnType<typeof FaissModule>>;
type FaissMetricType = FaissWASM["FaissMetricType"];

export class FaissIndexWrapper {
  static wasm: FaissWASM | undefined;
  private index: FaissIndex;

  constructor(index: FaissIndex) {
    this.index = index;
  }

  get_metric_type(): keyof FaissMetricType {
    const index_metric = this.index.get_metric_type().value;
    const metric_types = Object.keys(
      FaissIndexWrapper.wasm!.FaissMetricType
    ) as (keyof FaissMetricType)[];
    return metric_types.find(
      (key) =>
        FaissIndexWrapper.wasm!.FaissMetricType[key]?.value === index_metric
    )!;
  }

  clear(): void {
    return this.index.clear();
  }

  is_trained(): boolean {
    return this.index.is_trained();
  }

  get_dimension(): number {
    return this.index.get_dimension();
  }

  get_ntotal(): bigint {
    return this.index.get_ntotal();
  }

  train_index(
    training_vectors: Float32Array,
    num_training_vectors: number
  ): void {
    return this.index.train_index(training_vectors, num_training_vectors);
  }

  add_vectors_with_ids(
    vectors: Float32Array,
    num_vectors: number,
    ids: BigInt64Array
  ): void {
    return this.index.add_vectors_with_ids(vectors, num_vectors, ids);
  }

  search_vectors(
    query_vectors: Float32Array,
    k: number
  ): FaissIndexSearchResult {
    return this.index.search_vectors(query_vectors, k);
  }

  get_by_ids(ids: BigInt64Array): Float32Array {
    return this.index.get_by_ids(ids);
  }

  remove_vectors(ids: BigInt64Array): number {
    return this.index.remove_vectors(ids);
  }
}

export async function init(args: {
  dimension: number;
  description: string;
  metric: keyof FaissMetricType;
}) {
  // Load WASM
  if (FaissIndexWrapper.wasm === undefined) {
    FaissIndexWrapper.wasm = await FaissModule();
  }

  // Create Faiss Index
  const index = FaissIndexWrapper.wasm.create_index(
    args.dimension,
    args.description,
    FaissIndexWrapper.wasm.FaissMetricType[args.metric]
  );

  const vectorstore = new FaissIndexWrapper(index);
  return vectorstore;
}

export type { FaissIndexSearchResult };
