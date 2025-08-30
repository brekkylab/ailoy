import FaissModule from "./faiss/faiss_bridge";
import type { FaissIndexWrapper } from "./faiss/faiss_bridge";

type FaissWASM = Awaited<ReturnType<typeof FaissModule>>;
type FaissMetricType = FaissWASM["FaissMetricType"];
type NumericTypedArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

type BigIntTypedArray = BigInt64Array | BigUint64Array;

export function concatTypedArrays<
  T extends NumericTypedArray | BigIntTypedArray
>(arrays: T[]): T {
  if (arrays.length === 0) {
    throw new Error("concatTypedArrays requires at least one array");
  }

  const ArrayType = arrays[0].constructor as { new (length: number): T };

  const totalLength = arrays.reduce((acc, a) => acc + a.length, 0);
  const result = new ArrayType(totalLength);

  let offset = 0;
  for (const a of arrays) {
    if (typeof (a as any)[0] === "bigint") {
      (result as BigIntTypedArray).set(a as BigIntTypedArray, offset);
    } else {
      (result as NumericTypedArray).set(a as NumericTypedArray, offset);
    }
    offset += a.length;
  }

  return result;
}

export class FaissVectorStore {
  private wasm: FaissWASM;
  private index: FaissIndexWrapper;

  constructor(wasm: FaissWASM, index: FaissIndexWrapper) {
    this.wasm = wasm;
    this.index = index;
  }

  get_metric_type(): keyof FaissMetricType {
    const index_metric = this.index.get_metric_type().value;
    const metric_types = Object.keys(
      this.wasm.FaissMetricType
    ) as (keyof FaissMetricType)[];
    return metric_types.find(
      (key) => this.wasm.FaissMetricType[key]?.value === index_metric
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

  get_ntotal(): number {
    return Number(this.index.get_ntotal());
  }

  // This is used only for specific index types such as IVF
  train_index(training_vectors: Array<Float32Array>): void {
    const concatVector = concatTypedArrays(training_vectors);
    return this.index.train_index(concatVector, training_vectors.length);
  }

  add_vectors_with_ids(
    inputs: Array<{ vector: Float32Array; id: number }>
  ): void {
    const concatVector = concatTypedArrays(inputs.map((item) => item.vector));
    const ids = inputs.map((item) => item.id);
    return this.index.add_vectors_with_ids(concatVector, inputs.length, ids);
  }

  search_vectors(
    query_vectors: Array<Float32Array>,
    k: number
  ): Array<Array<{ id: number; distance: number }>> {
    const min_k = Math.min(this.get_ntotal(), k);
    if (min_k === 0) {
      return [];
    }

    const concatVector = concatTypedArrays(query_vectors);
    const search_results = this.index.search_vectors(concatVector, min_k);

    let results = [];
    for (let n = 0; n < query_vectors.length; n++) {
      let results_for_query = [];
      for (let i = 0; i < min_k; i++) {
        const id = Number(search_results.ids.get(i)!);
        const distance = search_results.distances.get(i)!;
        results_for_query.push({ id, distance });
      }
      results.push(results_for_query);
    }
    return results;
  }

  get_by_ids(ids: Array<number>): Array<Float32Array> {
    const raw_results = this.index.get_by_ids(
      new BigInt64Array(ids.map((id) => BigInt(id)))
    );
    const results = new Array<Float32Array>(ids.length);
    const dimension = this.get_dimension();
    for (let i = 0; i < ids.length; i++) {
      const vector = new Float32Array(dimension);
      const startIndex = i * dimension;

      for (let j = 0; j < dimension; j++) {
        vector[j] = raw_results.get(startIndex + j) ?? 0;
      }

      results[i] = vector;
    }

    return results;
  }

  remove_vectors(ids: Array<number>): number {
    const num_removed = this.index.remove_vectors(
      new BigInt64Array(ids.map((id) => BigInt(id)))
    );
    return num_removed;
  }
}

export async function init(args: {
  dimension: number;
  description: string;
  metric: keyof FaissMetricType;
}) {
  // Load WASM
  const wasm = await FaissModule();

  // Create Faiss Index
  const index = wasm.create_index(
    args.dimension,
    args.description,
    wasm.FaissMetricType[args.metric]
  );

  const vectorstore = new FaissVectorStore(wasm, index);
  return vectorstore;
}
