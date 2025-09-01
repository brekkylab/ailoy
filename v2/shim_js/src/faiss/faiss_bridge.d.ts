// TypeScript bindings for emscripten-generated code.  Automatically generated at compile time.
interface WasmModule {
}

type EmbindString = ArrayBuffer|Uint8Array|Uint8ClampedArray|Int8Array|string;
export interface ClassHandle {
  isAliasOf(other: ClassHandle): boolean;
  delete(): void;
  deleteLater(): this;
  isDeleted(): boolean;
  clone(): this;
}
export interface FaissMetricTypeValue<T extends number> {
  value: T;
}
export type FaissMetricType = FaissMetricTypeValue<0>|FaissMetricTypeValue<1>|FaissMetricTypeValue<2>|FaissMetricTypeValue<3>|FaissMetricTypeValue<4>|FaissMetricTypeValue<20>|FaissMetricTypeValue<21>|FaissMetricTypeValue<22>|FaissMetricTypeValue<23>;

export type FaissIndexSearchResult = {
  distances: Float32Array,
  indexes: BigInt64Array
};

export interface FaissIndex extends ClassHandle {
  get_metric_type(): FaissMetricType;
  clear(): void;
  is_trained(): boolean;
  get_dimension(): number;
  get_ntotal(): bigint;
  train_index(_0: any, _1: number): void;
  add_vectors_with_ids(_0: any, _1: number, _2: any): void;
  search_vectors(_0: any, _1: number): FaissIndexSearchResult;
  get_by_ids(_0: any): Float32Array;
  remove_vectors(_0: any): number;
}

interface EmbindModule {
  FaissMetricType: {InnerProduct: FaissMetricTypeValue<0>, L2: FaissMetricTypeValue<1>, L1: FaissMetricTypeValue<2>, Linf: FaissMetricTypeValue<3>, Lp: FaissMetricTypeValue<4>, Canberra: FaissMetricTypeValue<20>, BrayCurtis: FaissMetricTypeValue<21>, JensenShannon: FaissMetricTypeValue<22>, Jaccard: FaissMetricTypeValue<23>};
  FaissIndex: {};
  create_index(_0: number, _1: EmbindString, _2: FaissMetricType): FaissIndex;
}

export type MainModule = WasmModule & EmbindModule;
export default function MainModuleFactory (options?: unknown): Promise<MainModule>;
