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

export interface FaissIndexWrapper extends ClassHandle {
  get_metric_type(): FaissMetricType;
  clear(): void;
  is_trained(): boolean;
  get_dimension(): number;
  get_ntotal(): bigint;
  train_index(_0: any, _1: number): void;
  add_vectors_with_ids(_0: any, _1: number, _2: any): void;
  search_vectors(_0: any, _1: number): FaissIndexSearchResult;
  get_by_ids(_0: any): VectorFloat;
  remove_vectors(_0: any): number;
}

export interface VectorFloat extends ClassHandle {
  size(): number;
  get(_0: number): number | undefined;
  push_back(_0: number): void;
  resize(_0: number, _1: number): void;
  set(_0: number, _1: number): boolean;
}

export interface VectorInt64 extends ClassHandle {
  size(): number;
  get(_0: number): bigint | undefined;
  push_back(_0: bigint): void;
  resize(_0: number, _1: bigint): void;
  set(_0: number, _1: bigint): boolean;
}

export type FaissIndexSearchResult = {
  distances: VectorFloat,
  ids: VectorInt64
};

interface EmbindModule {
  FaissMetricType: {METRIC_INNER_PRODUCT: FaissMetricTypeValue<0>, METRIC_L2: FaissMetricTypeValue<1>, METRIC_L1: FaissMetricTypeValue<2>, METRIC_Linf: FaissMetricTypeValue<3>, METRIC_Lp: FaissMetricTypeValue<4>, METRIC_Canberra: FaissMetricTypeValue<20>, METRIC_BrayCurtis: FaissMetricTypeValue<21>, METRIC_JensenShannon: FaissMetricTypeValue<22>, METRIC_Jaccard: FaissMetricTypeValue<23>};
  FaissIndexWrapper: {};
  VectorFloat: {
    new(): VectorFloat;
  };
  VectorInt64: {
    new(): VectorInt64;
  };
  create_index(_0: number, _1: EmbindString, _2: FaissMetricType): FaissIndexWrapper;
}

export type MainModule = WasmModule & EmbindModule;
export default function MainModuleFactory (options?: unknown): Promise<MainModule>;
