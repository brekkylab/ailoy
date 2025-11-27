import FaissModule from "./faiss_bridge";
import type { FaissIndexSearchResult, FaissIndexInner } from "./faiss_bridge";

type FaissWASM = Awaited<ReturnType<typeof FaissModule>>;
type FaissMetricType = FaissWASM["FaissMetricType"];

declare global {
  interface Window {
    __faiss_module__: FaissWASM | undefined;
  }
}

export async function create_faiss_index(
  dimension: number,
  description: string,
  metric: keyof FaissMetricType
) {
  // Load Faiss WASM module
  if (window.__faiss_module__ == undefined) {
    window.__faiss_module__ = await FaissModule();
  }
  const module = window.__faiss_module__;

  // Create Faiss VectorStore instance
  const vectorstore = new module.FaissIndexInner(
    dimension,
    description,
    await get_metric_type(metric)
  );
  return vectorstore;
}

export async function get_metric_type(type: keyof FaissMetricType) {
  // Load Faiss WASM module
  if (window.__faiss_module__ == undefined) {
    window.__faiss_module__ = await FaissModule();
  }
  const module = window.__faiss_module__;

  return module.FaissMetricType[type];
}

export type { FaissMetricType, FaissIndexSearchResult, FaissIndexInner };
