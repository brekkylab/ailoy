export { create_faiss_index, get_metric_type } from "./faiss";
export type {
  FaissIndexInner,
  FaissIndexSearchResult,
  FaissMetricType,
} from "./faiss";

export {
  Scalar,
  DLDevice,
  TVMObject,
  TVMArray,
  Tensor,
  Module,
  Instance,
  instantiate,
  getGPUDevice,
} from "./tvmjs";
