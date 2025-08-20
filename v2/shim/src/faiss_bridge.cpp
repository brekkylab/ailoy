#include "faiss_bridge.hpp"
#include <iostream>
#include <stdexcept>

namespace faiss_bridge {

FaissIndexWrapper::FaissIndexWrapper(std::unique_ptr<faiss::Index> index)
    : index_(std::move(index)) {
  if (!index_) {
    throw std::runtime_error("FaissIndexWrapper: null index provided");
  }
}

bool FaissIndexWrapper::is_trained() const { return index_->is_trained; }

int64_t FaissIndexWrapper::get_ntotal() const { return index_->ntotal; }

int32_t FaissIndexWrapper::get_dimension() const {
  return static_cast<int32_t>(index_->d);
}

void FaissIndexWrapper::train_index(rust::Slice<const float> training_vectors,
                                    size_t num_training_vectors) {
  if (index_->is_trained)
    return; // skip training
  index_->train(static_cast<faiss::idx_t>(num_training_vectors),
                training_vectors.data());
}

void FaissIndexWrapper::add_vectors_with_ids(rust::Slice<const float> vectors,
                                             size_t num_vectors,
                                             rust::Slice<const int64_t> ids) {
  // FAISS는 faiss::idx_t를 사용하므로 변환이 필요할 수 있습니다.
  std::vector<faiss::idx_t> faiss_ids(ids.begin(), ids.end());

  index_->add_with_ids(static_cast<faiss::idx_t>(num_vectors), vectors.data(),
                       faiss_ids.data());
}

void FaissIndexWrapper::search_vectors(rust::Slice<const float> query_vectors,
                                       size_t num_queries, size_t k,
                                       rust::Slice<float> distances,
                                       rust::Slice<int64_t> indices) const {

  // FAISS 검색 결과를 임시 배열에 저장
  std::vector<faiss::idx_t> faiss_indices(num_queries * k);

  index_->search(static_cast<faiss::idx_t>(num_queries), query_vectors.data(),
                 static_cast<faiss::idx_t>(k), distances.data(),
                 faiss_indices.data());

  // faiss::idx_t를 int64_t로 변환
  for (size_t i = 0; i < faiss_indices.size(); ++i) {
    indices[i] = static_cast<int64_t>(faiss_indices[i]);
  }
}

std::unique_ptr<FaissIndexWrapper>
create_index(int32_t dimension, rust::Str description, FaissMetricType metric) {
  try {
    faiss::MetricType faiss_metric = faiss::MetricType(metric);

    std::string desc_str(description);
    auto faiss_index = std::unique_ptr<faiss::Index>(faiss::index_factory(
        static_cast<int>(dimension), desc_str.c_str(), faiss_metric));

    return std::make_unique<FaissIndexWrapper>(std::move(faiss_index));
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to create FAISS index: " +
                             std::string(e.what()));
  }
}

} // namespace faiss_bridge
