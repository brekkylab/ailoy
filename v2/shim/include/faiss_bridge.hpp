#pragma once

#include <memory>
#include <string>
#include <vector>

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/index_factory.h>

#include "rust/cxx.h"

enum class FaissMetricType : uint8_t;

namespace faiss_bridge {

class FaissIndexWrapper {
private:
  std::unique_ptr<faiss::Index> index_;

public:
  explicit FaissIndexWrapper(std::unique_ptr<faiss::Index> index);
  ~FaissIndexWrapper() = default;

  // No copy constructors
  FaissIndexWrapper(const FaissIndexWrapper &) = delete;
  FaissIndexWrapper &operator=(const FaissIndexWrapper &) = delete;

  // Move constructors
  FaissIndexWrapper(FaissIndexWrapper &&) = default;
  FaissIndexWrapper &operator=(FaissIndexWrapper &&) = default;

  bool is_trained() const;
  int64_t get_ntotal() const;
  int32_t get_dimension() const;

  // FAISS 작업들
  void train_index(rust::Slice<const float> training_vectors,
                   size_t num_training_vectors);
  void add_vectors_with_ids(rust::Slice<const float> vectors,
                            size_t num_vectors, rust::Slice<const int64_t> ids);
  void search_vectors(rust::Slice<const float> query_vectors,
                      size_t num_queries, size_t k,
                      rust::Slice<float> distances,
                      rust::Slice<int64_t> indices) const;
};

// FaissIndexWrapper 생성 함수
std::unique_ptr<FaissIndexWrapper>
create_index(int32_t dimension, rust::Str description, FaissMetricType metric);

} // namespace faiss_bridge
