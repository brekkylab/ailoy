#pragma once

#include <memory>
#include <string>
#include <vector>

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/index_factory.h>

#include "rust/cxx.h"

// Forward Declaration for cxx_bridge.rs.h
enum class FaissMetricType : uint8_t;
struct FaissSearchResult;

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
  FaissMetricType get_metric_type() const;

  void add_vectors_with_ids(rust::Slice<const float> vectors,
                            size_t num_vectors, rust::Slice<const int64_t> ids);
  FaissSearchResult search_vectors(rust::Slice<const float> query_vectors,
                                   size_t k) const;
  void train_index(rust::Slice<const float> training_vectors,
                   size_t num_training_vectors);
  rust::Vec<float> get_by_id(int64_t id) const;
  size_t remove_vectors(rust::Slice<const int64_t> ids);
  void clear();

  void write_index(rust::Str filename) const;
};

// FaissIndexWrapper 생성 함수
std::unique_ptr<FaissIndexWrapper>
create_index(int32_t dimension, rust::Str description, FaissMetricType metric);

std::unique_ptr<FaissIndexWrapper> read_index(rust::Str filename);

} // namespace faiss_bridge
