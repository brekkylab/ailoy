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
struct FaissIndexSearchResult;

namespace faiss_bridge {

class FaissIndexInner {
private:
  std::unique_ptr<faiss::Index> index_;

public:
  explicit FaissIndexInner(std::unique_ptr<faiss::Index> index);
  ~FaissIndexInner() = default;

  // No copy constructors
  FaissIndexInner(const FaissIndexInner &) = delete;
  FaissIndexInner &operator=(const FaissIndexInner &) = delete;

  // Move constructors
  FaissIndexInner(FaissIndexInner &&) = default;
  FaissIndexInner &operator=(FaissIndexInner &&) = default;

  bool is_trained() const;
  int64_t get_ntotal() const;
  int32_t get_dimension() const;
  FaissMetricType get_metric_type() const;

  void add_vectors_with_ids(rust::Slice<const float> vectors,
                            size_t num_vectors, rust::Slice<const int64_t> ids);
  FaissIndexSearchResult search_vectors(rust::Slice<const float> query_vectors,
                                        size_t k) const;
  void train_index(rust::Slice<const float> training_vectors,
                   size_t num_training_vectors);
  rust::Vec<float> get_by_id(int64_t id) const;

  // assume that for every id, there is a vector corresponding to that id.
  // This should be guaranteed before call this function.
  rust::Vec<float> get_by_ids(rust::Slice<const int64_t> ids) const;

  // assume that for every id, there is a vector corresponding to that id.
  // This should be guaranteed before call this function.
  size_t remove_vectors(rust::Slice<const int64_t> ids);
  void clear();

  void write_index(rust::Str filename) const;
};

// FaissIndexInner factory
std::unique_ptr<FaissIndexInner>
create_index(int32_t dimension, rust::Str description, FaissMetricType metric);

std::unique_ptr<FaissIndexInner> read_index(rust::Str filename);

} // namespace faiss_bridge
