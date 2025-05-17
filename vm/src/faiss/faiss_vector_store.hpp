#pragma once

#include "../vector_store.hpp"

namespace ailoy {

class faiss_vector_store_impl_t;

class faiss_vector_store_t : public vector_store_t {
public:
  faiss_vector_store_t(const size_t dimension);
  faiss_vector_store_t(std::shared_ptr<const value_t> attrs);
  ~faiss_vector_store_t();

  std::string add_vector(const vector_store_add_input_t &input) override;

  std::vector<std::string>
  add_vectors(std::vector<vector_store_add_input_t> &inputs) override;

  std::optional<vector_store_get_result_t>
  get_by_id(const std::string &id) override;

  std::vector<vector_store_retrieve_result_t>
  retrieve(embedding_t query_embedding, uint64_t top_k) override;

  void remove_vector(const std::string &id) override;

  void clear() override;

private:
  faiss_vector_store_impl_t *impl_;
};

} // namespace ailoy
