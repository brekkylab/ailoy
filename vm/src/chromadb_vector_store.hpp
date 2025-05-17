#pragma once

#include <httplib.h>

#include "vector_store.hpp"

namespace ailoy {

const std::string CHROMADB_DEFAULT_URL = "http://localhost:8000";
const std::string CHROMADB_DEFAULT_COLLECTION = "default_collection";

class chromadb_vector_store_t : public vector_store_t {
public:
  chromadb_vector_store_t(
      const std::string &url = CHROMADB_DEFAULT_URL,
      const std::string &collection = CHROMADB_DEFAULT_COLLECTION,
      bool delete_collection_on_cleanup = false);
  chromadb_vector_store_t(std::shared_ptr<const value_t> attrs);
  ~chromadb_vector_store_t();

  void _create_collection();

  void _delete_collection();

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
  std::shared_ptr<httplib::Client> cli_;
  std::string collection_id_;
  std::string collection_name_;
  bool delete_collection_on_cleanup_;
};

} // namespace ailoy