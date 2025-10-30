#include "faiss_vector_store.hpp"

#include <atomic>
#include <unordered_map>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/impl/FaissException.h>
#include <nlohmann/json.hpp>

namespace ailoy {

static std::shared_ptr<module_t> faiss_vector_store_module =
    ailoy::create<module_t>();

struct document_store_t {
  std::string document;
  std::optional<nlohmann::json> metadata = std::nullopt;
};

class faiss_vector_store_impl_t {
public:
  faiss_vector_store_impl_t(const size_t dimension) {
    index_flat_ = faiss::IndexFlatIP(dimension);
    index_ = faiss::IndexIDMap2(&index_flat_);
  }

  std::string add_vector(const vector_store_add_input_t &input) {
    if (!is_valid_embedding(input.embedding)) {
      throw ailoy::runtime_error("[FAISS] invalid embedding shape: " +
                                 input.embedding->shape_str());
    }

    int64_t id = id_counter_.fetch_add(1, std::memory_order_relaxed);
    auto vec = input.embedding->operator std::vector<float>();
    index_.add_with_ids(1, vec.data(), &id);
    document_store_[id] = document_store_t{.document = input.document,
                                           .metadata = input.metadata};
    return std::to_string(id);
  }

  std::vector<std::string>
  add_vectors(std::vector<vector_store_add_input_t> inputs) {
    // check shape of every embeddings first
    for (const auto &input : inputs) {
      if (!is_valid_embedding(input.embedding)) {
        throw ailoy::runtime_error("[FAISS] invalid embedding shape: " +
                                   input.embedding->shape_str());
      }
    }

    // insert embeddings to index
    std::vector<int64_t> ids;
    std::vector<float> embeddings(index_.d * inputs.size());
    size_t idx = 0;
    for (const auto &input : inputs) {
      int64_t id = id_counter_.fetch_add(1, std::memory_order_relaxed);
      ids.push_back(id);

      // concatenate each embedding to embeddings
      auto vec = input.embedding->operator std::vector<float>();
      memcpy(embeddings.data() + (idx * index_.d), vec.data(),
             index_.d * sizeof(float));

      idx++;
    }

    index_.add_with_ids(inputs.size(), embeddings.data(), ids.data());

    // insert documents and metadatas
    for (size_t i = 0; i < inputs.size(); i++) {
      document_store_[ids[i]] = document_store_t{
          .document = inputs[i].document,
          .metadata = inputs[i].metadata,
      };
    }

    std::vector<std::string> string_ids(ids.size());
    std::transform(ids.begin(), ids.end(), string_ids.begin(),
                   [](int64_t i) { return std::to_string(i); });
    return string_ids;
  }

  std::optional<vector_store_get_result_t> get_by_id(const std::string &id) {
    int64_t _id = std::strtoll(id.c_str(), nullptr, 10);
    std::vector<float> vec(index_.d);
    try {
      index_.reconstruct(_id, vec.data());
    } catch (faiss::FaissException e) {
      return std::nullopt;
    }

    auto ndarray = ailoy::create<ailoy::ndarray_t>();
    ndarray->shape.push_back(index_.d);
    ndarray->dtype = {.code = kDLFloat, .bits = 32, .lanes = 1};
    ndarray->data.resize(sizeof(float) * vec.size());
    memcpy(ndarray->data.data(), vec.data(), sizeof(float) * vec.size());

    return vector_store_get_result_t{
        .id = id,
        .document = document_store_[_id].document,
        .metadata = document_store_[_id].metadata,
        .embedding = std::move(ndarray),
    };
  }

  std::vector<vector_store_retrieve_result_t>
  retrieve(std::shared_ptr<const ailoy::ndarray_t> query_embedding,
           uint64_t top_k) {
    if (!is_valid_embedding(query_embedding)) {
      throw ailoy::runtime_error("[FAISS] invalid query embedding shape: " +
                                 query_embedding->shape_str());
    }

    faiss::idx_t min_k =
        std::min(index_.ntotal, static_cast<faiss::idx_t>(top_k));
    if (min_k == 0) {
      return {};
    }

    std::vector<float> similarities(min_k);
    std::vector<faiss::idx_t> ids(min_k);
    auto vec = query_embedding->operator std::vector<float>();

    index_.search(1, vec.data(), min_k, similarities.data(), ids.data());

    std::vector<vector_store_retrieve_result_t> results(min_k);
    for (int i = 0; i < min_k; i++) {
      faiss::idx_t id = ids[i];
      results[i] = {.id = std::to_string(id),
                    .document = document_store_[id].document,
                    .metadata = document_store_[id].metadata,
                    .similarity = similarities[i]};
    }

    return results;
  }

  void remove_vector(const std::string &id) {
    int64_t _id = std::strtoll(id.c_str(), nullptr, 10);
    index_.remove_ids(faiss::IDSelectorArray(1, &_id));
    document_store_.erase(_id);
  }

  void clear() {
    index_.remove_ids(faiss::IDSelectorAll());
    document_store_.clear();
  }

private:
  bool is_valid_embedding(std::shared_ptr<const ailoy::ndarray_t> embedding) {
    // should be 1-D array with length same as dimension
    return embedding->shape.size() == 1 && embedding->shape[0] == index_.d;
  }

  faiss::IndexFlatIP index_flat_;
  faiss::IndexIDMap2 index_;
  std::atomic<int64_t> id_counter_{0};
  std::unordered_map<int64_t, document_store_t> document_store_;
};

faiss_vector_store_t::faiss_vector_store_t(const size_t dimension) {
  impl_ = new faiss_vector_store_impl_t(dimension);
}

faiss_vector_store_t::faiss_vector_store_t(
    std::shared_ptr<const value_t> attrs) {
  auto dimension = attrs->as<map_t>()->at<uint_t>("dimension");
  impl_ = new faiss_vector_store_impl_t(*dimension);
}

faiss_vector_store_t::~faiss_vector_store_t() {
  if (impl_) {
    delete impl_;
    impl_ = nullptr;
  }
}

std::string
faiss_vector_store_t::add_vector(const vector_store_add_input_t &input) {
  return impl_->add_vector(input);
}

std::vector<std::string> faiss_vector_store_t::add_vectors(
    std::vector<vector_store_add_input_t> &inputs) {
  return impl_->add_vectors(inputs);
}

std::optional<vector_store_get_result_t>
faiss_vector_store_t::get_by_id(const std::string &id) {
  return impl_->get_by_id(id);
}

std::vector<vector_store_retrieve_result_t> faiss_vector_store_t::retrieve(
    std::shared_ptr<const ailoy::ndarray_t> query_embedding, uint64_t k) {
  return impl_->retrieve(query_embedding, k);
}

void faiss_vector_store_t::remove_vector(const std::string &id) {
  impl_->remove_vector(id);
}

void faiss_vector_store_t::clear() { impl_->clear(); }

} // namespace ailoy
