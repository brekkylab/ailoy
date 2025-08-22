#include "faiss_bridge.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <faiss/impl/IDSelector.h>
#include <faiss/index_io.h>

#include "cxx_bridge.rs.h"

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

FaissMetricType FaissIndexWrapper::get_metric_type() const {
  return FaissMetricType(index_->metric_type);
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
  std::vector<faiss::idx_t> faiss_ids(ids.begin(), ids.end());
  index_->add_with_ids(static_cast<faiss::idx_t>(num_vectors), vectors.data(),
                       faiss_ids.data());
}

FaissSearchResult
FaissIndexWrapper::search_vectors(rust::Slice<const float> query_vectors,
                                  size_t k) const {
  faiss::idx_t num_queries = query_vectors.size() / index_->d;
  std::vector<float> distances_vec(num_queries * k);
  std::vector<faiss::idx_t> indexes_vec(num_queries * k);

  index_->search(num_queries, query_vectors.data(),
                 static_cast<faiss::idx_t>(k), distances_vec.data(),
                 indexes_vec.data());

  rust::Vec<float> rust_distances;
  rust::Vec<int64_t> rust_indexes;

  rust_distances.reserve(distances_vec.size());
  std::copy(distances_vec.begin(), distances_vec.end(),
            std::back_inserter(rust_distances));

  rust_indexes.reserve(indexes_vec.size());
  std::copy(indexes_vec.begin(), indexes_vec.end(),
            std::back_inserter(rust_indexes));

  return FaissSearchResult{std::move(rust_distances), std::move(rust_indexes)};
}

// rust::Vec<float> FaissIndexWrapper::get_by_id(int64_t id) const {
//   try {
//     std::vector<float> reconstructed_vector(index_->d);

//     index_->reconstruct(id, reconstructed_vector.data());

//     rust::Vec<float> result;
//     result.reserve(reconstructed_vector.size());
//     std::copy(reconstructed_vector.begin(), reconstructed_vector.end(),
//               std::back_inserter(result));

//     return result;

//   } catch (const std::exception &e) {
//     throw std::runtime_error("Failed to get vector by ID " +
//                              std::to_string(id) + ": " + e.what());
//   }
// }

rust::Vec<float>
FaissIndexWrapper::get_by_ids(rust::Slice<const int64_t> ids) const {
  if (ids.empty()) {
    return rust::Vec<float>();
  }

  try {
    size_t num_ids = ids.size();
    size_t dimension = index_->d;
    std::vector<float> reconstructed_vectors(num_ids * dimension);

    for (size_t i = 0; i < num_ids; ++i) {
      faiss::idx_t current_id = static_cast<faiss::idx_t>(ids[i]);
      float *destination = reconstructed_vectors.data() + (i * dimension);
      index_->reconstruct(current_id, destination);
    }

    rust::Vec<float> result;
    result.reserve(reconstructed_vectors.size());
    std::copy(reconstructed_vectors.begin(), reconstructed_vectors.end(),
              std::back_inserter(result));

    return result;
  } catch (const std::exception &e) {
    throw std::runtime_error("FAISS reconstruct failed: " +
                             std::string(e.what()));
  }
}

size_t FaissIndexWrapper::remove_vectors(rust::Slice<const int64_t> ids) {
  if (ids.empty()) {
    return 0;
  }

  try {
    faiss::IDSelectorBatch selector(ids.size(), ids.data());

    size_t num_removed = index_->remove_ids(selector);

    return num_removed;
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to remove vectors: " +
                             std::string(e.what()));
  }
}

void FaissIndexWrapper::clear() {
  try {
    if (index_->ntotal > 0) {
      faiss::IDSelectorRange selector(0, index_->ntotal);

      size_t num_removed = index_->remove_ids(selector);
    } else {
      // log.debug("Index is already empty.")
    }
  } catch (const std::exception &e) {
    // call faiss::Index::reset() if index doesn't support remove_ids()
    try {
      index_->reset();
    } catch (const std::exception &reset_error) {
      throw std::runtime_error(
          "Failed to clear index: " + std::string(e.what()) +
          ". Reset also failed: " + std::string(reset_error.what()));
    }
  }
}

void FaissIndexWrapper::write_index(rust::Str filename) const {
  try {
    std::string filename_str(filename);

    if (!index_) {
      throw std::runtime_error("Cannot write index: index is null");
    }

    std::filesystem::path file_path(filename_str);
    std::filesystem::path directory = file_path.parent_path();

    if (!directory.empty() && !std::filesystem::exists(directory)) {
      std::filesystem::create_directories(directory);
    }

    faiss::write_index(index_.get(), filename_str.c_str());
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to write index to file '" +
                             std::string(filename) + "': " + e.what());
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

std::unique_ptr<FaissIndexWrapper> read_index(rust::Str filename) {
  try {
    std::string filename_str(filename);

    if (!std::filesystem::exists(filename_str)) {
      throw std::runtime_error("File does not exist: " + filename_str);
    }

    std::ifstream file_check(filename_str, std::ios::binary);
    if (!file_check.is_open()) {
      throw std::runtime_error("Cannot open file for reading: " + filename_str);
    }
    file_check.close();

    std::unique_ptr<faiss::Index> loaded_index(
        faiss::read_index(filename_str.c_str()));

    if (!loaded_index) {
      throw std::runtime_error(
          "Failed to load index: read_index returned null");
    }

    return std::make_unique<FaissIndexWrapper>(std::move(loaded_index));

  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to read index from file '" +
                             std::string(filename) + "': " + e.what());
  }
}

} // namespace faiss_bridge
