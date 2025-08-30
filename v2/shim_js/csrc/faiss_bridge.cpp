#include <stdexcept>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <faiss/impl/IDSelector.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

using namespace emscripten;

enum class FaissMetricType : uint8_t {
  METRIC_INNER_PRODUCT = 0,
  METRIC_L2 = 1,
  METRIC_L1 = 2,
  METRIC_Linf = 3,
  METRIC_Lp = 4,
  METRIC_Canberra = 20,
  METRIC_BrayCurtis = 21,
  METRIC_JensenShannon = 22,
  METRIC_Jaccard = 23
};

struct FaissIndexSearchResult {
  std::vector<float> distances;
  std::vector<int64_t> ids;
};

class FaissIndexWrapper {
private:
  std::unique_ptr<faiss::Index> index_;

public:
  explicit FaissIndexWrapper(std::unique_ptr<faiss::Index> index)
      : index_(std::move(index)) {
    if (!index_) {
      throw std::runtime_error("FaissIndexWrapper: null index provided");
    }
  }

  ~FaissIndexWrapper() = default;

  // No copy constructors
  FaissIndexWrapper(const FaissIndexWrapper &) = delete;
  FaissIndexWrapper &operator=(const FaissIndexWrapper &) = delete;

  // Move constructors
  FaissIndexWrapper(FaissIndexWrapper &&) = default;
  FaissIndexWrapper &operator=(FaissIndexWrapper &&) = default;

  bool is_trained() const { return index_->is_trained; }

  int64_t get_ntotal() const { return index_->ntotal; }

  int32_t get_dimension() const { return static_cast<int32_t>(index_->d); }

  FaissMetricType get_metric_type() const {
    return FaissMetricType(index_->metric_type);
  }

  void train_index(const val &training_vectors_js,
                   size_t num_training_vectors) {
    if (index_->is_trained)
      return; // skip training

    // Convert JavaScript Float32Array to C++ vector
    std::vector<float> training_vectors =
        convertTypedArray<float>(training_vectors_js);

    index_->train(static_cast<faiss::idx_t>(num_training_vectors),
                  training_vectors.data());
  }

  void add_vectors_with_ids(const val &vectors_js, size_t num_vectors,
                            const val &ids_js) {
    // Convert JavaScript typed arrays to C++ vectors
    std::vector<float> vectors = convertTypedArray<float>(vectors_js);
    std::vector<int64_t> ids_temp = convertTypedArray<int64_t>(ids_js);

    std::vector<faiss::idx_t> faiss_ids(ids_temp.begin(), ids_temp.end());
    index_->add_with_ids(static_cast<faiss::idx_t>(num_vectors), vectors.data(),
                         faiss_ids.data());
  }

  FaissIndexSearchResult search_vectors(const val &query_vectors_js,
                                        size_t k) const {
    std::vector<float> query_vectors =
        convertTypedArray<float>(query_vectors_js);

    faiss::idx_t num_queries = query_vectors.size() / index_->d;
    std::vector<float> distances_vec(num_queries * k);
    std::vector<faiss::idx_t> ids_vec(num_queries * k);

    index_->search(num_queries, query_vectors.data(),
                   static_cast<faiss::idx_t>(k), distances_vec.data(),
                   ids_vec.data());

    std::vector<int64_t> result_ids(ids_vec.begin(), ids_vec.end());

    return FaissIndexSearchResult{std::move(distances_vec),
                                  std::move(result_ids)};
  }

  std::vector<float> get_by_ids(const val &ids_js) const {
    std::vector<int64_t> ids = convertTypedArray<int64_t>(ids_js);

    if (ids.empty()) {
      return std::vector<float>();
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

      return reconstructed_vectors;
    } catch (const std::exception &e) {
      throw std::runtime_error("FAISS reconstruct failed: " +
                               std::string(e.what()));
    }
  }

  size_t remove_vectors(const val &ids_js) {
    std::vector<int64_t> ids = convertTypedArray<int64_t>(ids_js);

    if (ids.empty()) {
      return 0;
    }

    try {
      // Convert int64_t to faiss::idx_t for IDSelectorBatch
      std::vector<faiss::idx_t> faiss_ids(ids.begin(), ids.end());
      faiss::IDSelectorBatch selector(faiss_ids.size(), faiss_ids.data());

      size_t num_removed = index_->remove_ids(selector);
      return num_removed;
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to remove vectors: " +
                               std::string(e.what()));
    }
  }

  void clear() {
    try {
      if (index_->ntotal > 0) {
        faiss::IDSelectorAll all_selector;
        size_t num_removed = index_->remove_ids(all_selector);
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

  // TODO: Handle write_index (filesystem should be determined)
  //   void write_index(const std::string &filename) const {
  //     try {
  //       if (!index_) {
  //         throw std::runtime_error("Cannot write index: index is null");
  //       }

  //       std::filesystem::path file_path(filename);
  //       std::filesystem::path directory = file_path.parent_path();

  //       if (!directory.empty() && !std::filesystem::exists(directory)) {
  //         std::filesystem::create_directories(directory);
  //       }

  //       faiss::write_index(index_.get(), filename.c_str());
  //     } catch (const std::exception &e) {
  //       throw std::runtime_error("Failed to write index to file '" + filename
  //       +
  //                                "': " + e.what());
  //     }
  //   }

private:
  // Helper function to convert JavaScript typed arrays to C++ vectors
  template <typename T>
  std::vector<T> convertTypedArray(const val &js_array) const {
    std::vector<T> result;

    // Get the length of the JavaScript array
    int length = js_array["length"].as<int>();
    result.reserve(length);

    // Copy data from JavaScript array to C++ vector
    for (int i = 0; i < length; ++i) {
      result.push_back(js_array[i].as<T>());
    }

    return result;
  }
};

// Factory functions
std::unique_ptr<FaissIndexWrapper> create_index(int32_t dimension,
                                                const std::string &description,
                                                FaissMetricType metric) {
  try {
    faiss::MetricType faiss_metric = faiss::MetricType(metric);

    auto faiss_index = std::unique_ptr<faiss::Index>(
        faiss::index_factory(dimension, description.c_str(), faiss_metric));

    return std::make_unique<FaissIndexWrapper>(std::move(faiss_index));
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to create FAISS index: " +
                             std::string(e.what()));
  }
}

// TODO: Handle read_index (filesystem should be determined)
// std::unique_ptr<FaissIndexWrapper> read_index(const std::string &filename) {
//   try {
//     if (!std::filesystem::exists(filename)) {
//       throw std::runtime_error("File does not exist: " + filename);
//     }

//     std::ifstream file_check(filename, std::ios::binary);
//     if (!file_check.is_open()) {
//       throw std::runtime_error("Cannot open file for reading: " + filename);
//     }
//     file_check.close();

//     std::unique_ptr<faiss::Index> loaded_index(
//         faiss::read_index(filename.c_str()));

//     if (!loaded_index) {
//       throw std::runtime_error(
//           "Failed to load index: read_index returned null");
//     }

//     return std::make_unique<FaissIndexWrapper>(std::move(loaded_index));

//   } catch (const std::exception &e) {
//     throw std::runtime_error("Failed to read index from file '" + filename +
//                              "': " + e.what());
//   }
// }

// Emscripten bindings
EMSCRIPTEN_BINDINGS(faiss_bridge) {
  // Bind the FaissMetricType enum
  enum_<FaissMetricType>("FaissMetricType")
      .value("METRIC_INNER_PRODUCT", FaissMetricType::METRIC_INNER_PRODUCT)
      .value("METRIC_L2", FaissMetricType::METRIC_L2)
      .value("METRIC_L1", FaissMetricType::METRIC_L1)
      .value("METRIC_Linf", FaissMetricType::METRIC_Linf)
      .value("METRIC_Lp", FaissMetricType::METRIC_Lp)
      .value("METRIC_Canberra", FaissMetricType::METRIC_Canberra)
      .value("METRIC_BrayCurtis", FaissMetricType::METRIC_BrayCurtis)
      .value("METRIC_JensenShannon", FaissMetricType::METRIC_JensenShannon)
      .value("METRIC_Jaccard", FaissMetricType::METRIC_Jaccard);

  // Bind the search result structure
  value_object<FaissIndexSearchResult>("FaissIndexSearchResult")
      .field("distances", &FaissIndexSearchResult::distances)
      .field("ids", &FaissIndexSearchResult::ids);

  // Bind the main wrapper class
  class_<FaissIndexWrapper>("FaissIndexWrapper")
      .function("is_trained", &FaissIndexWrapper::is_trained)
      .function("get_ntotal", &FaissIndexWrapper::get_ntotal)
      .function("get_dimension", &FaissIndexWrapper::get_dimension)
      .function("get_metric_type", &FaissIndexWrapper::get_metric_type)
      .function("train_index", &FaissIndexWrapper::train_index)
      .function("add_vectors_with_ids",
                &FaissIndexWrapper::add_vectors_with_ids)
      .function("search_vectors", &FaissIndexWrapper::search_vectors)
      .function("get_by_ids", &FaissIndexWrapper::get_by_ids)
      .function("remove_vectors", &FaissIndexWrapper::remove_vectors)
      .function("clear", &FaissIndexWrapper::clear);

  // Bind factory functions
  function("create_index", &create_index, allow_raw_pointers());

  // Register std::vector types for automatic conversion
  register_vector<float>("VectorFloat");
  register_vector<int64_t>("VectorInt64");
}
