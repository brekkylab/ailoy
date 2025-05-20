#pragma once

#include <filesystem>
#include <functional>
#include <optional>

#include <nlohmann/json.hpp>

#include "module.hpp"

namespace ailoy {

using model_cache_callback_t =
    std::function<void(const size_t,        // current_file_idx
                       const size_t,        // total_files
                       const std::string &, // current filename
                       const float          // current_file_progress
                       )>;

struct model_cache_list_result_t {
  std::string model_type;
  std::string model_id;
  nlohmann::json attributes;
  std::filesystem::path model_path;
  size_t total_bytes;
};

struct model_cache_download_result_t {
  bool success;
  std::optional<std::filesystem::path> model_path = std::nullopt;
  std::optional<std::filesystem::path> model_lib_path = std::nullopt;
  std::optional<std::string> error_message = std::nullopt;
};

struct model_cache_remove_result_t {
  bool success;
  bool skipped = false;
  std::optional<std::filesystem::path> model_path = std::nullopt;
  std::optional<std::string> error_message = std::nullopt;
};

/**
 * @brief Get the cache root directory. If the AILOY_CACHE_ROOT environment
 * variable is set, it will be used; otherwise, the default path will be used.
 */
std::filesystem::path get_cache_root();

std::vector<model_cache_list_result_t> list_local_models();

model_cache_download_result_t
download_model(const std::string &model_id, const std::string &quantization,
               const std::string &target_device,
               std::optional<model_cache_callback_t> callback = std::nullopt,
               bool print_progress_bar = true);

model_cache_remove_result_t remove_model(const std::string &model_name,
                                         bool ask_prompt = false);

namespace operators {

value_or_error_t list_local_models(std::shared_ptr<const value_t> inputs);

value_or_error_t download_model(std::shared_ptr<const value_t> inputs);

value_or_error_t remove_model(std::shared_ptr<const value_t> inputs);

} // namespace operators

} // namespace ailoy
