#pragma once

#include <filesystem>
#include <functional>
#include <optional>

namespace ailoy {

using model_cache_callback_t =
    std::function<void(const size_t,        // current_file_idx
                       const size_t,        // total_files
                       const std::string &, // current filename
                       const float          // current_file_progress
                       )>;

struct model_cache_get_result_t {
  bool success;
  std::optional<std::filesystem::path> model_path = std::nullopt;
  std::optional<std::filesystem::path> model_lib_path = std::nullopt;
  std::optional<std::string> error_message = std::nullopt;
};

/**
 * @brief Get the cache root directory. If the AILOY_CACHE_ROOT environment
 * variable is set, it will be used; otherwise, the default path will be used.
 */
std::filesystem::path get_cache_root();

model_cache_get_result_t
get_model(const std::string &model_name, const std::string &quantization,
          const std::string &target_device,
          std::optional<model_cache_callback_t> callback = std::nullopt,
          bool print_progress_bar = true);

void remove_model(const std::string &model_name,
                  const std::string &quantization);

} // namespace ailoy