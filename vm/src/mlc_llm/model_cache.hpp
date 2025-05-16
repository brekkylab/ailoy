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

std::tuple<std::filesystem::path, std::filesystem::path>
get_model(const std::string &model_name, const std::string &quantization,
          const std::string &target_device,
          std::optional<model_cache_callback_t> callback = std::nullopt,
          bool print_progress_bar = false);

void remove_model(const std::string &model_name,
                  const std::string &quantization);

} // namespace ailoy