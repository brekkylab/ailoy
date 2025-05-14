#include <format>

#include <gtest/gtest.h>

#include "mlc_llm/model_cache.hpp"

using namespace std::chrono_literals;

#if defined(USE_METAL)
std::string device = "metal";
#elif defined(USE_VULKAN)
std::string device = "vulkan";
#else
std::string device = "cpu";
#endif

TEST(ModelCacheTest, BGEM3) {
  ailoy::get_model("BAAI/bge-m3", "q4f16_1", device, std::nullopt, true);
}

TEST(ModelCacheTest, Qwen3_8B) {
  ailoy::get_model("Qwen/Qwen3-8B", "q4f16_1", device, std::nullopt, true);
}

TEST(ModelCacheTest, Qwen3_4B) {
  ailoy::get_model("Qwen/Qwen3-4B", "q4f16_1", device, std::nullopt, true);
}

TEST(ModelCacheTest, Qwen3_1_7B) {
  ailoy::get_model("Qwen/Qwen3-1.7B", "q4f16_1", device, std::nullopt, true);
}

TEST(ModelCacheTest, Qwen3_0_6B) {
  ailoy::get_model("Qwen/Qwen3-0.6B", "q4f16_1", device, std::nullopt, true);
}

TEST(ModelCacheTest, BGEM3_Callback) {
  ailoy::remove_model("BAAI/bge-m3", "q4f16_1");

  std::string current_downloading_file;
  size_t last_line_length = 0;
  auto callback = [&](const size_t current_file_idx, const size_t total_files,
                      const std::string &filename, const float progress) {
    std::ostringstream oss;
    oss << std::format("[{}/{}] Downloading {}: {}%", current_file_idx + 1,
                       total_files, filename, progress);
    std::string line = oss.str();
    if (line.length() < last_line_length) {
      line += std::string(last_line_length - line.length(), ' ');
    }
    std::cout << "\r" << line << std::flush;
    last_line_length = line.length();

    if (progress >= 100) {
      std::cout << std::endl;
      last_line_length = 0;
    }
  };
  ailoy::get_model("BAAI/bge-m3", "q4f16_1", device, callback, false);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
