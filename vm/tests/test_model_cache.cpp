#include <csignal>
#include <format>
#include <iostream>
#include <thread>

#ifndef EMSCRIPTEN
#include <gtest/gtest.h>
#endif

#include "filesystem.hpp"
#include "model_cache.hpp"

using namespace std::chrono_literals;

#ifndef EMSCRIPTEN

#if defined(USE_METAL)
std::string device = "metal";
#elif defined(USE_VULKAN)
std::string device = "vulkan";
#else
std::string device = "cpu";
#endif

TEST(ModelCacheTest, BGEM3) {
  ailoy::download_model("BAAI/bge-m3", "q4f16_1", device, std::nullopt, true);
}

TEST(ModelCacheTest, Qwen3_8B) {
  ailoy::download_model("Qwen/Qwen3-8B", "q4f16_1", device, std::nullopt, true);
}

TEST(ModelCacheTest, Qwen3_4B) {
  ailoy::download_model("Qwen/Qwen3-4B", "q4f16_1", device, std::nullopt, true);
}

TEST(ModelCacheTest, Qwen3_1_7B) {
  ailoy::download_model("Qwen/Qwen3-1.7B", "q4f16_1", device, std::nullopt,
                        true);
}

TEST(ModelCacheTest, Qwen3_0_6B) {
  ailoy::download_model("Qwen/Qwen3-0.6B", "q4f16_1", device, std::nullopt,
                        true);
}

TEST(ModelCacheTest, SigintWhileGetModel) {
  auto remove_result = ailoy::remove_model("BAAI/bge-m3");
  if (!remove_result.success) {
    throw std::runtime_error(remove_result.error_message.value());
  }

  ailoy::model_cache_download_result_t result;

  auto t1 = std::thread([&]() {
    result = ailoy::download_model("BAAI/bge-m3", "q4f16_1", device,
                                   std::nullopt, true);
  });
  auto t2 = std::thread([]() {
    std::this_thread::sleep_for(3s);
    std::raise(SIGINT);
  });

  t1.join();
  t2.join();

  ASSERT_EQ(result.success, false);
  ASSERT_EQ(result.error_message.value(),
            "Interrupted while downloading the model");
}

TEST(ModelCacheTest, BGEM3_Callback) {
  auto remove_result = ailoy::remove_model("BAAI/bge-m3");
  if (!remove_result.success) {
    throw std::runtime_error(remove_result.error_message.value());
  }

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
  ailoy::download_model("BAAI/bge-m3", "q4f16_1", device, callback, false);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#else

int main() {
  ailoy::remove_model("Qwen/Qwen3-0.6B");

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

  ailoy::download_model("Qwen/Qwen3-0.6B", "q4f16_1", "webgpu", callback,
                        false);

  std::function<void(ailoy::fs::dir_entry_t, int)> print_directory_entries =
      [&](ailoy::fs::dir_entry_t entry, int tab) {
        std::cout << std::string(tab, ' ') << entry.name
                  << (entry.is_directory() ? "/" : "")
                  << (entry.is_regular_file()
                          ? "\t" + std::to_string(entry.size) + "B"
                          : "")
                  << std::endl;
        if (entry.is_directory()) {
          auto entries = ailoy::fs::list_directory(entry.path).unwrap();
          for (const auto &subentry : entries) {
            print_directory_entries(subentry, tab + 2);
          }
        }
      };

  ailoy::fs::path_t base_dir = "/ailoy";
  auto list_result = ailoy::fs::list_directory(base_dir);
  std::cout << base_dir.string() << "/" << std::endl;
  for (const auto &info : list_result.unwrap()) {
    print_directory_entries(info, 2);
  }

  return 0;
}

#endif