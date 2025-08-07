#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

#include "tvm_runtime.hpp"

TEST(TestTVM, HelloWorld) {
  const char *home = std::getenv("HOME");
  ASSERT_NE(home, nullptr) << "HOME environment variable not set";

  std::filesystem::path model_path =
      std::filesystem::path(home) / ".cache" / "ailoy";
  auto lib_filename = model_path /
                      "Qwen--Qwen3-0.6B--aarch64-apple-darwin--metal" /
                      "lib.dylib";
  // const std::unordered_map<std::string, std::string> contents;
  // ailoy_tvm_runtime_create();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
