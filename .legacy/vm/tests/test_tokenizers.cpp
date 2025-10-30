#ifdef EMSCRIPTEN
#include <emscripten.h>
#include <emscripten/bind.h>
#else
#include <gtest/gtest.h>
#endif

#include <format>
#include <iostream>

#include "model_cache.hpp"
#include "tokenizer.hpp"

static std::string expected_text = "What is your name?";
static std::vector<uint32_t> expected_tokens{3838, 374, 697, 829, 30};

#ifdef EMSCRIPTEN

int main() {
  auto cache_root = ailoy::get_cache_root();
  auto tokenizer_path = cache_root / "tvm-models" / "Qwen--Qwen3-0.6B" /
                        "q4f16_1" / "tokenizer.json";
  if (!ailoy::fs::file_exists(tokenizer_path).unwrap()) {
    std::cout << "tokenizer.json for Qwen/Qwen3-0.6B is not present. Download "
                 "the model first."
              << std::endl;
    return 0;
  }

  ailoy::tokenizer_t tokenizer(tokenizer_path);

  std::cout << std::format("Encode text: \"{}\"", expected_text) << std::endl;
  auto tokens = tokenizer.encode(expected_text);
  if (tokens.size() != expected_tokens.size()) {
    throw std::runtime_error("The number of tokens is not same as expected");
  }

  for (int i = 0; i < tokens.size(); i++) {
    std::cout << std::format("{}th token: {}", i, tokens[i]) << std::endl;
    if (tokens[i] != expected_tokens[i]) {
      throw std::runtime_error(
          std::format("The {}th token({}) is not same as expected({})", i,
                      tokens[i], expected_tokens[i]));
    }
  }

  auto decoded_text = tokenizer.decode(tokens);
  std::cout << std::format("Decoded text: {}", decoded_text) << std::endl;
  if (decoded_text != expected_text) {
    throw std::runtime_error(std::format(
        "The decoded text is not same as expected: \"{}\"", expected_text));
  }

  return 0;
}

#else

TEST(TokenizersTest, EncodeDecode) {
  auto cache_root = ailoy::get_cache_root();
  auto tokenizer_path = cache_root / "tvm-models" / "Qwen--Qwen3-0.6B" /
                        "q4f16_1" / "tokenizer.json";
  if (!ailoy::fs::file_exists(tokenizer_path).unwrap()) {
    GTEST_SKIP() << "tokenizer.json for Qwen/Qwen3-0.6B is not present. "
                    "Download the model first.";
    return;
  }

  ailoy::tokenizer_t tokenizer(tokenizer_path);

  auto encoded_tokens = tokenizer.encode(expected_text);
  ASSERT_EQ(encoded_tokens.size(), expected_tokens.size());
  for (int i = 0; i < encoded_tokens.size(); i++) {
    ASSERT_EQ(encoded_tokens[i], expected_tokens[i]);
  }

  auto decoded_text = tokenizer.decode(encoded_tokens);
  ASSERT_EQ(decoded_text, expected_text);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif