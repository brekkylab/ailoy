#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <tvm/ffi/function.h>
#include <tvm/runtime/ndarray.h>

#include "tvm_runtime.hpp"

namespace ailoy {

class tvm_model_t;

class chat_manager_t;

class tokenizer_t;

struct kv_cache_t;

struct tokenizer_info_t;

struct grammar_t;

struct grammar_matcher_t;

class tvm_language_model_t {
public:
  struct config_t {
    double temperature;
    double top_p;
  };

  /**
   * Constructor
   */
  tvm_language_model_t(
      const std::string &lib_path,
      std::unordered_map<std::string, std::string> &file_contents,
      DLDevice device);

  void clear();

  /** Prefill */
  int32_t prefill(const std::vector<int32_t> &tokens);

  /** Decode */
  int32_t decode(int32_t last_token);

  config_t config;

  const config_t &get_default_config() const { return default_config_; }

private:
  std::unique_ptr<tvm_runtime_t> rt_;

  std::unique_ptr<kv_cache_t> kv_cache_;

  config_t default_config_;

  std::vector<int32_t> history_;

  std::vector<int32_t> output_stream_;

  tvm::ffi::Function fembed_;

  tvm::ffi::Function fprefill_;

  tvm::ffi::Function fdecode_;

  tvm::ffi::Function fapply_bitmask_inplace_;

  tvm::ffi::Function fsample_top_p_from_logits_;
};

} // namespace ailoy
