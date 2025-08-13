#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <rust/cxx.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/ndarray.h>

#include "tvm_runtime.hpp"

namespace ailoy {

struct CacheContents;

struct kv_cache_t {
public:
  kv_cache_t(tvm_runtime_t &rt);

  ~kv_cache_t() { remove_sequence(); }

  tvm::ObjectRef get() { return kv_cache_; }

  void clear() {
    fkv_state_clear_(kv_cache_);
    add_sequence();
  }

  void add_sequence();

  void remove_sequence();

  void begin_forward(size_t sequence_length);

  void end_forward();

  void popn(size_t num_tokens);

  int get_num_available_pages();

  int get_total_sequence_length();

private:
  tvm::ObjectRef kv_cache_;

  tvm::ffi::Function fkv_state_clear_;

  tvm::ffi::Function fkv_state_add_sequence_;

  tvm::ffi::Function fkv_state_fork_sequence_;

  tvm::ffi::Function fkv_state_remove_sequence_;

  tvm::ffi::Function fkv_state_begin_forward_;

  tvm::ffi::Function fkv_state_end_forward_;

  tvm::ffi::Function fkv_state_popn_;

  tvm::ffi::Function fkv_cache_get_num_available_pages_;

  tvm::ffi::Function fkv_cache_get_total_sequence_length_;
};

class tvm_language_model_t {
public:
  struct config_t {
    double temperature;
    double top_p;
  };

  /**
   * Constructor
   */
  tvm_language_model_t(ailoy::CacheContents &contents, DLDevice device);

  void clear();

  /** Prefill */
  void prefill(const std::vector<uint32_t> &tokens);

  void prefill_from_rs(const rust::Vec<uint32_t> &tokens) {
    std::lock_guard<std::mutex> lk(m_);
    std::vector<uint32_t> converted(tokens.begin(), tokens.end());
    return prefill(converted);
  }

  /** Decode */
  tvm::runtime::NDArray decode(uint32_t last_token);

  std::unique_ptr<DLManagedTensorVersioned>
  decode_from_rs(uint32_t last_token) {
    std::lock_guard<std::mutex> lk(m_);
    return std::unique_ptr<DLManagedTensorVersioned>(
        std::move(decode(last_token)).ToDLPackVersioned());
  }

  /** Sample */
  uint32_t sample(tvm::runtime::NDArray);

  uint32_t sample_from_rs(std::unique_ptr<DLManagedTensorVersioned> logits) {
    std::lock_guard<std::mutex> lk(m_);
    auto v =
        tvm::runtime::NDArray::FromDLPackVersioned(std::move(logits.release()));
    return sample(v);
  }

  config_t config;

  const config_t &get_default_config() const { return default_config_; }

private:
  std::unique_ptr<tvm_runtime_t> rt_;

  std::unique_ptr<kv_cache_t> kv_cache_;

  config_t default_config_;

  std::vector<uint32_t> history_;

  tvm::ffi::Function fembed_;

  tvm::ffi::Function fprefill_;

  tvm::ffi::Function fdecode_;

  tvm::ffi::Function fapply_bitmask_inplace_;

  tvm::ffi::Function fsample_top_p_from_logits_;

  mutable std::mutex m_;
};

std::unique_ptr<DLDevice> create_dldevice(int device_type, int device_id);

std::unique_ptr<tvm_language_model_t>
create_tvm_language_model(ailoy::CacheContents &contents,
                          std::unique_ptr<DLDevice> device);

} // namespace ailoy
