#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include <rust/cxx.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/tensor.h>

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
  /**
   * Constructor
   */
  tvm_language_model_t(ailoy::CacheContents &contents, DLDevice device);

  void clear();

  /** Prefill */
  void prefill(const std::vector<uint32_t> &tokens);

  void prefill_from_rs(rust::Slice<const uint32_t> tokens);

  /** Decode */
  tvm::runtime::Tensor decode(uint32_t last_token);

  DLPackTensor decode_from_rs(uint32_t last_token);

  /** Sample */
  uint32_t sample(tvm::runtime::Tensor, double temperature, double top_p);

  uint32_t sample_from_rs(DLPackTensor logits, double temperature,
                          double top_p);

private:
  std::unique_ptr<tvm_runtime_t> rt_;

  std::unique_ptr<kv_cache_t> kv_cache_;

  std::vector<uint32_t> history_;

  tvm::ffi::Function fembed_;

  tvm::ffi::Function fprefill_;

  tvm::ffi::Function fdecode_;

  tvm::ffi::Function fapply_bitmask_inplace_;

  tvm::ffi::Function fsample_top_p_from_logits_;
};

std::unique_ptr<tvm_language_model_t>
create_tvm_language_model(ailoy::CacheContents &contents,
                          std::unique_ptr<DLDevice> device);

} // namespace ailoy
