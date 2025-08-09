#include "language_model.hpp"

#include <random>

#include <dlpack/dlpack.h>
#include <nlohmann/json.hpp>
#include <tvm/ffi/container/shape.h>
#include <tvm/runtime/int_tuple.h>

#include "tvm_runtime.hpp"

#include <iostream>

using namespace tvm;
using namespace tvm::runtime;

namespace ailoy {

double random_float(double min, double max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min, max);
  return dis(gen);
}

constexpr size_t page_size = 16;

kv_cache_t::kv_cache_t(tvm_runtime_t &rt) {
  auto fn = rt.get_vm_function("create_tir_paged_kv_cache");
  if (!fn.defined())
    throw std::runtime_error("create_tir_paged_kv_cache not defined");
  kv_cache_ =
      fn(IntTuple{1}, // max_num_sequence
         IntTuple{rt.get_metadata()["context_window_size"].operator int()},
         IntTuple{rt.get_metadata()["prefill_chunk_size"].operator int()},
         IntTuple{page_size}, // page size
         IntTuple{rt.get_metadata()["sliding_window_size"].operator int() !=
                  -1})
          .cast<ObjectRef>();
  fkv_state_clear_ = rt.get_function("vm.builtin.kv_state_clear");
  fkv_state_add_sequence_ = rt.get_function("vm.builtin.kv_state_add_sequence");
  fkv_state_remove_sequence_ =
      rt.get_function("vm.builtin.kv_state_remove_sequence");
  fkv_state_fork_sequence_ =
      rt.get_function("vm.builtin.kv_state_fork_sequence");
  fkv_state_begin_forward_ =
      rt.get_function("vm.builtin.kv_state_begin_forward");
  fkv_state_end_forward_ = rt.get_function("vm.builtin.kv_state_end_forward");
  fkv_state_popn_ = rt.get_function("vm.builtin.kv_state_popn");
  fkv_cache_get_num_available_pages_ =
      rt.get_function("vm.builtin.attention_kv_cache_get_num_available_pages");
  fkv_cache_get_total_sequence_length_ = rt.get_function(
      "vm.builtin.attention_kv_cache_get_total_sequence_length");

  // Register sequence index
  add_sequence();
}

void kv_cache_t::add_sequence() {
  fkv_state_add_sequence_(kv_cache_, 0 /* Sequence ID */);
}

void kv_cache_t::remove_sequence() {
  fkv_state_remove_sequence_(kv_cache_, 0 /* Sequence ID */);
}

void kv_cache_t::begin_forward(size_t sequence_length) {
  fkv_state_begin_forward_(kv_cache_, IntTuple{0 /* Sequence ID */},
                           IntTuple{static_cast<int32_t>(sequence_length)});
}

void kv_cache_t::end_forward() { fkv_state_end_forward_(kv_cache_); }

void kv_cache_t::popn(size_t num_tokens) {
  fkv_state_popn_(kv_cache_, 0 /* Sequence ID */, (int)(num_tokens));
}

int kv_cache_t::get_num_available_pages() {
  return fkv_cache_get_num_available_pages_(kv_cache_).cast<int>();
}

int kv_cache_t::get_total_sequence_length() {
  return fkv_cache_get_total_sequence_length_(kv_cache_).cast<int>();
}

tvm_language_model_t::tvm_language_model_t(const std::string &lib_path,
                                           cache_t file_contents,
                                           DLDevice device) {
  rt_ = std::make_unique<tvm_runtime_t>(lib_path, file_contents, device);
  kv_cache_ = std::make_unique<kv_cache_t>(*rt_);
  config = config_t{.temperature = 0.6, .top_p = 0.9};
  default_config_ = config;

  // Packed functions
  fembed_ = rt_->get_vm_function("embed");
  if (!fembed_.defined())
    throw std::runtime_error("Cannot find embed function");
  fprefill_ = rt_->get_vm_function("prefill");
  if (!fprefill_.defined())
    throw std::runtime_error("Cannot find embed function");
  fdecode_ = rt_->get_vm_function("decode");
  if (!fdecode_.defined())
    throw std::runtime_error("Cannot find embed function");
  fapply_bitmask_inplace_ = rt_->get_vm_function("apply_bitmask_inplace", true);
  if (!fapply_bitmask_inplace_.defined())
    throw std::runtime_error("Cannot find embed function");
  fsample_top_p_from_logits_ =
      rt_->get_function("vm.builtin.sample_top_p_from_logits");
  if (!fsample_top_p_from_logits_.defined())
    throw std::runtime_error("Cannot find embed function");
}

void tvm_language_model_t::clear() {
  kv_cache_->clear();
  history_.clear();
}

void tvm_language_model_t::prefill(const std::vector<uint32_t> &tokens) {
  if (tokens.empty())
    throw std::runtime_error("Token must not be empty");

  // Make sure that kv-cache and history is sync
  if (kv_cache_->get_total_sequence_length() != history_.size())
    this->clear();

  // The longest common prefix (LCP) between inputs & previous conversations
  size_t lcp_index = 0;
  while (lcp_index < history_.size() && lcp_index < tokens.size()) {
    if (history_[lcp_index] != tokens[lcp_index])
      break;
    ++lcp_index;
  }

  // Rewind the head of kv-cache to the LCP
  if (lcp_index < history_.size()) {
    kv_cache_->popn(history_.size() - lcp_index);
  }

  // Tokens to be added (wihout common prefixes)
  std::vector<int32_t> new_tokens(tokens.begin() + lcp_index, tokens.end());
  if (new_tokens.empty())
    return;

  // Calculate remaining space in KV cache
  if (new_tokens.size() >= kv_cache_->get_num_available_pages() * page_size)
    throw std::runtime_error("Context length limit exceed");

  // Chunk size to be split
  size_t prefill_chunk_size = rt_->get_metadata()["prefill_chunk_size"];
  for (size_t i = 0; i < new_tokens.size(); i += prefill_chunk_size) {
    // Prefill i to j
    size_t j = (i + prefill_chunk_size < new_tokens.size())
                   ? i + prefill_chunk_size
                   : new_tokens.size();
    int32_t length = j - i;
    DLDataType I32 = DLDataType{.code = kDLInt, .bits = 32, .lanes = 1};

    // Input NDArray
    NDArray input = NDArray::Empty({length}, I32, rt_->get_device());
    input.CopyFromBytes(&*(new_tokens.begin() + i), length * sizeof(int32_t));

    // Embedding of the input
    NDArray embedding = fembed_(input, rt_->get_params()).cast<NDArray>();
    NDArray embedding_reshaped = embedding.CreateView(
        tvm::ffi::Shape{1, embedding->shape[0], embedding->shape[1]},
        embedding.DataType());

    // Forward prefill
    kv_cache_->begin_forward(length);
    fprefill_(embedding_reshaped, kv_cache_->get(), rt_->get_params());
    kv_cache_->end_forward();
  }

  // Update history
  history_ = tokens;
}

NDArray tvm_language_model_t::decode(uint32_t last_token) {
  DLDataType U32 = DLDataType{.code = kDLUInt, .bits = 32, .lanes = 1};
  DLDataType I32 = DLDataType{.code = kDLInt, .bits = 32, .lanes = 1};
  DLDataType F32 = DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1};

  // Calculate remaining space in KV cache
  if (kv_cache_->get_num_available_pages() < 1) {
    throw std::runtime_error("Context length limit exceed");
  }

  // Input NDArray
  NDArray token_ids = NDArray::Empty({1}, I32, rt_->get_device());
  token_ids.CopyFromBytes(&last_token, sizeof(int32_t));

  // Embed
  NDArray embed = rt_->get_vm_function("embed")(token_ids, rt_->get_params())
                      .cast<NDArray>();
  tvm::runtime::NDArray embed_reshaped = embed.CreateView(
      tvm::ffi::Shape{1, 1, embed->shape[1]}, embed.DataType());

  // In decode, the sequence length of new tokens are always 1
  kv_cache_->begin_forward(1);
  // Forward decode (output: [logits, kv_caches])
  ObjectRef output =
      fdecode_(embed_reshaped, kv_cache_->get(), rt_->get_params())
          .cast<ObjectRef>();
  kv_cache_->end_forward();

  // Extract logits (1 x seq_len x vocab_size)
  // Note that the seq_len is the ID of the seqence, used for decoding multiple
  // context in parallel. In our cases, it always set to 1.
  NDArray logits = Downcast<Array<NDArray>>(output)[0];
  return logits;
}

uint32_t tvm_language_model_t::sample(NDArray logits) {
  // Sample token from logits
  uint32_t sampled_token =
      fsample_top_p_from_logits_(logits, config.temperature, config.top_p,
                                 random_float(0.0, 1.0))
          .cast<int32_t>();

  // Register it to history
  history_.push_back(sampled_token);

  return sampled_token;
}

std::unique_ptr<tvm_language_model_t>
create_tvm_language_model(rust::String lib_filename,
                          std::unique_ptr<cache_t> cache_content,
                          std::unique_ptr<DLDevice> device) {
  return std::make_unique<tvm_language_model_t>(
      std::string(lib_filename), std::move(*cache_content), std::move(*device));
}

std::unique_ptr<DLDevice> create_dldevice(int device_type, int device_id) {
  auto dev = std::make_unique<DLDevice>();
  dev->device_type = static_cast<DLDeviceType>(device_type);
  dev->device_id = device_id;
  return dev;
}

} // namespace ailoy
