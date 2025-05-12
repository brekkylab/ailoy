#pragma once

#include <nlohmann/json.hpp>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/relax_vm/ndarray_cache_support.h>

#include "model_cache.hpp"
#include "module.hpp"
#include "value.hpp"

namespace ailoy {

std::shared_ptr<ndarray_t> ndarray_from_tvm(tvm::runtime::NDArray ndarray);

class tokenizer_t : public object_t {
  using token_t = int32_t;

public:
  tokenizer_t(const std::filesystem::path &json_file_path);

  std::vector<token_t> encode(const std::string &text,
                              bool add_special_token = true);

  std::string decode(const std::vector<token_t> &ids,
                     bool skip_special_tokens = true);

private:
  /**
   * @brief tokenizer handle
   */
  void *handle_;
  // std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

class tvm_model_t {
public:
  tvm_model_t(const std::string &model_name, const std::string &quantization,
              DLDevice device);

  tvm::runtime::Module get_module() const {
    if (!mod_.defined())
      throw std::runtime_error("VM not created yet");
    return mod_;
  }

  const nlohmann::json &get_metadata() const { return metadata_; }

  tvm::runtime::PackedFunc get_vm_function(const std::string_view fname) {
    return get_module().GetFunction(std::string(fname));
  }

  tvm::runtime::ObjectRef get_params() const { return params_; }

  DLDevice get_device() const { return device_; }

  std::filesystem::path get_model_path() const { return model_path_; }

private:
  void load_ndarray_cache_metadata(const std::string &bytes);

  void load_ndarray_cache_shard(const size_t &shard_idx,
                                const std::string &bytes);

  void load_params_from_cache();

  std::string model_name_;
  std::string quantization_;
  DLDevice device_;

  std::filesystem::path model_path_;
  tvm::runtime::Module mod_;
  nlohmann::json metadata_ = {};
  tvm::runtime::relax_vm::NDArrayCacheMetadata ndarray_cache_metadata_;
  tvm::runtime::ObjectRef params_;

  std::string err_;
};

class tvm_embedding_model_t : public object_t {
public:
  tvm_embedding_model_t(const std::string &model_name,
                        const std::string &quantization);

  void postprocess_embedding_ndarray(const tvm::runtime::NDArray &from,
                                     tvm::runtime::NDArray &to);

  const tvm::runtime::NDArray infer(std::vector<int> tokens);

  std::filesystem::path get_model_path() const {
    return engine_->get_model_path();
  }

private:
  std::shared_ptr<tvm_model_t> prepare_engine(const std::string &model_name,
                                              const std::string &quantization) {
    // Parse device
// @jhlee: TODO implement other environment
#if defined(USE_METAL)
    auto device = DLDevice{kDLMetal, 0};
#elif defined(USE_VULKAN)
    auto device = DLDevice{kDLVulkan, 0};
#else
    auto device = DLDevice{kDLCPU, 0};
#endif
    return create<tvm_model_t>(model_name, quantization, device);
  }

  tvm::runtime::PackedFunc fprefill_;
  std::shared_ptr<tvm_model_t> engine_ = nullptr;
};

component_or_error_t
create_tvm_embedding_model_component(std::shared_ptr<const value_t> attrs);

} // namespace ailoy
