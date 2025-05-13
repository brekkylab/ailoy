#pragma once

#include <fstream>
#include <thread>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include <tokenizers_c.h>
#include <tvm/runtime/builtin_fp16.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/relax_vm/ndarray_cache_support.h>

#include "model_cache.hpp"
#include "module.hpp"
#include "value.hpp"

namespace ailoy {

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

} // namespace ailoy
