#pragma once

#include <optional>

#include <nlohmann/json.hpp>
#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/vm/ndarray_cache_support.h>

#include "filesystem.hpp"
#include "logging.hpp"
#include "module.hpp"
#include "value.hpp"

namespace ailoy {

std::shared_ptr<ndarray_t> ndarray_from_tvm(tvm::runtime::NDArray ndarray);

bool tvm_device_exist(DLDevice device);

std::optional<DLDevice> get_tvm_device(int32_t device_id);

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

  const nlohmann::json &get_mlc_chat_config() const { return mlc_chat_config_; }

  tvm::ffi::Function get_function(const std::string_view fname) {
    return *tvm::ffi::Function::GetGlobal(std::string(fname));
  }

  tvm::ffi::Function get_vm_function(const std::string_view fname,
                                     bool query_imports = false) {
    return get_module().GetFunction(std::string(fname), query_imports);
  }

  tvm::runtime::ObjectRef get_params() const { return params_; }

  DLDevice get_device() const { return device_; }

  ailoy::fs::path_t get_model_path() const { return model_path_; }

private:
  void load_ndarray_cache_shard(const size_t &shard_idx,
                                const std::string &bytes);

  void load_params_from_cache();

  std::string model_name_;
  std::string quantization_;
  DLDevice device_;

  ailoy::fs::path_t model_path_;
  tvm::runtime::Module mod_;
  nlohmann::json metadata_ = {};
  nlohmann::json mlc_chat_config_ = {};
  tvm::runtime::vm::NDArrayCacheMetadata ndarray_cache_metadata_;
  tvm::Array<tvm::runtime::NDArray> params_;

  std::string err_;
};

} // namespace ailoy
