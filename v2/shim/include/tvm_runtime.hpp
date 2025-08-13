#pragma once

#include <cstddef>

#include <dlpack/dlpack.h>
#include <nlohmann/json.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

namespace ailoy {

struct CacheContents;

class tvm_runtime_t {
public:
  tvm_runtime_t(CacheContents &contents, const DLDevice &device);

  tvm::runtime::Module get_vm() const {
    if (!vm_.defined())
      throw std::runtime_error("VM not created yet");
    return vm_;
  }

  const nlohmann::json &get_metadata() const { return metadata_; }

  tvm::ffi::Function get_function(const std::string_view fname) {
    return *tvm::ffi::Function::GetGlobal(std::string(fname));
  }

  tvm::ffi::Function get_vm_function(const std::string_view fname,
                                     bool query_imports = false) {
    return get_vm().GetFunction(std::string(fname), query_imports);
  }

  tvm::runtime::ObjectRef get_params() const { return params_; }

  DLDevice get_device() const { return device_; }

private:
  DLDevice device_;
  tvm::runtime::Module vm_;
  nlohmann::json metadata_;
  tvm::Array<tvm::runtime::NDArray> params_;
};

} // namespace ailoy
