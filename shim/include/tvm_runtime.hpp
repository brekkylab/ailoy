#pragma once

#include <cstddef>

#include <dlpack/dlpack.h>
#include <picojson.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/tensor.h>

// Forward Declaration for cxx_bridge.rs.h
struct DLPackTensor;

namespace ailoy {

struct CacheContents;

class tvm_runtime_t {
public:
  tvm_runtime_t(CacheContents &contents, const DLDevice &device);

  tvm::ffi::Module get_vm() const {
    if (!vm_.defined())
      throw std::runtime_error("VM not created yet");
    return vm_.value();
  }

  const picojson::object &get_metadata() const { return metadata_; }

  tvm::ffi::Function get_function(const std::string_view fname) {
    return *tvm::ffi::Function::GetGlobal(std::string(fname));
  }

  tvm::ffi::Function get_vm_function(const std::string_view fname,
                                     bool query_imports = false) {
    tvm::ffi::Optional<tvm::ffi::Function> func =
        get_vm()->GetFunction(std::string(fname), query_imports);
    ICHECK(func.defined()) << "Cannot find function: " << fname;
    return func.value();
  }

  tvm::runtime::ObjectRef get_params() const { return params_; }

  DLDevice get_device() const { return device_; }

private:
  DLDevice device_;
  tvm::ffi::Optional<tvm::ffi::Module> vm_ = std::nullopt;
  picojson::object metadata_;
  tvm::ffi::Array<tvm::ffi::Tensor> params_;
};

} // namespace ailoy
