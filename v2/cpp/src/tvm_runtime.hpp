#pragma once

#include <cstddef>

#include <nlohmann/json.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

namespace ailoy {

class tvm_runtime_t {
public:
  tvm_runtime_t(const std::string &lib_path,
                std::unordered_map<std::string, std::string> &file_contents,
                DLDevice device);

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

extern "C" {
struct ailoy_file_contents_t;

int ailoy_file_contents_create(ailoy_file_contents_t **out);

void ailoy_file_contents_destroy(ailoy_file_contents_t *contents);

int ailoy_file_contents_insert(ailoy_file_contents_t *contents,
                               char const *filename, size_t len,
                               char const *content);

int ailoy_tvm_runtime_create(const char *lib_path,
                             ailoy_file_contents_t const *contents,
                             ailoy::tvm_runtime_t **out);

void ailoy_tvm_runtime_destroy(ailoy::tvm_runtime_t *model);
}
