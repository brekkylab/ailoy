#include "tvm_runtime.hpp"

#include <filesystem>
#include <format>
#include <string>
#include <unordered_map>

#include <dlpack/dlpack.h>
#include <nlohmann/json.hpp>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/vm/ndarray_cache_support.h>

#include <iostream>

using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::runtime::vm;

namespace ailoy {

// Array<NDArray> load_params_from_cache() {
//   constexpr const char *name_loader =
//       "vm.builtin.param_array_from_cache_by_name";
//   const tvm::ffi::Function fload_params =
//       tvm::ffi::Function::GetGlobal(name_loader).value();

//   Array<String> param_names;
//   param_names.reserve(get_metadata()["params"].size());
//   for (const auto &param : get_metadata()["params"]) {
//     std::string param_name = param["name"];
//     param_names.push_back(param_name);
//   }
//   return fload_params(param_names).cast<Array<NDArray>>();
// }

class tvm_runtime_t {
public:
  tvm_runtime_t(
      const std::string &lib_filename,
      const std::unordered_map<std::string, std::string> file_contents,
      DLDevice device);

private:
  Module vm_;
};

tvm_runtime_t::tvm_runtime_t(
    const std::string &lib_filename,
    const std::unordered_map<std::string, std::string> file_contents,
    DLDevice device) {
  // Load module
  const auto floadfile_so =
      tvm::ffi::Function::GetGlobal("runtime.module.loadfile_so");
  if (!floadfile_so)
    throw std::runtime_error("Failed to get runtime.module.loadfile_so");
  Module executable = tvm::runtime::Module::LoadFromFile(lib_filename);
  if (!executable.defined())
    throw std::runtime_error("Failed to load system");
  auto fload_exec = executable->GetFunction("vm_load_executable");
  if (!fload_exec.defined())
    throw std::runtime_error("Failed to get executable loader");
  auto vm = fload_exec().cast<Module>();
  vm->GetFunction("vm_initialization")(
      static_cast<int>(device.device_type), device.device_id,
      static_cast<int>(memory::AllocatorType::kPooled),
      static_cast<int>(kDLCPU), 0,
      static_cast<int>(memory::AllocatorType::kPooled));
  vm_ = vm;

  // // Load model metadata
  // tvm::ffi::TypedFunction<tvm::String()> fmetadata =
  //     vm.GetFunction("_metadata");
  // auto metadata =
  // nlohmann::json::parse(static_cast<std::string>(fmetadata())); std::cout <<
  // metadata << std::endl;

  // Load ndarray cache metadata
  // NDArrayCacheMetadata::LoadFromStr()

  // from_json(nlohmann::json::parse(
  //               ailoy::fs::read_file_text(model_path_ / "ndarray-cache.json")
  //                   .unwrap()),
  //           ndarray_cache_metadata_);

  // // Load ndarray cache
  // int record_idx = 0;
  // for (const auto &record : ndarray_cache_metadata_.records) {
  //   auto contents =
  //       ailoy::fs::read_file_bytes(model_path_ / record.data_path).unwrap();
  //   load_ndarray_cache_shard(record_idx,
  //                            std::string(contents.begin(), contents.end()));
  //   record_idx++;
  // }
}

} // namespace ailoy

extern "C" {
struct ailoy_file_contents_t {
  std::unordered_map<std::string, std::string> inner;
};

int ailoy_file_contents_create(ailoy_file_contents_t **out) {
  *out = new ailoy_file_contents_t{};
  return 0;
}

void ailoy_file_contents_destroy(ailoy_file_contents_t *contents) {
  delete contents;
}

int ailoy_file_contents_insert(ailoy_file_contents_t *contents,
                               char const *filename, size_t len,
                               char const *content) {
  contents->inner.insert_or_assign(filename, std::string(content, len));
  return 0;
}

int ailoy_tvm_runtime_create(char const *lib_filename,
                             ailoy_file_contents_t const *contents,
                             ailoy::tvm_runtime_t **out) {
  *out = new ailoy::tvm_runtime_t(
      lib_filename, contents->inner,
      DLDevice{.device_type = DLDeviceType::kDLMetal, .device_id = 0});
  return 0;
}

void ailoy_tvm_runtime_destroy(ailoy::tvm_runtime_t *model) { delete model; }
}
