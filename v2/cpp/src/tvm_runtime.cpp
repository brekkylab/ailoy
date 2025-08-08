#include "tvm_runtime.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <unordered_map>

#include <dlpack/dlpack.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/vm/ndarray_cache_support.h>

#include <iostream>

namespace fs = std::filesystem;
using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::runtime::vm;

namespace ailoy {

tvm_runtime_t::tvm_runtime_t(const std::string &lib_path,
                             cache_t cache_contents, const DLDevice &device) {
  // Device
  device_ = device;

  // Load module
  const auto floadfile_so =
      tvm::ffi::Function::GetGlobal("runtime.module.loadfile_so");
  if (!floadfile_so)
    throw std::runtime_error("Failed to get runtime.module.loadfile_so");
  auto executable = tvm::runtime::Module::LoadFromFile(lib_path);
  if (!executable.defined())
    throw std::runtime_error("Failed to load system");

  // Load vm
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

  // Load model metadata
  tvm::ffi::TypedFunction<tvm::String()> fmetadata =
      vm.GetFunction("_metadata");
  metadata_ = nlohmann::json::parse(static_cast<std::string>(fmetadata()));

  // Load ndarray cache metadata
  auto ndarray_cache_metadata = NDArrayCacheMetadata::LoadFromStr(
      *cache_contents.read_and_remove("ndarray-cache.json"),
      "ndarray-cache.json");

  // Load ndarray cache
  int shard_idx = 0;
  for (const auto &record : ndarray_cache_metadata.records) {
    auto bytes = *cache_contents.read_and_remove(record.data_path);
    {
      const NDArrayCacheMetadata::FileRecord &shard_rec =
          ndarray_cache_metadata.records[shard_idx];
      if (shard_rec.format != "raw-shard")
        throw std::runtime_error("Only `raw-shard` format is supported");
      if (shard_rec.nbytes != bytes.length())
        throw std::runtime_error("Encountered an corrupted parameter shard.");
      const tvm::ffi::Function fupdate_cache =
          tvm::ffi::Function::GetGlobal("vm.builtin.ndarray_cache.update")
              .value();
      Optional<NDArray> staging_buffer;
      for (const NDArrayCacheMetadata::FileRecord::ParamRecord &param_record :
           shard_rec.records) {
        NDArray param;
        param = param_record.Load(device, &bytes, &staging_buffer);
        fupdate_cache(param_record.name, param, true);
      }
    }
    shard_idx++;
  }

  // Load parameters from the cache
  constexpr const char *name_loader =
      "vm.builtin.param_array_from_cache_by_name";
  const tvm::ffi::Function fload_params =
      tvm::ffi::Function::GetGlobal(name_loader).value();

  Array<String> param_names;
  param_names.reserve(metadata_["params"].size());
  for (const auto &param : metadata_["params"]) {
    std::string param_name = param["name"];
    param_names.push_back(param_name);
  }
  params_ = fload_params(param_names).cast<Array<NDArray>>();
}

} // namespace ailoy
