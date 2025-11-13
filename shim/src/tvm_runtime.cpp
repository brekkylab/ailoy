#include "tvm_runtime.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <unordered_map>

#include <dlpack/dlpack.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/vm/tensor_cache_support.h>

#include "cxx_bridge.rs.h"

namespace fs = std::filesystem;
using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::runtime::vm;

inline const char *rt_lib_name() {
#if defined(_WIN32)
  return "rt.dll";
#elif defined(__APPLE__)
  return "rt.dylib";
#else
  return "rt.so";
#endif
}

namespace ailoy {

tvm_runtime_t::tvm_runtime_t(CacheContents &contents, const DLDevice &device) {
  // Device
  device_ = device;

  auto get_path = [&contents](const std::string &filename) -> std::string {
    std::string rv;
    rust::String root;
    rust::String dirname;
    rust::String filename_;
    rust::Vec<uint8_t> bytes;
    cache_contents_remove_with_filename_out(contents, filename, dirname,
                                            filename_, bytes);
    rv =
        (std::filesystem::path(std::string(cache_contents_get_root(contents))) /
         std::string(dirname) / std::string(filename_))
            .string();
    return std::move(rv);
  };

  auto read_bytes = [&contents](const std::string &filename) -> std::string {
    std::string rv;
    rust::String root;
    rust::String dirname;
    rust::String filename_;
    rust::Vec<uint8_t> bytes;
    cache_contents_remove_with_filename_out(contents, filename, dirname,
                                            filename_, bytes);
    rv.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    return std::move(rv);
  };

  // Load module
  auto executable = tvm::ffi::Module::LoadFromFile(get_path(rt_lib_name()));
  if (!executable.defined())
    throw std::runtime_error("Failed to load system");

  // Load vm
  tvm::ffi::Optional<tvm::ffi::Function> fload_exec =
      executable->GetFunction("vm_load_executable");
  if (!fload_exec.defined())
    throw std::runtime_error("Failed to get executable loader");
  auto vm = fload_exec.value()().cast<tvm::ffi::Module>();
  vm->GetFunction("vm_initialization")
      .value()(static_cast<int>(device.device_type), device.device_id,
               static_cast<int>(memory::AllocatorType::kPooled),
               static_cast<int>(kDLCPU), 0,
               static_cast<int>(memory::AllocatorType::kPooled));
  vm_ = vm;

  // Load model metadata
  std::string json_str = "";
  tvm::ffi::Optional<tvm::ffi::Function> fmetadata =
      vm->GetFunction("_metadata");
  ICHECK(fmetadata.defined())
      << "ValueError: _metadata function not found in module";
  json_str = fmetadata.value()().cast<tvm::ffi::String>();

  picojson::value parsed;
  std::string err = picojson::parse(parsed, json_str);
  metadata_ = parsed.get<picojson::object>();

  if (!err.empty()) {
    std::cerr << "JSON parsing error: " << err << std::endl;
    return;
  }

  // Load tensor cache metadata
  auto tensor_cache_json = read_bytes("tensor-cache.json");
  // If tensor-cache.json is empty, fallback to ndarray-cache.json
  if (tensor_cache_json.empty()) {
    tensor_cache_json = read_bytes("ndarray-cache.json");
  }
  auto tensor_cache_metadata =
      TensorCacheMetadata::LoadFromStr(tensor_cache_json, "tensor-cache.json");

  // Load tensor cache
  int shard_idx = 0;
  for (const auto &record : tensor_cache_metadata.records) {
    auto bytes = read_bytes(record.data_path);
    {
      const TensorCacheMetadata::FileRecord &shard_rec =
          tensor_cache_metadata.records[shard_idx];
      if (shard_rec.format != "raw-shard")
        throw std::runtime_error("Only `raw-shard` format is supported");
      if (shard_rec.nbytes != bytes.length())
        throw std::runtime_error("Encountered an corrupted parameter shard.");
      const tvm::ffi::Function fupdate_cache =
          tvm::ffi::Function::GetGlobal("vm.builtin.tensor_cache.update")
              .value();
      tvm::ffi::Optional<Tensor> staging_buffer;
      for (const TensorCacheMetadata::FileRecord::ParamRecord &param_record :
           shard_rec.records) {
        Tensor param;
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

  const picojson::array &params_array =
      metadata_.at("params").get<picojson::array>();
  tvm::ffi::Array<tvm::ffi::String> param_names;
  param_names.reserve(params_array.size());
  for (const picojson::value &param : params_array) {
    const picojson::object &param_object = param.get<picojson::object>();
    param_names.push_back(param_object.at("name").get<std::string>());
  }
  params_ = fload_params(param_names).cast<tvm::ffi::Array<tvm::ffi::Tensor>>();
}

} // namespace ailoy
