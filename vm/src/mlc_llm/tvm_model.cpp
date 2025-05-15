#include "tvm_model.hpp"

#include <filesystem>
#include <fstream>
#include <regex>
#include <thread>

#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/registry.h>

#include "../file_util.hpp"
#include "model_cache.hpp"

using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::runtime::relax_vm;

namespace ailoy {

/* value_t interface related to tvm */

std::shared_ptr<ndarray_t> ndarray_from_tvm(tvm::runtime::NDArray tvm_ndarray) {
  auto shape = tvm_ndarray.Shape();
  auto dtype = tvm_ndarray->dtype;
  size_t nbytes =
      std::reduce(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  nbytes *= (dtype.bits * dtype.lanes + 7) / 8;

  std::vector<uint8_t> data(nbytes);
  tvm_ndarray.CopyToBytes(data.data(), nbytes);

  auto ndarray =
      create<ndarray_t>(std::vector<size_t>(shape.begin(), shape.end()), dtype,
                        data.data(), nbytes);
  return ndarray;
}

/* tvm_model_t */

void from_json(const nlohmann::json &j,
               NDArrayCacheMetadata::FileRecord::ParamRecord &param_record) {
  j.at("name").get_to(param_record.name);
  if (j.contains("dtype")) {
    param_record.dtype =
        DataType(String2DLDataType(j["dtype"].get<std::string>()));
  }
  j.at("format").get_to(param_record.format);
  j.at("nbytes").get_to(param_record.nbytes);
  j.at("byteOffset").get_to(param_record.byte_offset);
  if (j.contains("shape")) {
    std::vector<ShapeTuple::index_type> shape;
    nlohmann::json::array_t shape_json =
        j["shape"].get<nlohmann::json::array_t>();
    shape.reserve(shape_json.size());
    for (const auto &dim : shape_json) {
      shape.push_back(dim.get<int64_t>());
    }
    param_record.shape = ShapeTuple(std::move(shape));
  }
}

void from_json(const nlohmann::json &j,
               NDArrayCacheMetadata::FileRecord &file_record) {
  j.at("dataPath").get_to(file_record.data_path);
  j.at("format").get_to(file_record.format);
  j.at("nbytes").get_to(file_record.nbytes);
  if (j.contains("records")) {
    nlohmann::json::array_t records =
        j["records"].get<nlohmann::json::array_t>();
    file_record.records.reserve(records.size());
    for (const auto &item : records) {
      NDArrayCacheMetadata::FileRecord::ParamRecord record;
      from_json(item, record);
      file_record.records.push_back(record);
    }
  }
}

void from_json(const nlohmann::json &j, NDArrayCacheMetadata &metadata) {
  if (j.contains("records")) {
    nlohmann::json::array_t records =
        j["records"].get<nlohmann::json::array_t>();
    metadata.records.reserve(records.size());
    for (const auto &item : records) {
      NDArrayCacheMetadata::FileRecord record;
      from_json(item, record);
      metadata.records.push_back(record);
    }
  }
}

void tvm_model_t::load_ndarray_cache_metadata(const std::string &bytes) {
  auto j = nlohmann::json::parse(bytes);
  from_json(j, ndarray_cache_metadata_);
}

void tvm_model_t::load_ndarray_cache_shard(const size_t &shard_idx,
                                           const std::string &bytes) {
  const NDArrayCacheMetadata::FileRecord &shard_rec =
      ndarray_cache_metadata_.records[shard_idx];
  CHECK_EQ(shard_rec.format, "raw-shard")
      << "ValueError: Only `raw-shard` format is supported";
  CHECK_EQ(shard_rec.nbytes, bytes.length())
      << "ValueError: Encountered an corrupted parameter shard. It means it is "
         "not downloaded completely or downloading is interrupted. Please try "
         "to download again.";
  const PackedFunc *fupdate_cache =
      Registry::Get("vm.builtin.ndarray_cache.update");
  Optional<NDArray> staging_buffer;
  for (const NDArrayCacheMetadata::FileRecord::ParamRecord &param_record :
       shard_rec.records) {
    NDArray param;
    try {
      param = param_record.Load(device_, &bytes, &staging_buffer);
    } catch (const dmlc::Error &e) {
      LOG(FATAL) << "ValueError: Error when loading parameters for "
                 << param_record.name << ": " << e.what();
    }
    (*fupdate_cache)(param_record.name, param, true);
  }
}

void tvm_model_t::load_params_from_cache() {
  constexpr const char *name_loader =
      "vm.builtin.param_array_from_cache_by_name";
  const PackedFunc *fload_params = Registry::Get(name_loader);
  ICHECK(fload_params) << "Cannot find env function: " << name_loader;

  Array<String> param_names;
  param_names.reserve(get_metadata()["params"].size());
  for (const auto &param : get_metadata()["params"]) {
    std::string param_name = param["name"];
    param_names.push_back(param_name);
  }
  params_ = (*fload_params)(param_names);
}

tvm_model_t::tvm_model_t(const std::string &model_name,
                         const std::string &quantization, DLDevice device)
    : model_name_(model_name), quantization_(quantization), device_(device) {
  auto [model_path, model_lib_path] =
      get_model(model_name, quantization,
                tvm::runtime::DLDeviceType2Str(device.device_type));
  model_path_ = model_path;

  Module executable =
      tvm::runtime::Module::LoadFromFile(model_lib_path.string());
  if (!executable.defined())
    throw std::runtime_error("Failed to load system");
  auto fload_exec = executable->GetFunction("vm_load_executable");
  if (!fload_exec.defined())
    throw std::runtime_error("Failed to get executable loader");
  auto vm = fload_exec().operator Module();
  vm->GetFunction("vm_initialization")(
      static_cast<int>(device_.device_type), device_.device_id,
      static_cast<int>(memory::AllocatorType::kPooled),
      static_cast<int>(kDLCPU), 0,
      static_cast<int>(memory::AllocatorType::kPooled));
  mod_ = vm;

  // Load model metadata
  TypedPackedFunc<tvm::String()> fmetadata = vm.GetFunction("_metadata");
  metadata_ = nlohmann::json::parse(static_cast<std::string>(fmetadata()));

  // Load ndarray cache metadata
  auto contents = utils::LoadBytesFromFile(model_path / "ndarray-cache.json");
  load_ndarray_cache_metadata(contents);

  // Load ndarray cache
  std::regex re("params_shard_(\\d+)\\.bin");
  std::smatch match;
  for (const auto &entry : std::filesystem::directory_iterator(model_path)) {
    auto file_path = entry.path().string();
    auto file_name = entry.path().filename().string();
    if (std::regex_match(file_name, match, re)) {
      size_t i = std::stoi(match[1].str());
      auto contents = utils::LoadBytesFromFile(file_path);
      load_ndarray_cache_shard(i, contents);
    }
  }

  // Initialize parameters
  load_params_from_cache();
}

} // namespace ailoy
