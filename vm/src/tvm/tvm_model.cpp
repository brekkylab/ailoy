#include "tvm_model.hpp"

#include <filesystem>
#include <regex>

#include <tvm/ffi/any.h>
#include <tvm/runtime/memory/memory_manager.h>

#include "model_cache.hpp"

using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::runtime::vm;

namespace ailoy {

bool tvm_device_exist(DLDevice device) {
  tvm::ffi::Any rv;
  tvm::runtime::DeviceAPI::Get(device, true)
      ->GetAttr(device, tvm::runtime::DeviceAttrKind::kExist, &rv);
  return rv.cast<bool>();
}

std::optional<DLDevice> get_tvm_device(int32_t device_id) {
#if defined(USE_METAL)
  auto device_type = kDLMetal;
#elif defined(USE_VULKAN)
  auto device_type = kDLVulkan;
#else
  auto device_type = kDLCPU;
#endif

  auto device_type_str = tvm::runtime::DLDeviceType2Str(device_type);
  if (tvm_device_exist(DLDevice{device_type, device_id})) {
    debug("using device {}:{}", device_type_str, device_id);
    return DLDevice{device_type, device_id};
  } else if (tvm_device_exist(DLDevice{device_type, 0})) {
    info("Device {}:{} doesn't exist, use {}:0 instead.", device_type_str,
         device_id, device_type_str);
    return DLDevice{device_type, 0};
  } else {
    debug("No {} device is detected.", device_type_str);
    return std::nullopt;
  }
}

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
        DataType(StringToDLDataType(j["dtype"].get<std::string>()));
  }
  j.at("format").get_to(param_record.format);
  j.at("nbytes").get_to(param_record.nbytes);
  j.at("byteOffset").get_to(param_record.byte_offset);
  if (j.contains("shape")) {
    std::vector<ffi::Shape::index_type> shape;
    nlohmann::json::array_t shape_json =
        j["shape"].get<nlohmann::json::array_t>();
    shape.reserve(shape_json.size());
    for (const auto &dim : shape_json) {
      shape.push_back(dim.get<int64_t>());
    }
    param_record.shape = ffi::Shape(std::move(shape));
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
  const tvm::ffi::Function fupdate_cache =
      tvm::ffi::Function::GetGlobal("vm.builtin.ndarray_cache.update").value();
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
    fupdate_cache(param_record.name, param, true);
  }
}

void tvm_model_t::load_params_from_cache() {
  constexpr const char *name_loader =
      "vm.builtin.param_array_from_cache_by_name";
  const tvm::ffi::Function fload_params =
      tvm::ffi::Function::GetGlobal(name_loader).value();

  Array<String> param_names;
  param_names.reserve(get_metadata()["params"].size());
  for (const auto &param : get_metadata()["params"]) {
    std::string param_name = param["name"];
    param_names.push_back(param_name);
  }
  params_ = fload_params(param_names).cast<Array<NDArray>>();
}

tvm_model_t::tvm_model_t(const std::string &model_name,
                         const std::string &quantization, DLDevice device)
    : model_name_(model_name), quantization_(quantization), device_(device) {
  auto download_model_result =
      download_model(model_name, quantization,
                     tvm::runtime::DLDeviceType2Str(device.device_type));
  if (!download_model_result.success) {
    throw ailoy::runtime_error(download_model_result.error_message.value());
  }

  model_path_ = download_model_result.model_path.value().string();

  Module executable = tvm::runtime::Module::LoadFromFile(
      download_model_result.model_lib_path.value().string());
  if (!executable.defined())
    throw std::runtime_error("Failed to load system");
  auto fload_exec = executable->GetFunction("vm_load_executable");
  if (!fload_exec.defined())
    throw std::runtime_error("Failed to get executable loader");
  auto vm = fload_exec().cast<Module>();
  vm->GetFunction("vm_initialization")(
      static_cast<int>(device_.device_type), device_.device_id,
      static_cast<int>(memory::AllocatorType::kPooled),
      static_cast<int>(kDLCPU), 0,
      static_cast<int>(memory::AllocatorType::kPooled));
  mod_ = vm;

  // Load model metadata
  tvm::ffi::TypedFunction<tvm::String()> fmetadata =
      vm.GetFunction("_metadata");
  metadata_ = nlohmann::json::parse(static_cast<std::string>(fmetadata()));

  // Load mlc-chat-config.json
  mlc_chat_config_ = nlohmann::json::parse(
      ailoy::fs::read_file_text(model_path_ / "mlc-chat-config.json").unwrap());

  // Load ndarray cache metadata
  from_json(nlohmann::json::parse(
                ailoy::fs::read_file_text(model_path_ / "ndarray-cache.json")
                    .unwrap()),
            ndarray_cache_metadata_);

  // Load ndarray cache
  int record_idx = 0;
  for (const auto &record : ndarray_cache_metadata_.records) {
    auto contents =
        ailoy::fs::read_file_bytes(model_path_ / record.data_path).unwrap();
    load_ndarray_cache_shard(record_idx,
                             std::string(contents.begin(), contents.end()));
    record_idx++;
  }

  // Initialize parameters
  load_params_from_cache();
}

} // namespace ailoy
