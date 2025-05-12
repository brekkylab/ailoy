#include "embedding_model.hpp"

#include <filesystem>
#include <fstream>
#include <regex>
#include <thread>

#include <support/json_parser.h>
#include <tokenizers_c.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include "../file_util.hpp"
#include "../ndarray_util.hpp"

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

/* tokenizer_t */

tokenizer_t::tokenizer_t(const std::filesystem::path &json_file_path) {
  // tokenizer_ =
  //     tokenizers::Tokenizer::FromBlobJSON(LoadBytesFromFile(json_file));
  auto contents = utils::LoadBytesFromFile(json_file_path);
  handle_ = tokenizers_new_from_str(contents.data(), contents.size());
}

std::vector<tokenizer_t::token_t> tokenizer_t::encode(const std::string &text,
                                                      bool add_special_token) {
  TokenizerEncodeResult result;
  tokenizers_encode(handle_, text.data(), text.length(),
                    static_cast<int>(add_special_token), &result);

  std::vector<tokenizer_t::token_t> rv(result.token_ids,
                                       result.token_ids + result.len);
  tokenizers_free_encode_results(&result, 1);
  return rv;
  // return tokenizer_->Encode(text);
}

std::string tokenizer_t::decode(const std::vector<tokenizer_t::token_t> &ids,
                                bool skip_special_tokens) {
  size_t ids_size = ids.size();
  std::vector<uint32_t> ids_data(ids_size);
  // uint32_t ids_data[ids_size];
  for (size_t i = 0; i < ids_size; i++)
    ids_data[i] = static_cast<uint32_t>(ids.at(i));
  tokenizers_decode(handle_, ids_data.data(), ids_size,
                    static_cast<int>(skip_special_tokens));

  char *rv_data;
  size_t len;
  tokenizers_get_decode_str(handle_, const_cast<const char **>(&rv_data), &len);

  std::string rv(rv_data, len);
  return rv;
  // return tokenizer_->Decode(ids);
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

/* tvm_embedding_model_t */

tvm_embedding_model_t::tvm_embedding_model_t(const std::string &model_name,
                                             const std::string &quantization)
    : object_t() {
  if (engine_ == nullptr)
    engine_ = prepare_engine(model_name, quantization);
  fprefill_ = engine_->get_vm_function("prefill");
}

void tvm_embedding_model_t::postprocess_embedding_ndarray(
    const tvm::runtime::NDArray &from, tvm::runtime::NDArray &to) {
  // from: F16 or F32 NDArray
  if (!(from.DataType().code() == kDLFloat &&
        (from.DataType().bits() == 16 || from.DataType().bits() == 32))) {
    throw ailoy::exception();
  }
  // to: 1D F32 NDArray
  if (!(to.DataType().code() == kDLFloat && to.DataType().bits() == 32 &&
        to.Shape().size() == 1)) {
    throw ailoy::exception();
  }

  // get sizes
  int64_t from_size = 1;
  for (int64_t dim : from.Shape())
    from_size *= dim;
  int64_t to_size = to.Shape().at(0);

  // size assertion
  if (from_size < to_size) {
    throw ailoy::exception(
        "size of input NDArray is too small to fill output NDArray.");
  }

  // process
  auto to_data = static_cast<float *>(to.ToDLPack()->dl_tensor.data);
  if (from.DataType().bits() == 16) {
    auto from_data = static_cast<uint16_t *>(from.ToDLPack()->dl_tensor.data);
    for (size_t i = 0; i < to_size; i++) {
      to_data[i] = float16_to_float32(from_data[i]);
    }
  } else {
    auto from_data = static_cast<float *>(from.ToDLPack()->dl_tensor.data);
    for (size_t i = 0; i < to_size; i++) {
      to_data[i] = from_data[i];
    }
  }
}

const tvm::runtime::NDArray
tvm_embedding_model_t::infer(std::vector<int> tokens) {
  Device cpu = Device{kDLCPU, 0};
  DLDataType I32 = DLDataType{.code = kDLInt, .bits = 32, .lanes = 1};
  DLDataType F32 = DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1};

  int32_t tokens_length = tokens.size();

  NDArray inputNDArrayCPU = NDArray::Empty({1, tokens_length}, I32, cpu);
  NDArray maskNDArrayCPU = NDArray::Empty({1, tokens_length}, I32, cpu);
  auto input_nd_array = static_cast<int32_t *>(inputNDArrayCPU->data);
  auto mask_nd_array = static_cast<int32_t *>(maskNDArrayCPU->data);
  for (size_t i = 0; i < tokens_length; i++) {
    input_nd_array[i] = tokens.at(i);
    mask_nd_array[i] = 1;
  }

  NDArray inputNDArrayGPU =
      NDArray::Empty({1, tokens_length}, I32, engine_->get_device());
  inputNDArrayGPU.CopyFrom(inputNDArrayCPU);
  NDArray maskNDArrayGPU =
      NDArray::Empty({1, tokens_length}, I32, engine_->get_device());
  maskNDArrayGPU.CopyFrom(maskNDArrayCPU);

  NDArray logitsCurBatchOnGPU =
      fprefill_(inputNDArrayGPU, maskNDArrayGPU, engine_->get_params());
  NDArray logitsCurBatchOnCPU = NDArray::Empty(
      logitsCurBatchOnGPU.Shape(), logitsCurBatchOnGPU.DataType(), cpu);
  logitsCurBatchOnCPU.CopyFrom(logitsCurBatchOnGPU);

  NDArray processed_embedding =
      NDArray::Empty(ShapeTuple{logitsCurBatchOnCPU.Shape().back()}, F32, cpu);
  postprocess_embedding_ndarray(logitsCurBatchOnCPU, processed_embedding);

  return processed_embedding;
}

component_or_error_t
create_tvm_embedding_model_component(std::shared_ptr<const value_t> inputs) {
  if (!inputs->is_type_of<map_t>())
    return error_output_t(type_error("TVM Embedding Model: create", "inputs",
                                     "map_t", inputs->get_type()));

  auto input_map = inputs->as<map_t>();

  // Parse model name(optional)
  std::string model_name;
  if (input_map->contains("model")) {
    if (input_map->at("model")->is_type_of<string_t>())
      model_name = *input_map->at<string_t>("model");
    else
      return error_output_t(type_error("TVM Embedding Model: create", "model",
                                       "string_t",
                                       input_map->at("model")->get_type()));
  } else {
    model_name = "BAAI/bge-m3";
  }

  // Parse quantization(optional)
  std::string quantization;
  if (input_map->contains("quantization")) {
    if (input_map->at("model")->is_type_of<string_t>())
      quantization = *input_map->at<string_t>("quantization");
    else
      return error_output_t(type_error("TVM Embedding Model: create",
                                       "quantization", "string_t",
                                       input_map->at("model")->get_type()));
  } else {
    quantization = "q4f16_1";
  }

  auto infer = [](std::shared_ptr<component_t> component,
                  std::shared_ptr<const value_t> inputs) -> value_or_error_t {
    if (!inputs->is_type_of<map_t>())
      return error_output_t(type_error("TVM Embedding Model: infer", "inputs",
                                       "map_t", inputs->get_type()));

    auto input_map = inputs->as<map_t>();

    // Get input prompt
    if (!input_map->contains("prompt"))
      return error_output_t(
          range_error("TVM Embedding Model: infer", "prompt"));
    if (!input_map->at("prompt")->is_type_of<string_t>())
      return error_output_t(type_error("TVM Embedding Model: infer", "prompt",
                                       "string_t",
                                       input_map->at("prompt")->get_type()));
    auto prompt = input_map->at<string_t>("prompt");

    auto tokens =
        component->get_obj("tokenizer")->as<tokenizer_t>()->encode(*prompt);

    // Run inference with embedding model
    auto embedding = component->get_obj("embedding_model")
                         ->as<tvm_embedding_model_t>()
                         ->infer(tokens);

    auto outputs = create<map_t>();
    outputs->insert_or_assign("embedding", ndarray_from_tvm(embedding));
    return outputs;
  };

  auto tokenize =
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<const value_t> inputs) -> value_or_error_t {
    if (!inputs->is_type_of<map_t>())
      return error_output_t(type_error("TVM Embedding Model: infer", "inputs",
                                       "map_t", inputs->get_type()));

    auto input_map = inputs->as<map_t>();

    // Get input prompt
    if (!input_map->contains("prompt"))
      return error_output_t("Input prompt not specified");
    if (!input_map->at("prompt")->is_type_of<string_t>())
      return error_output_t(type_error("TVM Embedding Model: infer", "prompt",
                                       "string_t",
                                       input_map->at("prmopt")->get_type()));
    auto prompt = input_map->at<string_t>("prompt");

    auto tokens = create<array_t>();
    for (auto id :
         component->get_obj("tokenizer")->as<tokenizer_t>()->encode(*prompt))
      tokens->push_back(create<int_t>(id));

    auto outputs = create<map_t>();
    outputs->insert_or_assign("tokens", tokens);
    return outputs;
  };

  auto tvm_embedding_model =
      create<tvm_embedding_model_t>(model_name, quantization);

  auto tokenizer = create<tokenizer_t>(tvm_embedding_model->get_model_path() /
                                       "tokenizer.json");

  auto component = create<component_t>(
      std::initializer_list<
          std::pair<const std::string, std::shared_ptr<method_operator_t>>>{
          {"tokenize", create<instant_method_operator_t>(tokenize)},
          {"infer", create<instant_method_operator_t>(infer)},
      });
  component->set_obj("embedding_model", tvm_embedding_model);
  component->set_obj("tokenizer", tokenizer);
  return component;
}

} // namespace ailoy
