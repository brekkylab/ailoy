#include "embedding_model.hpp"

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/runtime/int_tuple.h>

#include "cxx_bridge.rs.h"
#include "tvm_runtime.hpp"

using namespace tvm;
using namespace tvm::runtime;

namespace ailoy {

tvm_embedding_model_t::tvm_embedding_model_t(CacheContents &contents,
                                             DLDevice device) {
  rt_ = std::make_unique<tvm_runtime_t>(contents, device);

  fprefill_ = rt_->get_vm_function("prefill");
  if (!fprefill_.defined())
    throw std::runtime_error("Cannot find embed function");
}

void tvm_embedding_model_t::extract_tensor_part(const Tensor &from,
                                                Tensor &to) {
  // from: F16 or F32 Tensor
  if (!(from.DataType().code() == kDLFloat &&
        (from.DataType().bits() == 16 || from.DataType().bits() == 32))) {
    throw std::runtime_error("Datatype of 'from' array is invalid.");
  }

  // to: 1D Tensor with the same data type with 'from' array
  if (!(to.DataType().code() == kDLFloat &&
        (to.DataType().bits() == from.DataType().bits()) &&
        to.Shape().size() == 1)) {
    throw std::runtime_error("Datatype of 'to' array is invalid.");
  }

  // get sizes
  int64_t from_size = 1;
  for (int64_t dim : from.Shape())
    from_size *= dim;
  int64_t to_size = to.Shape().at(0);

  // size assertion
  if (from_size < to_size) {
    throw std::runtime_error(
        "size of input Tensor is too small to fill output Tensor.");
  }

  // process
  if (to.DataType().bits() == 16) {
    auto to_data = static_cast<uint16_t *>(to.ToDLPack()->dl_tensor.data);
    auto from_data = static_cast<uint16_t *>(from.ToDLPack()->dl_tensor.data);
    for (size_t i = 0; i < to_size; i++) {
      to_data[i] = from_data[i];
    }
  } else { // to.DataType().bits() == 32
    auto to_data = static_cast<float *>(to.ToDLPack()->dl_tensor.data);
    auto from_data = static_cast<float *>(from.ToDLPack()->dl_tensor.data);
    for (size_t i = 0; i < to_size; i++) {
      to_data[i] = from_data[i];
    }
  }
}

const Tensor tvm_embedding_model_t::infer(std::vector<int> tokens) {
  Device cpu = Device{kDLCPU, 0};
  DLDataType I32 = DLDataType{.code = kDLInt, .bits = 32, .lanes = 1};
  DLDataType F32 = DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1};

  int32_t tokens_length = tokens.size();

  Tensor inputTensorCPU = Tensor::Empty({1, tokens_length}, I32, cpu);
  Tensor maskTensorCPU = Tensor::Empty({1, tokens_length}, I32, cpu);
  auto input_tensor = static_cast<int32_t *>(inputTensorCPU->data);
  auto mask_tensor = static_cast<int32_t *>(maskTensorCPU->data);
  for (size_t i = 0; i < tokens_length; i++) {
    input_tensor[i] = tokens.at(i);
    mask_tensor[i] = 1;
  }

  Tensor inputTensorGPU =
      Tensor::Empty({1, tokens_length}, I32, rt_->get_device());
  inputTensorGPU.CopyFrom(inputTensorCPU);
  Tensor maskTensorGPU =
      Tensor::Empty({1, tokens_length}, I32, rt_->get_device());
  maskTensorGPU.CopyFrom(maskTensorCPU);

  Tensor logitsCurBatchOnGPU =
      fprefill_(inputTensorGPU, maskTensorGPU, rt_->get_params())
          .cast<Tensor>();
  Tensor logitsCurBatchOnCPU = Tensor::Empty(
      logitsCurBatchOnGPU.Shape(), logitsCurBatchOnGPU.DataType(), cpu);
  logitsCurBatchOnCPU.CopyFrom(logitsCurBatchOnGPU);

  Tensor processed_embedding =
      Tensor::Empty(tvm::ffi::Shape{logitsCurBatchOnCPU.Shape().back()},
                    logitsCurBatchOnCPU.DataType(), cpu);
  extract_tensor_part(logitsCurBatchOnCPU, processed_embedding);

  return processed_embedding;
}

DLPackTensor
tvm_embedding_model_t::infer_from_rs(rust::Slice<const uint32_t> tokens) {
  std::lock_guard<std::mutex> lk(m_);
  std::vector<int> converted;
  converted.reserve(tokens.size());

  std::transform(tokens.begin(), tokens.end(), std::back_inserter(converted),
                 [](uint32_t val) { return static_cast<int>(val); });

  auto tensor = infer(converted);
  auto raw_dlpack_ptr = std::move(tensor).ToDLPackVersioned();
  auto safe_managed_tensor =
      dlpack_bridge::create_managed_tensor(raw_dlpack_ptr);
  return DLPackTensor{std::move(safe_managed_tensor)};
}

std::unique_ptr<tvm_embedding_model_t>
create_tvm_embedding_model(CacheContents &contents,
                           std::unique_ptr<DLDevice> device) {
  return std::make_unique<tvm_embedding_model_t>(contents, std::move(*device));
}

} // namespace ailoy
