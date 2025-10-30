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

void tvm_embedding_model_t::extract_ndarray_part(
    const tvm::runtime::NDArray &from, tvm::runtime::NDArray &to) {
  // from: F16 or F32 NDArray
  if (!(from.DataType().code() == kDLFloat &&
        (from.DataType().bits() == 16 || from.DataType().bits() == 32))) {
    throw std::runtime_error("Datatype of 'from' array is invalid.");
  }

  // to: 1D NDArray with the same data type with 'from' array
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
        "size of input NDArray is too small to fill output NDArray.");
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
      NDArray::Empty({1, tokens_length}, I32, rt_->get_device());
  inputNDArrayGPU.CopyFrom(inputNDArrayCPU);
  NDArray maskNDArrayGPU =
      NDArray::Empty({1, tokens_length}, I32, rt_->get_device());
  maskNDArrayGPU.CopyFrom(maskNDArrayCPU);

  NDArray logitsCurBatchOnGPU =
      fprefill_(inputNDArrayGPU, maskNDArrayGPU, rt_->get_params())
          .cast<NDArray>();
  NDArray logitsCurBatchOnCPU = NDArray::Empty(
      logitsCurBatchOnGPU.Shape(), logitsCurBatchOnGPU.DataType(), cpu);
  logitsCurBatchOnCPU.CopyFrom(logitsCurBatchOnGPU);

  NDArray processed_embedding =
      NDArray::Empty(tvm::ffi::Shape{logitsCurBatchOnCPU.Shape().back()},
                     logitsCurBatchOnCPU.DataType(), cpu);
  extract_ndarray_part(logitsCurBatchOnCPU, processed_embedding);

  return processed_embedding;
}

DLPackTensor
tvm_embedding_model_t::infer_from_rs(rust::Slice<const uint32_t> tokens) {
  std::lock_guard<std::mutex> lk(m_);
  std::vector<int> converted;
  converted.reserve(tokens.size());

  std::transform(tokens.begin(), tokens.end(), std::back_inserter(converted),
                 [](uint32_t val) { return static_cast<int>(val); });

  auto ndarray = infer(converted);
  auto raw_dlpack_ptr = std::move(ndarray).ToDLPackVersioned();
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
