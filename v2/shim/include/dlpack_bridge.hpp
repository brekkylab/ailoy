#pragma once

#include "dlpack/dlpack.h"
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace dlpack_bridge {

class ManagedTensor {
private:
  DLManagedTensorVersioned *tensor_;
  bool owned_;

public:
  explicit ManagedTensor(DLManagedTensorVersioned *tensor)
      : tensor_(tensor), owned_(true) {
    if (!tensor_) {
      throw std::invalid_argument("Null tensor provided");
    }
  }

  // No copy constructors
  ManagedTensor(const ManagedTensor &) = delete;
  ManagedTensor &operator=(const ManagedTensor &) = delete;

  // Move constructors
  ManagedTensor(ManagedTensor &&other) noexcept
      : tensor_(other.tensor_), owned_(other.owned_) {
    other.tensor_ = nullptr;
    other.owned_ = false;
  }

  // call deleter in descrtuctor
  ~ManagedTensor() {
    if (owned_ && tensor_ && tensor_->deleter) {
      tensor_->deleter(tensor_);
    }
  }

  int get_ndim() const {
    if (!tensor_)
      return false;

    const auto &dl_tensor = tensor_->dl_tensor;

    return dl_tensor.ndim;
  }

  int64_t get_dimension() const {
    if (!tensor_)
      return false;

    const auto &dl_tensor = tensor_->dl_tensor;

    // return -1 if tensor is not 1-dimensional
    for (int i = 0; i < dl_tensor.ndim - 1; i++) {
      if (dl_tensor.shape[i] != 1)
        return -1;
    }
    return dl_tensor.shape[dl_tensor.ndim - 1];
  }

  bool is_cpu_tensor() const {
    if (!tensor_)
      return false;
    return tensor_->dl_tensor.device.device_type == kDLCPU;
  }

  bool has_int_dtype(uint8_t bits = 32) const {
    if (!tensor_)
      return false;
    const auto &dl_tensor = tensor_->dl_tensor;
    return (dl_tensor.dtype.code == kDLInt && dl_tensor.dtype.bits == bits);
  }

  bool has_uint_dtype(uint8_t bits = 32) const {
    if (!tensor_)
      return false;
    const auto &dl_tensor = tensor_->dl_tensor;
    return (dl_tensor.dtype.code == kDLUInt && dl_tensor.dtype.bits == bits);
  }

  bool has_float_dtype(uint8_t bits = 32) const {
    if (!tensor_)
      return false;
    const auto &dl_tensor = tensor_->dl_tensor;
    return (dl_tensor.dtype.code == kDLFloat && dl_tensor.dtype.bits == bits);
  }

  const uint16_t *get_data_ptr_u16() const {
    if (!tensor_)
      return nullptr;
    return static_cast<const uint16_t *>(tensor_->dl_tensor.data);
  }

  const float *get_data_ptr_f32() const {
    if (!tensor_)
      return nullptr;
    return static_cast<const float *>(tensor_->dl_tensor.data);
  }

  DLManagedTensorVersioned *release_tensor() {
    owned_ = false; // release ownership not to call deleter
    return tensor_;
  }
};

std::unique_ptr<DLDevice> create_dldevice(int device_type, int device_id);

std::unique_ptr<ManagedTensor>
create_managed_tensor(DLManagedTensorVersioned *tensor);

} // namespace dlpack_bridge
