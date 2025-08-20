#pragma once

#include "dlpack/dlpack.h"
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace dlpack_bridge {

// DLPack 텐서를 안전하게 관리하는 RAII 래퍼
class ManagedTensor {
private:
  DLManagedTensorVersioned *tensor_;
  bool owned_;

public:
  // 생성자: DLPack 텐서의 소유권을 받음
  explicit ManagedTensor(DLManagedTensorVersioned *tensor)
      : tensor_(tensor), owned_(true) {
    if (!tensor_) {
      throw std::invalid_argument("Null tensor provided");
    }
  }

  // 복사 생성자 삭제 (소유권을 명확히 하기 위해)
  ManagedTensor(const ManagedTensor &) = delete;
  ManagedTensor &operator=(const ManagedTensor &) = delete;

  // 이동 생성자
  ManagedTensor(ManagedTensor &&other) noexcept
      : tensor_(other.tensor_), owned_(other.owned_) {
    other.tensor_ = nullptr;
    other.owned_ = false;
  }

  // 소멸자: deleter 호출하여 메모리 해제
  ~ManagedTensor() {
    if (owned_ && tensor_ && tensor_->deleter) {
      tensor_->deleter(tensor_);
    }
  }

  // 텐서가 1차원 float32인지 검사
  bool is_1d_float32() const {
    if (!tensor_)
      return false;

    const auto &dl_tensor = tensor_->dl_tensor;

    // 1차원인지 확인
    if (dl_tensor.ndim != 1)
      return false;

    // float32 타입인지 확인
    if (dl_tensor.dtype.code != kDLFloat || dl_tensor.dtype.bits != 32)
      return false;

    return true;
  }

  // 텐서의 길이(요소 개수) 반환
  int64_t get_size() const {
    if (!tensor_ || tensor_->dl_tensor.ndim == 0)
      return 0;
    return tensor_->dl_tensor.shape[0];
  }

  // CPU 메모리에 있는지 확인
  bool is_cpu_tensor() const {
    if (!tensor_)
      return false;
    return tensor_->dl_tensor.device.device_type == kDLCPU;
  }

  // float32 데이터 포인터 반환 (unsafe, Rust에서 안전하게 처리)
  const float *get_data_ptr() const {
    if (!tensor_)
      return nullptr;
    return static_cast<const float *>(tensor_->dl_tensor.data);
  }

  DLManagedTensorVersioned *release_tensor() {
    owned_ = false; // 소유권을 포기했으므로 소멸자에서 deleter를 호출하면 안됨
    return tensor_;
  }
};

std::unique_ptr<DLDevice> create_dldevice(int device_type, int device_id);

// 팩토리 함수들
std::unique_ptr<ManagedTensor>
create_managed_tensor(DLManagedTensorVersioned *tensor);

} // namespace dlpack_bridge
