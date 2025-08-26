#include "dlpack_bridge.hpp"

namespace dlpack_bridge {

std::unique_ptr<ManagedTensor>
create_managed_tensor(DLManagedTensorVersioned *tensor) {
  if (!tensor) {
    throw std::invalid_argument(
        "Cannot create ManagedTensor from null pointer");
  }
  return std::make_unique<ManagedTensor>(tensor);
}

std::unique_ptr<DLDevice> create_dldevice(int device_type, int device_id) {
  auto dev = std::make_unique<DLDevice>();
  dev->device_type = static_cast<DLDeviceType>(device_type);
  dev->device_id = device_id;
  return dev;
}

} // namespace dlpack_bridge
