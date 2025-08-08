// #include "rust_bridge.hpp"

// #include <iostream>

// std::unique_ptr<std::string> make_unique_string(std::string v) {
//   return std::make_unique<std::string>(std::move(v));
// }

// std::unique_ptr<DLDevice> create_dldevice(int32_t device_type,
//                                           int32_t device_id) {
//   auto rv = std::make_unique<DLDevice>();
//   rv->device_type = static_cast<DLDeviceType>(device_type);
//   rv->device_id = device_id;
//   return std::move(rv);
// }

// TVMLanguageModel::TVMLanguageModel(
//     const std::string &lib_filename,
//     //  std::unique_ptr<FileContents> file_contents,
//     std::unique_ptr<DLDevice> device) {}

// int32_t TVMLanguageModel::get_result() const { return 0; }

// std::unique_ptr<TVMLanguageModel>
// create_tvm_language_model(const std::string &lib_filename,
//                           // std::unique_ptr<FileContents> file_contents,
//                           std::unique_ptr<DLDevice> device) {
//   std::cout << "lib " << lib_filename << std::endl;
//   std::cout << "device " << device->device_type << std::endl;
//   return std::make_unique<TVMLanguageModel>(lib_filename, std::move(device));
// }
