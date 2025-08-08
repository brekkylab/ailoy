// #pragma once

// #include <cstddef>
// #include <memory>
// #include <unordered_map>
// #include <vector>

// #include <dlpack/dlpack.h>

// namespace ailoy {

// class tvm_language_model_t;

// } // namespace ailoy

// std::unique_ptr<std::string> make_unique_string(std::string v);

// std::unique_ptr<DLDevice> create_dldevice(int32_t device_type,
//                                           int32_t device_id);

// class TVMLanguageModel {
// public:
//   TVMLanguageModel(const std::string &lib_filename,
//                    std::unique_ptr<DLDevice> device);

//   int32_t get_result() const;

// private:
//   std::shared_ptr<ailoy::tvm_language_model_t> inner_;
// };

// // Factory function
// std::unique_ptr<TVMLanguageModel>
// create_tvm_language_model(const std::string &lib_filename,
//                           std::unique_ptr<DLDevice> device);
