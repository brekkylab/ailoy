#pragma once

#include <tvm/runtime/packed_func.h>

#include "module.hpp"
#include "tvm_model.hpp"
#include "value.hpp"

namespace ailoy {

class tvm_embedding_model_t : public object_t {
public:
  tvm_embedding_model_t(const std::string &model_name,
                        const std::string &quantization, DLDevice device);

  void postprocess_embedding_ndarray(const tvm::runtime::NDArray &from,
                                     tvm::runtime::NDArray &to);

  const tvm::runtime::NDArray infer(std::vector<int> tokens);

  std::filesystem::path get_model_path() const {
    return engine_->get_model_path();
  }

private:
  tvm::runtime::PackedFunc fprefill_;
  std::shared_ptr<tvm_model_t> engine_ = nullptr;
};

component_or_error_t
create_tvm_embedding_model_component(std::shared_ptr<const value_t> attrs);

} // namespace ailoy
