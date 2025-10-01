#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include <rust/cxx.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/ndarray.h>

#include "tvm_runtime.hpp"

// Forward Declaration for cxx_bridge.rs.h
struct DLPackTensor;

namespace ailoy {

struct CacheContents;

class tvm_embedding_model_t {
public:
  tvm_embedding_model_t(CacheContents &contents, DLDevice device);

  void extract_ndarray_part(const tvm::runtime::NDArray &from,
                            tvm::runtime::NDArray &to);

  const tvm::runtime::NDArray infer(std::vector<int> tokens);

  DLPackTensor infer_from_rs(rust::Slice<const uint32_t> tokens);

private:
  std::unique_ptr<tvm_runtime_t> rt_;

  tvm::ffi::Function fprefill_;

  mutable std::mutex m_;
};

std::unique_ptr<tvm_embedding_model_t>
create_tvm_embedding_model(ailoy::CacheContents &contents,
                           std::unique_ptr<DLDevice> device);

} // namespace ailoy
