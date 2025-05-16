#include <dlpack/dlpack.h>

#include "value.hpp"

// Convert IEEE 754 float16 (uint16_t) to float32
float float16_to_float32(uint16_t h);

void print_dl_tensor(const DLTensor *tensor);

void print_ndarray(std::shared_ptr<ailoy::ndarray_t> arr);
