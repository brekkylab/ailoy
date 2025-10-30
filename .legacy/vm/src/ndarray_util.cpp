#include "ndarray_util.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>

float float16_to_float32(uint16_t h) {
  uint16_t h_sign = (h & 0x8000u);
  uint16_t h_exp = (h & 0x7C00u);
  uint16_t h_mant = (h & 0x03FFu);

  uint32_t f_sign = (uint32_t)(h_sign) << 16;
  if (h_exp == 0x0000) {
    // subnormal
    if (h_mant == 0) {
      return std::bit_cast<float>(f_sign);
    } else {
      // Convert subnormal to normalized float
      float mant = h_mant / 1024.0f;
      float val = std::ldexp(mant, -14);
      return std::bit_cast<float>(f_sign | std::bit_cast<uint32_t>(val));
    }
  } else if (h_exp == 0x7C00) {
    // inf or NaN
    return std::bit_cast<float>(f_sign | 0x7F800000u | (h_mant << 13));
  } else {
    // normal number
    uint32_t f_exp = ((h_exp >> 10) + 112) << 23;
    uint32_t f_mant = (uint32_t)(h_mant) << 13;
    return std::bit_cast<float>(f_sign | f_exp | f_mant);
  }
}

void print_ndarray(std::shared_ptr<ailoy::ndarray_t> arr) {
  using namespace std;

  const auto &shape = arr->shape;
  const auto ndim = shape.size();
  const auto &dtype = arr->dtype;
  const auto &raw = arr->data;

  // Print shape
  cout << "ndarray of shape (";
  for (size_t i = 0; i < ndim; ++i) {
    cout << shape[i];
    if (i + 1 < ndim)
      cout << ", ";
  }
  if (ndim == 1)
    cout << ",";
  cout << "), dtype=";

  // Formatter
  auto print_val = [](float val) {
    cout << fixed << setw(8) << setprecision(4) << val;
  };

  // Handle float32
  if (dtype.code == kDLFloat && dtype.bits == 32) {
    cout << "float32\n";
    const float *data = reinterpret_cast<const float *>(raw.data());
    size_t size = raw.size() / sizeof(float);

    if (ndim == 1) {
      cout << "[";
      for (size_t i = 0; i < std::min(size_t(10), shape[0]); ++i) {
        print_val(data[i]);
        if (i + 1 < shape[0])
          cout << ", ";
      }
      if (shape[0] > 10)
        cout << " ...";
      cout << "]\n";
    } else if (ndim == 2) {
      size_t rows = shape[0], cols = shape[1];
      for (size_t i = 0; i < std::min(rows, size_t(10)); ++i) {
        cout << "[";
        for (size_t j = 0; j < std::min(cols, size_t(10)); ++j) {
          print_val(data[i * cols + j]);
          if (j + 1 < cols)
            cout << ", ";
        }
        if (cols > 10)
          cout << " ...";
        cout << "]\n";
      }
      if (rows > 10)
        cout << "...\n";
    }

    // Handle float16
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    cout << "float16\n";
    const uint16_t *data = reinterpret_cast<const uint16_t *>(raw.data());
    size_t size = raw.size() / sizeof(uint16_t);

    if (ndim == 1) {
      cout << "[";
      for (size_t i = 0; i < std::min(size_t(10), shape[0]); ++i) {
        print_val(float16_to_float32(data[i]));
        if (i + 1 < shape[0])
          cout << ", ";
      }
      if (shape[0] > 10)
        cout << " ...";
      cout << "]\n";
    } else if (ndim == 2) {
      size_t rows = shape[0], cols = shape[1];
      for (size_t i = 0; i < std::min(rows, size_t(10)); ++i) {
        cout << "[";
        for (size_t j = 0; j < std::min(cols, size_t(10)); ++j) {
          print_val(float16_to_float32(data[i * cols + j]));
          if (j + 1 < cols)
            cout << ", ";
        }
        if (cols > 10)
          cout << " ...";
        cout << "]\n";
      }
      if (rows > 10)
        cout << "...\n";
    }

  } else {
    cout << "unsupported (code=" << int(dtype.code)
         << ", bits=" << int(dtype.bits) << ")\n";
  }
}

void print_dl_tensor(const DLTensor *tensor) {
  if (tensor->device.device_type != kDLCPU) {
    std::cerr << "Only CPU tensors are supported.\n";
    return;
  }

  int ndim = tensor->ndim;
  int64_t *shape = tensor->shape;
  int64_t offset_bytes = tensor->byte_offset;
  size_t total_elements = 1;
  for (int i = 0; i < ndim; ++i)
    total_elements *= shape[i];

  std::ostringstream shape_str;
  shape_str << "(";
  for (int i = 0; i < ndim; ++i) {
    shape_str << shape[i];
    if (i != ndim - 1)
      shape_str << ", ";
  }
  if (ndim == 1)
    shape_str << ",";
  shape_str << ")";

  std::cout << "DLTensor of shape " << shape_str.str() << ", dtype=";

  if (tensor->dtype.code == kDLFloat && tensor->dtype.bits == 32) {
    std::cout << "float32\n";
    float *data =
        reinterpret_cast<float *>((uint8_t *)tensor->data + offset_bytes);

    for (int64_t i = 0; i < std::min<int64_t>(total_elements, 10); ++i) {
      std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
    }
    if (total_elements > 10)
      std::cout << "...";
    std::cout << "\n";

  } else if (tensor->dtype.code == kDLFloat && tensor->dtype.bits == 16) {
    std::cout << "float16\n";
    uint16_t *raw =
        reinterpret_cast<uint16_t *>((uint8_t *)tensor->data + offset_bytes);

    for (int64_t i = 0; i < std::min<int64_t>(total_elements, 10); ++i) {
      float val = float16_to_float32(raw[i]);
      std::cout << std::fixed << std::setprecision(4) << val << " ";
    }
    if (total_elements > 10)
      std::cout << "...";
    std::cout << "\n";
  } else {
    std::cerr << "Unsupported dtype: code=" << (int)tensor->dtype.code
              << " bits=" << (int)tensor->dtype.bits << "\n";
  }
}