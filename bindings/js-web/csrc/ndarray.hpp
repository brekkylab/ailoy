#pragma once

#include <emscripten.h>
#include <emscripten/val.h>

#include "value.hpp"

using namespace emscripten;

class js_ndarray_t {
public:
  js_ndarray_t(const val &params) {
    // shape
    if (!params.hasOwnProperty("shape") ||
        !params["shape"].instanceof(val::global("Array"))) {
      throw std::runtime_error("object must have a shape array");
    }
    val shape = params["shape"];
    int shape_length = shape["length"].as<int>();
    shape_.resize(shape_length);
    for (int i = 0; i < shape_length; i++) {
      shape_[i] = shape[i].as<size_t>();
    }

    // dtype
    if (!params.hasOwnProperty("dtype") ||
        params["dtype"].typeOf().as<std::string>() != "string") {
      throw std::runtime_error("object must have a dtype string");
    }
    dtype_ = params["dtype"].as<std::string>();

    // data (TypedArray)
    if (!params.hasOwnProperty("data") || !is_typed_array(params["data"])) {
      throw std::runtime_error("object must have a data TypedArray");
    }
    data_ = params["data"];

    if (!check_dtype_match()) {
      throw std::runtime_error("Data type does not match to provided dtype");
    }

    // verify data size
    if (dtype_bytes() == 0) {
      throw std::runtime_error("Unsupported dtype: " + dtype_);
    }
    size_t nbytes = std::accumulate(shape_.begin(), shape_.end(), 1ULL,
                                    std::multiplies<size_t>()) *
                    dtype_bytes();
    if (data_["byteLength"].as<size_t>() != nbytes) {
      throw std::runtime_error(
          "Data buffer size doesn't match shape and dtype");
    }
  }

  std::shared_ptr<ailoy::ndarray_t> to_ailoy_ndarray_t() {
    auto ndarray = std::make_shared<ailoy::ndarray_t>();
    ndarray->shape = shape_;
    ndarray->dtype = {
        .code = dtype_code(),
        .bits = static_cast<uint8_t>(dtype_bytes() * 8),
        .lanes = 1,
    };

    // Copy data from TypedArray
    val buffer = data_["buffer"];
    int byte_offset = data_["byteOffset"].as<int>();
    int byte_length = data_["byteLength"].as<int>();

    ndarray->data.resize(byte_length);

    // Create Uint8Array view to copy bytes
    val uint8_view =
        val::global("Uint8Array").new_(buffer, byte_offset, byte_length);
    for (int i = 0; i < byte_length; i++) {
      ndarray->data[i] = static_cast<uint8_t>(uint8_view[i].as<int>());
    }

    return ndarray;
  }

  val get_shape() {
    val shape = val::array();
    for (size_t dim : shape_) {
      shape.call<void>("push", val(static_cast<int>(dim)));
    }
    return shape;
  }

  std::string get_dtype() { return dtype_; }

  val get_data() { return data_; }

private:
  bool is_typed_array(const val &v) {
    return v.instanceof(val::global("Int8Array")) ||
           v.instanceof(val::global("Uint8Array")) ||
           v.instanceof(val::global("Int16Array")) ||
           v.instanceof(val::global("Uint16Array")) ||
           v.instanceof(val::global("Int32Array")) ||
           v.instanceof(val::global("Uint32Array")) ||
           v.instanceof(val::global("Float32Array")) ||
           v.instanceof(val::global("Float64Array")) ||
           v.instanceof(val::global("BigInt64Array")) ||
           v.instanceof(val::global("BigUint64Array"));
  }

  uint8_t dtype_code() {
    if (dtype_ == "int8" || dtype_ == "int16" || dtype_ == "int32" ||
        dtype_ == "int64") {
      return kDLInt;
    } else if (dtype_ == "uint8" || dtype_ == "uint16" || dtype_ == "uint32" ||
               dtype_ == "uint64") {
      return kDLUInt;
    } else if (dtype_ == "float32" || dtype_ == "float64") {
      return kDLFloat;
    } else {
      throw std::runtime_error("unknown dtype: " + dtype_);
    }
  }

  uint8_t dtype_bytes() {
    if (dtype_ == "int8" || dtype_ == "uint8") {
      return 1;
    } else if (dtype_ == "int16" || dtype_ == "uint16") {
      return 2;
    } else if (dtype_ == "float32" || dtype_ == "int32" || dtype_ == "uint32") {
      return 4;
    } else if (dtype_ == "float64" || dtype_ == "int64" || dtype_ == "uint64") {
      return 8;
    } else {
      return 0;
    }
  }

  bool check_dtype_match() {
    std::string constructor_name =
        data_["constructor"]["name"].as<std::string>();

    if (dtype_ == "int8")
      return constructor_name == "Int8Array";
    else if (dtype_ == "int16")
      return constructor_name == "Int16Array";
    else if (dtype_ == "int32")
      return constructor_name == "Int32Array";
    else if (dtype_ == "int64")
      return constructor_name == "BigInt64Array";
    else if (dtype_ == "uint8")
      return constructor_name == "Uint8Array";
    else if (dtype_ == "uint16")
      return constructor_name == "Uint16Array";
    else if (dtype_ == "uint32")
      return constructor_name == "Uint32Array";
    else if (dtype_ == "uint64")
      return constructor_name == "BigUint64Array";
    else if (dtype_ == "float32")
      return constructor_name == "Float32Array";
    else if (dtype_ == "float64")
      return constructor_name == "Float64Array";
    else
      return false;
  }

  std::vector<size_t> shape_;
  std::string dtype_;
  val data_;
};
