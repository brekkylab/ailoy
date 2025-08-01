#pragma once

#include <emscripten.h>
#include <emscripten/val.h>

#include "ndarray.hpp"
#include "value.hpp"

using namespace emscripten;

static std::shared_ptr<ailoy::value_t> from_em_val(const val &arg) {
  if (arg.isUndefined()) {
    // JS undefined is considered same as null
    return ailoy::create<ailoy::null_t>();
  } else if (arg.isNull()) {
    return ailoy::create<ailoy::null_t>();
  } else if (arg.typeOf().as<std::string>() == "boolean") {
    return ailoy::create<ailoy::bool_t>(arg.as<bool>());
  } else if (arg.typeOf().as<std::string>() == "number") {
    double val_double = arg.as<double>();
    if (std::trunc(val_double) == val_double) {
      return ailoy::create<ailoy::int_t>(static_cast<int64_t>(val_double));
    } else {
      return ailoy::create<ailoy::double_t>(val_double);
    }
  } else if (arg.typeOf().as<std::string>() == "bigint") {
    // BigInt handling - convert to string if too large for int64
    try {
      // Try to convert to number first
      double num_val = arg.call<double>("valueOf");
      // Use double constants to avoid implicit conversion warnings
      constexpr double MAX_SAFE_INT64 = 9223372036854775808.0;  // 2^63
      constexpr double MIN_SAFE_INT64 = -9223372036854775808.0; // -2^63

      if (num_val >= MIN_SAFE_INT64 && num_val <= MAX_SAFE_INT64 &&
          std::trunc(num_val) == num_val) {
        return ailoy::create<ailoy::int_t>(static_cast<int64_t>(num_val));
      } else {
        // Fall back to string representation
        return ailoy::create<ailoy::string_t>(
            arg.call<std::string>("toString"));
      }
    } catch (...) {
      return ailoy::create<ailoy::string_t>(arg.call<std::string>("toString"));
    }
  } else if (arg.typeOf().as<std::string>() == "string") {
    return ailoy::create<ailoy::string_t>(arg.as<std::string>());
  } else if (arg.instanceof(val::global("ArrayBuffer"))) {
    // ArrayBuffer handling
    val uint8_view = val::global("Uint8Array").new_(arg);
    int length = uint8_view["byteLength"].as<int>();

    std::string bytes_data;
    bytes_data.reserve(length);

    for (int i = 0; i < length; i++) {
      bytes_data.push_back(static_cast<char>(uint8_view[i].as<uint8_t>()));
    }

    return ailoy::create<ailoy::bytes_t>(std::move(bytes_data));
  } else if (arg.instanceof(val::global("Int8Array")) ||
             arg.instanceof(val::global("Uint8Array")) ||
             arg.instanceof(val::global("Int16Array")) ||
             arg.instanceof(val::global("Uint16Array")) ||
             arg.instanceof(val::global("Int32Array")) ||
             arg.instanceof(val::global("Uint32Array")) ||
             arg.instanceof(val::global("Float32Array")) ||
             arg.instanceof(val::global("Float64Array")) ||
             arg.instanceof(val::global("BigInt64Array")) ||
             arg.instanceof(val::global("BigUint64Array"))) {

    // TypedArray handling
    val buffer = arg["buffer"];
    int byte_length = arg["byteLength"].as<int>();
    int element_length = arg["length"].as<int>();

    auto ndarray = ailoy::create<ailoy::ndarray_t>();
    ndarray->data.resize(byte_length);
    ndarray->shape = {static_cast<size_t>(element_length)}; // only 1-D array

    // Copy data from TypedArray
    val uint8_view =
        val::global("Uint8Array")
            .new_(buffer, arg["byteOffset"].as<int>(), byte_length);
    for (int i = 0; i < byte_length; i++) {
      ndarray->data[i] = static_cast<uint8_t>(uint8_view[i].as<int>());
    }

    // Set dtype based on TypedArray type
    std::string constructor_name = arg["constructor"]["name"].as<std::string>();
    if (constructor_name == "Int8Array") {
      ndarray->dtype = {kDLInt, 8, 1};
    } else if (constructor_name == "Uint8Array") {
      ndarray->dtype = {kDLUInt, 8, 1};
    } else if (constructor_name == "Int16Array") {
      ndarray->dtype = {kDLInt, 16, 1};
    } else if (constructor_name == "Uint16Array") {
      ndarray->dtype = {kDLUInt, 16, 1};
    } else if (constructor_name == "Int32Array") {
      ndarray->dtype = {kDLInt, 32, 1};
    } else if (constructor_name == "Uint32Array") {
      ndarray->dtype = {kDLUInt, 32, 1};
    } else if (constructor_name == "Float32Array") {
      ndarray->dtype = {kDLFloat, 32, 1};
    } else if (constructor_name == "Float64Array") {
      ndarray->dtype = {kDLFloat, 64, 1};
    } else if (constructor_name == "BigInt64Array") {
      ndarray->dtype = {kDLInt, 64, 1};
    } else if (constructor_name == "BigUint64Array") {
      ndarray->dtype = {kDLUInt, 64, 1};
    } else {
      throw std::runtime_error(
          "Unsupported TypedArray type for ndarray_t conversion");
    }

    return ndarray;
  } else if (arg.instanceof(val::global("Array"))) {
    // Array handling
    int length = arg["length"].as<int>();
    auto vec = ailoy::create<ailoy::array_t>();
    vec->reserve(length);

    for (int i = 0; i < length; i++) {
      val elem = arg[i];
      vec->push_back(from_em_val(elem));
    }

    return vec;
  } else if (arg.hasOwnProperty("__class__") &&
             arg["__class__"].as<std::string>() == "NDArray") {
    js_ndarray_t ndarray_obj = arg.as<js_ndarray_t>();
    return ndarray_obj.to_ailoy_ndarray_t();
  } else if (arg.typeOf().as<std::string>() == "object" &&
             !arg.instanceof(val::global("Function")) &&
             !arg.instanceof(val::global("Promise"))) {

    // Object handling
    auto map = ailoy::create<ailoy::map_t>();
    val keys = val::global("Object").call<val>("keys", arg);
    int keys_length = keys["length"].as<int>();

    for (int i = 0; i < keys_length; i++) {
      std::string key = keys[i].as<std::string>();
      val value = arg[key];
      (*map)[key] = from_em_val(value);
    }

    return map;
  } else {
    throw std::runtime_error(
        "Unsupported Emscripten val type for conversion to value_t");
  }
}

static val to_em_val(std::shared_ptr<ailoy::value_t> v) {
  std::string type = v->get_type();

  if (v->is_type_of<ailoy::null_t>()) {
    return val::null();
  } else if (v->is_type_of<ailoy::string_t>()) {
    std::string str = *v->as<ailoy::string_t>();
    return val(str);
  } else if (v->is_type_of<ailoy::bool_t>()) {
    bool b = *v->as<ailoy::bool_t>();
    return val(b);
  } else if (v->is_type_of<ailoy::int_t>()) {
    int i = *v->as<ailoy::int_t>();
    return val(i);
  } else if (v->is_type_of<ailoy::uint_t>()) {
    uint32_t i = *v->as<ailoy::uint_t>();
    return val(i);
  } else if (v->is_type_of<ailoy::float_t>()) {
    float f = *v->as<ailoy::float_t>();
    return val(f);
  } else if (v->is_type_of<ailoy::double_t>()) {
    double d = *v->as<ailoy::double_t>();
    return val(d);
  } else if (v->is_type_of<ailoy::bytes_t>()) {
    auto bytes = v->as<ailoy::bytes_t>();

    // Create ArrayBuffer and copy data
    val array_buffer =
        val::global("ArrayBuffer").new_(static_cast<int>(bytes->size()));
    val uint8_view = val::global("Uint8Array").new_(array_buffer);

    for (size_t i = 0; i < bytes->size(); i++) {
      uint8_view.set(i, static_cast<uint8_t>((*bytes)[i]));
    }

    return array_buffer;
  } else if (v->is_type_of<ailoy::array_t>()) {
    auto arr = v->as<ailoy::array_t>();
    val js_arr = val::array();

    for (size_t i = 0; i < arr->size(); i++) {
      val item = to_em_val((*arr)[i]);
      js_arr.call<void>("push", item);
    }

    return js_arr;
  } else if (v->is_type_of<ailoy::map_t>()) {
    auto map = v->as<ailoy::map_t>();
    val js_obj = val::object();

    for (const auto &[key, value] : *map) {
      js_obj.set(key, to_em_val(value));
    }

    return js_obj;
  } else if (v->is_type_of<ailoy::ndarray_t>()) {
    auto ndarray = v->as<ailoy::ndarray_t>();

    // Create shape array
    val shape = val::array();
    for (size_t dim : ndarray->shape) {
      shape.call<void>("push", val(static_cast<int>(dim)));
    }

    // Calculate number of elements
    size_t num_elements =
        std::accumulate(ndarray->shape.begin(), ndarray->shape.end(), 1ULL,
                        std::multiplies<size_t>());

    // Create ArrayBuffer and copy data
    val array_buffer =
        val::global("ArrayBuffer").new_(static_cast<int>(ndarray->data.size()));
    val uint8_view = val::global("Uint8Array").new_(array_buffer);

    for (size_t i = 0; i < ndarray->data.size(); i++) {
      uint8_view.set(i, static_cast<uint8_t>(ndarray->data[i]));
    }

    // Create appropriate TypedArray based on dtype
    std::string dtype_str;
    val typed_array = val::undefined();

    if (ndarray->dtype.code == kDLInt) {
      switch (ndarray->dtype.bits) {
      case 8:
        dtype_str = "int8";
        typed_array = val::global("Int8Array").new_(array_buffer);
        break;
      case 16:
        dtype_str = "int16";
        typed_array = val::global("Int16Array").new_(array_buffer);
        break;
      case 32:
        dtype_str = "int32";
        typed_array = val::global("Int32Array").new_(array_buffer);
        break;
      case 64:
        dtype_str = "int64";
        typed_array = val::global("BigInt64Array").new_(array_buffer);
        break;
      default:
        throw std::runtime_error("unsupported int bits");
      }
    } else if (ndarray->dtype.code == kDLUInt) {
      switch (ndarray->dtype.bits) {
      case 8:
        dtype_str = "uint8";
        typed_array = val::global("Uint8Array").new_(array_buffer);
        break;
      case 16:
        dtype_str = "uint16";
        typed_array = val::global("Uint16Array").new_(array_buffer);
        break;
      case 32:
        dtype_str = "uint32";
        typed_array = val::global("Uint32Array").new_(array_buffer);
        break;
      case 64:
        dtype_str = "uint64";
        typed_array = val::global("BigUint64Array").new_(array_buffer);
        break;
      default:
        throw std::runtime_error("unsupported uint bits");
      }
    } else if (ndarray->dtype.code == kDLFloat) {
      switch (ndarray->dtype.bits) {
      case 32:
        dtype_str = "float32";
        typed_array = val::global("Float32Array").new_(array_buffer);
        break;
      case 64:
        dtype_str = "float64";
        typed_array = val::global("Float64Array").new_(array_buffer);
        break;
      default:
        throw std::runtime_error("unsupported float bits");
      }
    } else {
      throw std::runtime_error("Unsupported dtype code");
    }

    // Create parameters object for js_ndarray_t constructor
    val params = val::object();
    params.set("shape", shape);
    params.set("dtype", val(dtype_str));
    params.set("data", typed_array);

    return val::global("Module")["NDArray"].new_(params);
  } else {
    return val("[Value: " + type + "]");
  }
}
