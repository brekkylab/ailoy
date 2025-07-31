#include <emscripten/val.h>

#include "value.hpp"

using emscripten::val;

std::shared_ptr<ailoy::value_t> from_js_val(const val &arg) {
  if (arg.isUndefined() || arg.isNull()) {
    return ailoy::create<ailoy::null_t>();
  } else if (arg.isTrue() || arg.isFalse()) {
    return ailoy::create<ailoy::bool_t>(arg.as<bool>());
  } else if (arg.isNumber()) {
    double d = arg.as<double>();
    if (std::trunc(d) == d) {
      return ailoy::create<ailoy::int_t>(static_cast<int64_t>(d));
    } else {
      return ailoy::create<ailoy::float_t>(d);
    }
  } else if (arg.typeOf().as<std::string>() == "string") {
    return ailoy::create<ailoy::string_t>(arg.as<std::string>());
  } else if (val::global("ArrayBuffer").instanceof(arg) ||
             val::global("Uint8Array").instanceof(arg)) {
    // ArrayBuffer/TypedArray: extract bytes
    std::vector<uint8_t> buffer =
        emscripten::convertJSArrayToNumberVector<uint8_t>(arg);
    return ailoy::create<ailoy::bytes_t>(buffer);
  } else if (arg.isArray()) {
    auto vec = ailoy::create<ailoy::vector_t>();
    unsigned len = arg["length"].as<unsigned>();
    for (unsigned i = 0; i < len; ++i)
      vec->push_back(from_js_val(arg[i]));
    return vec;
  } else if (arg.typeOf().as<std::string>() == "object") {
    auto map = ailoy::create<ailoy::map_t>();
    auto keys = val::global("Object").call<val>("keys", arg);
    unsigned len = keys["length"].as<unsigned>();
    for (unsigned i = 0; i < len; ++i) {
      std::string k = keys[i].as<std::string>();
      auto v = arg[k];
      (*map)[k] = from_js_val(v);
    }
    return map;
  }
  throw std::runtime_error(
      "Unsupported JS type for conversion to ailoy::value_t");
}

val to_js_val(std::shared_ptr<ailoy::value_t> value) {
  if (value->is_type_of<ailoy::null_t>()) {
    return val::null();
  } else if (value->is_type_of<ailoy::string_t>()) {
    return val(*value->as<ailoy::string_t>());
  } else if (value->is_type_of<ailoy::bool_t>()) {
    return val(*value->as<ailoy::bool_t>());
  } else if (value->is_type_of<ailoy::int_t>()) {
    return val(
        static_cast<double>(*value->as<ailoy::int_t>())); // JS: number (int)
  } else if (value->is_type_of<ailoy::float_t>()) {
    return val(*value->as<ailoy::float_t>()); // JS: number (float)
  } else if (value->is_type_of<ailoy::bytes_t>()) {
    // Convert to Uint8Array
    const auto &bytes = *value->as<ailoy::bytes_t>();
    val uint8arr = val::global("Uint8Array").new_(bytes.size());
    for (size_t i = 0; i < bytes.size(); ++i)
      uint8arr.set(i, bytes[i]);
    return uint8arr;
  } else if (value->is_type_of<ailoy::vector_t>()) {
    const auto &vec = *value->as<ailoy::vector_t>();
    val arr = val::array();
    for (size_t i = 0; i < vec.size(); ++i)
      arr.set(i, to_js_val(vec[i]));
    return arr;
  } else if (value->is_type_of<ailoy::map_t>()) {
    const auto &map = *value->as<ailoy::map_t>();
    val obj = val::object();
    for (const auto &[k, v] : *map) {
      obj.set(k, to_js_val(v));
    }
    return obj;
  }
  throw std::runtime_error("Unsupported value_t type for conversion to JS");
}
