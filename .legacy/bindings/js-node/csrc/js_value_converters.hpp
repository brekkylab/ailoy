#pragma once

#include <napi.h>

#include "js_ndarray.hpp"
#include "value.hpp"

static std::shared_ptr<ailoy::value_t> from_napi_value(Napi::Env env,
                                                       Napi::Value arg) {
  if (arg.IsUndefined()) {
    // JS undefined is considered same as null
    return ailoy::create<ailoy::null_t>();
  } else if (arg.IsNull()) {
    return ailoy::create<ailoy::null_t>();
  } else if (arg.IsBoolean()) {
    return ailoy::create<ailoy::bool_t>(arg.As<Napi::Boolean>().Value());
  } else if (arg.IsNumber()) {
    Napi::Number num = arg.As<Napi::Number>();
    double val = num.DoubleValue();
    if (std::trunc(val) == val) {
      return ailoy::create<ailoy::int_t>(num.Int64Value());
    } else {
      return ailoy::create<ailoy::double_t>(val);
    }
  } else if (arg.IsBigInt()) {
    bool lossless;
    int64_t val = arg.As<Napi::BigInt>().Int64Value(&lossless);
    if (lossless) {
      return ailoy::create<ailoy::int_t>(val);
    } else {
      return ailoy::create<ailoy::string_t>(arg.ToString().Utf8Value());
    }
  } else if (arg.IsString()) {
    return ailoy::create<ailoy::string_t>(arg.As<Napi::String>().Utf8Value());
  } else if (arg.IsArrayBuffer()) {
    Napi::ArrayBuffer ab = arg.As<Napi::ArrayBuffer>();
    return ailoy::create<ailoy::bytes_t>(
        std::string(static_cast<char *>(ab.Data()), ab.ByteLength()));
  } else if (arg.IsTypedArray()) {
    Napi::TypedArray ta = arg.As<Napi::TypedArray>();
    Napi::ArrayBuffer ab = ta.ArrayBuffer();
    const auto ab_data =
        std::string(static_cast<char *>(ab.Data()), ab.ByteLength());

    auto ndarray = ailoy::create<ailoy::ndarray_t>();
    ndarray->data.resize(ab.ByteLength());
    std::copy(ab_data.begin(), ab_data.end(), ndarray->data.data());
    ndarray->shape = {ta.ElementLength()}; // only 1-D array
    switch (ta.TypedArrayType()) {
    case napi_int8_array:
      ndarray->dtype = {kDLInt, 8, 1};
      break;
    case napi_uint8_array:
      ndarray->dtype = {kDLUInt, 8, 1};
      break;
    case napi_int16_array:
      ndarray->dtype = {kDLInt, 16, 1};
      break;
    case napi_uint16_array:
      ndarray->dtype = {kDLUInt, 16, 1};
      break;
    case napi_int32_array:
      ndarray->dtype = {kDLInt, 32, 1};
      break;
    case napi_uint32_array:
      ndarray->dtype = {kDLUInt, 32, 1};
      break;
    case napi_float32_array:
      ndarray->dtype = {kDLFloat, 32, 1};
      break;
    case napi_float64_array:
      ndarray->dtype = {kDLFloat, 64, 1};
      break;
    case napi_bigint64_array:
      ndarray->dtype = {kDLInt, 64, 1};
      break;
    case napi_biguint64_array:
      ndarray->dtype = {kDLUInt, 64, 1};
      break;
    default:
      Napi::Error::New(env,
                       "Unsupported TypedArray type for ndarray_t conversion")
          .ThrowAsJavaScriptException();
      return nullptr;
    }
    return ndarray;
  } else if (arg.IsArray()) {
    Napi::Array arr = arg.As<Napi::Array>();
    auto vec = ailoy::create<ailoy::array_t>();
    for (size_t i = 0; i < arr.Length(); i++) {
      Napi::Value elem = arr[i];
      vec->push_back(from_napi_value(env, elem));
    }
    return vec;
  } else if (arg.IsObject() && arg.As<Napi::Object>().InstanceOf(
                                   js_ndarray_t::constructor.Value())) {
    js_ndarray_t *js_ndarray =
        Napi::ObjectWrap<js_ndarray_t>::Unwrap(arg.As<Napi::Object>());
    return js_ndarray->to_ailoy_ndarray_t();
  } else if (arg.IsObject() && !arg.IsFunction() && !arg.IsPromise()) {
    Napi::Object obj = arg.As<Napi::Object>();
    auto map = ailoy::create<ailoy::map_t>();
    Napi::Array keys = obj.GetPropertyNames();
    for (size_t i = 0; i < keys.Length(); i++) {
      Napi::Value key = keys[i];
      std::string key_str = key.As<Napi::String>().Utf8Value();
      Napi::Value val = obj.Get(key);
      (*map)[key_str] = from_napi_value(env, val);
    }
    return map;
  } else {
    Napi::Error::New(env, "Unsupported Napi type for conversion to value_t")
        .ThrowAsJavaScriptException();
    return nullptr;
  }
}

static Napi::Value to_napi_value(Napi::Env env,
                                 std::shared_ptr<ailoy::value_t> val) {
  std::string type = val->get_type();
  if (val->is_type_of<ailoy::null_t>()) {
    return env.Null();
  } else if (val->is_type_of<ailoy::string_t>()) {
    return Napi::String::New(env, *val->as<ailoy::string_t>());
  } else if (val->is_type_of<ailoy::bool_t>()) {
    return Napi::Boolean::New(env, *val->as<ailoy::bool_t>());
  } else if (val->is_type_of<ailoy::int_t>()) {
    return Napi::Number::New(env, *val->as<ailoy::int_t>());
  } else if (val->is_type_of<ailoy::uint_t>()) {
    return Napi::Number::New(env, *val->as<ailoy::uint_t>());
  } else if (val->is_type_of<ailoy::float_t>()) {
    return Napi::Number::New(env, *val->as<ailoy::float_t>());
  } else if (val->is_type_of<ailoy::double_t>()) {
    return Napi::Number::New(env, *val->as<ailoy::double_t>());
  } else if (val->is_type_of<ailoy::bytes_t>()) {
    auto bytes = val->as<ailoy::bytes_t>();
    return Napi::String::New(env, std::string(bytes->begin(), bytes->end()));
  } else if (val->is_type_of<ailoy::array_t>()) {
    auto arr = val->as<ailoy::array_t>();
    Napi::Array js_arr = Napi::Array::New(env, arr->size());
    for (size_t i = 0; i < arr->size(); i++) {
      js_arr[i] = to_napi_value(env, (*arr)[i]);
    }
    return js_arr;
  } else if (val->is_type_of<ailoy::map_t>()) {
    auto map = val->as<ailoy::map_t>();
    Napi::Object js_obj = Napi::Object::New(env);
    for (const auto &[key, val] : *map) {
      js_obj.Set(key, to_napi_value(env, val));
    }
    return js_obj;
  } else if (val->is_type_of<ailoy::ndarray_t>()) {
    auto ndarray = val->as<ailoy::ndarray_t>();
    Napi::Object params = Napi::Object::New(env);

    // shape
    Napi::Array shape = Napi::Array::New(env, ndarray->shape.size());
    for (size_t i = 0; i < ndarray->shape.size(); i++) {
      shape[i] = Napi::Number::New(env, ndarray->shape[i]);
    }
    params.Set("shape", shape);

    // dtype and TypedArray creation
    std::string dtype_str;
    Napi::TypedArray data;
    size_t num_elements =
        std::accumulate(ndarray->shape.begin(), ndarray->shape.end(), 1ULL,
                        std::multiplies<size_t>());

    Napi::ArrayBuffer ab = Napi::ArrayBuffer::New(env, ndarray->data.size());
    memcpy(ab.Data(), ndarray->data.data(), ndarray->data.size());

    if (ndarray->dtype.code == kDLInt) {
      switch (ndarray->dtype.bits) {
      case 8:
        dtype_str = "int8";
        data = Napi::Int8Array::New(env, num_elements, ab, 0);
        break;
      case 16:
        dtype_str = "int16";
        data = Napi::Int16Array::New(env, num_elements, ab, 0);
        break;
      case 32:
        dtype_str = "int32";
        data = Napi::Int32Array::New(env, num_elements, ab, 0);
        break;
      case 64:
        dtype_str = "int64";
        data = Napi::BigInt64Array::New(env, num_elements, ab, 0);
        break;
      default:
        Napi::Error::New(env, "unsupported int bits")
            .ThrowAsJavaScriptException();
      }
    } else if (ndarray->dtype.code == kDLUInt) {
      switch (ndarray->dtype.bits) {
      case 8:
        dtype_str = "uint8";
        data = Napi::Uint8Array::New(env, num_elements, ab, 0);
        break;
      case 16:
        dtype_str = "uint16";
        data = Napi::Uint16Array::New(env, num_elements, ab, 0);
        break;
      case 32:
        dtype_str = "uint32";
        data = Napi::Uint32Array::New(env, num_elements, ab, 0);
        break;
      case 64:
        dtype_str = "uint64";
        data = Napi::BigUint64Array::New(env, num_elements, ab, 0);
        break;
      default:
        Napi::Error::New(env, "unsupported uint bits")
            .ThrowAsJavaScriptException();
      }
    } else if (ndarray->dtype.code == kDLFloat) {
      switch (ndarray->dtype.bits) {
      case 32:
        dtype_str = "float32";
        data = Napi::Float32Array::New(env, num_elements, ab, 0);
        break;
      case 64:
        dtype_str = "float64";
        data = Napi::Float64Array::New(env, num_elements, ab, 0);
        break;
      default:
        Napi::Error::New(env, "unsupported float bits")
            .ThrowAsJavaScriptException();
      }
    } else {
      Napi::Error::New(env, "Unsupported dtype code")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
    params.Set("dtype", Napi::String::New(env, dtype_str));
    params.Set("data", data);

    // create js_ndarray_t instance
    Napi::Function ctor =
        js_ndarray_t::constructor.Value().As<Napi::Function>();
    Napi::Object js_ndarray = ctor.New({params});
    return js_ndarray;
  } else {
    return Napi::String::New(env, "[Value: " + type + "]");
  }
}
