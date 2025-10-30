#pragma once

#include <numeric>

#include <napi.h>

#include "value.hpp"

class js_ndarray_t : public Napi::ObjectWrap<js_ndarray_t> {
public:
  static Napi::FunctionReference constructor;

  static Napi::Function initialize(Napi::Env env) {
    auto ctor =
        DefineClass(env, "NDArray",
                    {
                        InstanceMethod("getShape", &js_ndarray_t::get_shape),
                        InstanceMethod("getDtype", &js_ndarray_t::get_dtype),
                        InstanceMethod("getData", &js_ndarray_t::get_data),
                    });
    constructor = Napi::Persistent(ctor);
    constructor.SuppressDestruct();
    return ctor;
  }

  js_ndarray_t(const Napi::CallbackInfo &info)
      : Napi::ObjectWrap<js_ndarray_t>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
      Napi::TypeError::New(
          env, "Expected a single object argument with shape, dtype, and data")
          .ThrowAsJavaScriptException();
      return;
    }

    Napi::Object arg = info[0].As<Napi::Object>();

    // shape
    if (!arg.Has("shape") || !arg.Get("shape").IsArray()) {
      Napi::TypeError::New(env, "object must have a shape array")
          .ThrowAsJavaScriptException();
      return;
    }
    Napi::Array shape = arg.Get("shape").As<Napi::Array>();
    shape_.resize(shape.Length());
    for (size_t i = 0; i < shape.Length(); i++) {
      shape_[i] = shape.Get(i).As<Napi::Number>().Int64Value();
    }

    // dtype
    if (!arg.Has("dtype") || !arg.Get("dtype").IsString()) {
      Napi::TypeError::New(env, "object must have a dtype string")
          .ThrowAsJavaScriptException();
      return;
    }
    dtype_ = arg.Get("dtype").As<Napi::String>().Utf8Value();

    // data (TypedArray)
    if (!arg.Has("data") || !arg.Get("data").IsTypedArray()) {
      Napi::TypeError::New(env, "object must have a data TypedArray")
          .ThrowAsJavaScriptException();
      return;
    }
    data_ = Napi::Reference<Napi::TypedArray>::New(
        arg.Get("data").As<Napi::TypedArray>(), 1);
    if (!check_dtype_match()) {
      Napi::TypeError::New(env, "Data type does not match to provided dtype")
          .ThrowAsJavaScriptException();
      return;
    }

    // verify data size
    if (dtype_bytes() == 0) {
      Napi::TypeError::New(env, "Unsupported dtype: " + dtype_)
          .ThrowAsJavaScriptException();
      return;
    }
    size_t nbytes = std::accumulate(shape_.begin(), shape_.end(), 1.0,
                                    std::multiplies<size_t>()) *
                    dtype_bytes();
    if (data_.Value().ByteLength() != nbytes) {
      Napi::TypeError::New(env,
                           "Data buffer size doesn't match shape and dtype")
          .ThrowAsJavaScriptException();
      return;
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
    Napi::ArrayBuffer ab = data_.Value().ArrayBuffer();
    const auto ab_data =
        std::string(static_cast<char *>(ab.Data()), ab.ByteLength());
    ndarray->data.resize(ab.ByteLength());
    std::copy(ab_data.begin(), ab_data.end(), ndarray->data.data());
    return ndarray;
  }

  Napi::Value get_shape(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    Napi::Array shape = Napi::Array::New(env, shape_.size());
    for (size_t i = 0; i < shape_.size(); i++) {
      shape.Set(i, Napi::Number::New(env, shape_[i]));
    }
    return shape;
  }

  Napi::Value get_dtype(const Napi::CallbackInfo &info) {
    return Napi::String::New(info.Env(), dtype_);
  }

  Napi::Value get_data(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    if (!data_.Value().IsEmpty()) {
      return data_.Value();
    } else {
      Napi::Error::New(env, "TypedArray data is invalid")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
  }

private:
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
    auto ta_type = data_.Value().TypedArrayType();
    if (dtype_ == "int8")
      return ta_type == napi_int8_array;
    else if (dtype_ == "int16")
      return ta_type == napi_int16_array;
    else if (dtype_ == "int32")
      return ta_type == napi_int32_array;
    else if (dtype_ == "int64")
      return ta_type == napi_bigint64_array;
    else if (dtype_ == "uint8")
      return ta_type == napi_uint8_array;
    else if (dtype_ == "uint16")
      return ta_type == napi_uint16_array;
    else if (dtype_ == "uint32")
      return ta_type == napi_uint32_array;
    else if (dtype_ == "uint64")
      return ta_type == napi_biguint64_array;
    else if (dtype_ == "float32")
      return ta_type == napi_float32_array;
    else if (dtype_ == "float64")
      return ta_type == napi_float64_array;
    else
      return false;
  }

  std::vector<size_t> shape_;
  std::string dtype_;
  Napi::Reference<Napi::TypedArray> data_;
};

Napi::FunctionReference js_ndarray_t::constructor;