#pragma once

#include "value.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pybind11 {
namespace detail {

template <> struct type_caster<std::shared_ptr<ailoy::value_t>> {
public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<ailoy::value_t>, _("value_t"));

  // Python → C++
  bool load(handle src, bool) {
    if (src.is_none()) {
      value = ailoy::create<ailoy::null_t>();
      return true;
    } else if (isinstance<py::bool_>(src)) {
      value = ailoy::create<ailoy::bool_t>(src.cast<bool>());
      return true;
    } else if (isinstance<py::int_>(src)) {
      value = ailoy::create<ailoy::int_t>(src.cast<int64_t>());
      return true;
    } else if (isinstance<py::float_>(src)) {
      value = ailoy::create<ailoy::double_t>(src.cast<double>());
      return true;
    } else if (isinstance<py::str>(src)) {
      value = ailoy::create<ailoy::string_t>(src.cast<std::string>());
      return true;
    } else if (isinstance<py::bytes>(src)) {
      value = ailoy::create<ailoy::bytes_t>(src.cast<std::string>());
      return true;
    } else if (isinstance<py::list>(src)) {
      auto arr = ailoy::create<ailoy::array_t>();
      for (auto item : src.cast<py::list>())
        arr->push_back(py::cast<std::shared_ptr<ailoy::value_t>>(item));
      value = arr;
      return true;
    } else if (isinstance<py::dict>(src)) {
      auto map = ailoy::create<ailoy::map_t>();
      for (auto item : src.cast<py::dict>()) {
        (*map)[item.first.cast<std::string>()] =
            py::cast<std::shared_ptr<ailoy::value_t>>(item.second);
      }
      value = map;
      return true;
    } else if (py::isinstance<py::array>(src)) {
      py::array arr = py::cast<py::array>(src);
      py::buffer_info info = arr.request();
      auto ndarray = ailoy::create<ailoy::ndarray_t>();
      ndarray->shape.assign(info.shape.begin(), info.shape.end());
      if (info.format == py::format_descriptor<int8_t>::format())
        ndarray->dtype = {kDLInt, 8, 1};
      else if (info.format == py::format_descriptor<int16_t>::format())
        ndarray->dtype = {kDLInt, 16, 1};
      else if (info.format == py::format_descriptor<int32_t>::format())
        ndarray->dtype = {kDLInt, 32, 1};
      else if (info.format == py::format_descriptor<int64_t>::format())
        ndarray->dtype = {kDLInt, 64, 1};
      else if (info.format == py::format_descriptor<uint8_t>::format())
        ndarray->dtype = {kDLUInt, 8, 1};
      else if (info.format == py::format_descriptor<uint16_t>::format())
        ndarray->dtype = {kDLUInt, 16, 1};
      else if (info.format == py::format_descriptor<uint32_t>::format())
        ndarray->dtype = {kDLUInt, 32, 1};
      else if (info.format == py::format_descriptor<uint64_t>::format())
        ndarray->dtype = {kDLUInt, 64, 1};
      else if (info.format == py::format_descriptor<float>::format())
        ndarray->dtype = {kDLFloat, 32, 1};
      else if (info.format == py::format_descriptor<double>::format())
        ndarray->dtype = {kDLFloat, 64, 1};
      else
        throw ailoy::exception("Unsupported numpy dtype for ndarray_t");
      ndarray->data.resize(info.size * info.itemsize);
      std::memcpy(ndarray->data.data(), info.ptr, info.size * info.itemsize);
      value = ndarray;
      return true;
    } else {
      return false;
    }
  }

  // C++ → Python
  static handle cast(std::shared_ptr<ailoy::value_t> val, return_value_policy,
                     handle) {
    if (val->is_type_of<ailoy::null_t>()) {
      return py::none().release();
    } else if (val->is_type_of<ailoy::bool_t>()) {
      return py::bool_(*val->as<ailoy::bool_t>()).release();
    } else if (val->is_type_of<ailoy::int_t>()) {
      return py::int_((int64_t)(*val->as<ailoy::int_t>())).release();
    } else if (val->is_type_of<ailoy::uint_t>()) {
      return py::int_((uint64_t)(*val->as<ailoy::uint_t>())).release();
    } else if (val->is_type_of<ailoy::float_t>()) {
      return py::float_((float_t)(*val->as<ailoy::float_t>())).release();
    } else if (val->is_type_of<ailoy::double_t>()) {
      return py::float_((double_t)(*val->as<ailoy::double_t>())).release();
    } else if (val->is_type_of<ailoy::string_t>()) {
      return py::str(*val->as<ailoy::string_t>()).release();
    } else if (val->is_type_of<ailoy::bytes_t>()) {
      const auto &b = *val->as<ailoy::bytes_t>();
      return py::bytes(reinterpret_cast<const char *>(b.data()), b.size())
          .release();
    } else if (val->is_type_of<ailoy::array_t>()) {
      py::list rv;
      for (auto &item : *val->as<ailoy::array_t>())
        rv.append(py::cast(item));
      return rv.release();
    } else if (val->is_type_of<ailoy::map_t>()) {
      py::dict rv;
      for (auto &[k, v] : *val->as<ailoy::map_t>())
        rv[py::str(k)] = py::cast(v);
      return rv.release();
    } else if (val->is_type_of<ailoy::ndarray_t>()) {
      auto arr = val->as<ailoy::ndarray_t>();
      std::string format;
      switch (arr->dtype.code) {
      case kDLInt:
        switch (arr->dtype.bits) {
        case 8:
          format = py::format_descriptor<int8_t>::format();
          break;
        case 16:
          format = py::format_descriptor<int16_t>::format();
          break;
        case 32:
          format = py::format_descriptor<int32_t>::format();
          break;
        case 64:
          format = py::format_descriptor<int64_t>::format();
          break;
        default:
          throw std::runtime_error("Unsupported int bits");
        }
        break;
      case kDLUInt:
        switch (arr->dtype.bits) {
        case 8:
          format = py::format_descriptor<uint8_t>::format();
          break;
        case 16:
          format = py::format_descriptor<uint16_t>::format();
          break;
        case 32:
          format = py::format_descriptor<uint32_t>::format();
          break;
        case 64:
          format = py::format_descriptor<uint64_t>::format();
          break;
        default:
          throw std::runtime_error("Unsupported uint bits");
        }
        break;
      case kDLFloat:
        switch (arr->dtype.bits) {
        case 32:
          format = py::format_descriptor<float>::format();
          break;
        case 64:
          format = py::format_descriptor<double>::format();
          break;
        default:
          throw std::runtime_error("Unsupported float bits");
        }
        break;
      default:
        throw std::runtime_error("Unsupported dtype code");
      }
      std::vector<size_t> shape = arr->shape;
      std::vector<size_t> strides(shape.size());
      size_t stride = arr->dtype.bits / 8;
      for (ssize_t i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
      }
      return py::array(
                 py::buffer_info(static_cast<void *>(arr->data.data()), // ptr
                                 arr->itemsize(), // itemsize
                                 format,          // format
                                 shape.size(),    // ndim
                                 shape,           // shape
                                 strides          // strides
                                 ))
          .release();
    } else {
      return py::none().release();
    }
  }
};

} // namespace detail
} // namespace pybind11