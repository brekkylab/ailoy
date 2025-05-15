/**
 * @file value.hpp
 * @brief Defines the core data container types used in the Ailoy system
 * @details
 * `value_t` provides a type-erased container abstraction and its various
 * concrete subclasses such as `int_t`, `string_t`, `map_t`, `array_t`, etc.
 *
 * The module enables:
 * - Type-safe casting via `as<T>()`
 * - Serialization / deserialization (to CBOR or JSON)
 * - Structured value operations
 */
#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include <dlpack/dlpack.h>
#include <nlohmann/json.hpp>

#include "exception.hpp"
#include "object.hpp"

namespace ailoy {

enum class encoding_method_t {
  cbor = 0,
  json = 1,
};

class bytes_t;

/**
 * @brief Abstract base class for JSON-compatible value types
 */
class value_t : public object_t {
public:
  value_t() = default;

  value_t(const value_t &) = default;

  value_t(value_t &&) = default;

  virtual ~value_t() = default;

  std::shared_ptr<bytes_t> encode(encoding_method_t method) const;

  virtual nlohmann::json to_nlohmann_json() const = 0;

  operator nlohmann::json() const { return to_nlohmann_json(); }

  virtual std::string get_type() const = 0;

  template <typename derived_t>
    requires std::is_base_of_v<object_t, derived_t>
  bool is_type_of() const {
    return typeid(derived_t).name() == get_type();
  }
};

/**
 * @brief Specializations for downcasting `value_t` to a concrete subclass
 * @tparam derived_t The desired target type
 * @throws ailoy::exception if type does not match
 */
template <> struct object_cast_t<const value_t> {
  template <typename derived_t>
    requires std::is_base_of_v<value_t, derived_t>
  static bool is_type_of(std::shared_ptr<const value_t> val) {
    return typeid(std::remove_cv_t<derived_t>).name() != val->get_type();
  }

  template <typename derived_t>
    requires std::is_base_of_v<value_t, derived_t>
  static std::shared_ptr<const derived_t>
  downcast(std::shared_ptr<const value_t> val) {
    if (typeid(std::remove_cv_t<derived_t>).name() != val->get_type())
      throw ailoy::exception(
          std::format("{} cannot be casted to {}.", val->get_type(),
                      typeid(std::remove_cv_t<derived_t>).name()));
    return std::static_pointer_cast<derived_t>(val);
  }
};

template <> struct object_cast_t<value_t> {
  template <typename derived_t>
    requires std::is_base_of_v<value_t, derived_t>
  static bool is_type_of(std::shared_ptr<value_t> val) {
    return typeid(derived_t).name() != val->get_type();
  }

  template <typename derived_t>
    requires std::is_base_of_v<value_t, derived_t>
  static std::shared_ptr<derived_t> downcast(std::shared_ptr<value_t> val) {
    if (typeid(derived_t).name() != val->get_type())
      throw ailoy::exception(std::format("{} cannot be casted to {}.",
                                         val->get_type(),
                                         typeid(derived_t).name()));
    return std::static_pointer_cast<derived_t>(val);
  }
};

/**
 * @brief Binary byte container
 * @details
 * - Inherits both from `value_t` and `std::vector<uint8_t>`
 * - Converts to and from string
 * - Encodes as binary when serialized
 */
class bytes_t : public value_t, public std::vector<uint8_t> {
public:
  template <typename... args_t>
  bytes_t(args_t... args) : std::vector<uint8_t>(args...) {}

  bytes_t(const std::string &v) : std::vector<uint8_t>(v.begin(), v.end()) {}

  bytes_t(const char *v) : bytes_t(std::string(v)) {}

  virtual ~bytes_t() = default;

  nlohmann::json to_nlohmann_json() const override {
    const std::vector<uint8_t> &v = *this;
    return nlohmann::json::binary(v);
  }

  std::string get_type() const noexcept override {
    return typeid(decltype(*this)).name();
  }

  operator std::string() const { return std::string{begin(), end()}; }

  friend std::ostream &operator<<(std::ostream &, const ailoy::bytes_t &);
};

std::ostream &operator<<(std::ostream &, const ailoy::bytes_t &);

template <typename t> class pod_value_t : public value_t {
public:
  pod_value_t(t v) : v_(v) {}

  nlohmann::json to_nlohmann_json() const override {
    return nlohmann::json(v_);
  }

  operator t() const { return v_; }

protected:
  t v_;
};

/**
 * @brief Represents a `null` value
 * @details
 * - Used to encode/decode `nullptr`-like semantics
 */
class null_t : public value_t {
public:
  null_t() = default;

  nlohmann::json to_nlohmann_json() const override { return nullptr; }

  std::string get_type() const noexcept override {
    return typeid(decltype(*this)).name();
  }
};

/**
 * @brief Represents a boolean value
 * @details
 * - Wraps a `bool`
 * - Supports assignment from primitive `bool`
 */
class bool_t : public pod_value_t<bool> {
public:
  bool_t(bool v) : pod_value_t<bool>(v) {}

  bool_t &operator=(const bool &rhs) {
    v_ = rhs;
    return *this;
  }

  std::string get_type() const noexcept override {
    return typeid(decltype(*this)).name();
  }
};

/**
 * @brief Represents an unsigned integer value
 * @details
 * - Wraps an `unsigned long long`
 * - Supports arithmetic operators
 * - Enables direct assignment from `unsigned long long`
 */
class uint_t : public pod_value_t<unsigned long long> {
public:
  uint_t(unsigned long long v) : pod_value_t<unsigned long long>(v) {}

  uint_t operator+(uint_t v) {
    return uint_t(v_ + v.operator unsigned long long());
  }

  uint_t operator-(uint_t v) {
    return uint_t(v_ - v.operator unsigned long long());
  }

  uint_t operator*(uint_t v) {
    return uint_t(v_ * v.operator unsigned long long());
  }

  uint_t operator/(uint_t v) {
    return uint_t(v_ / v.operator unsigned long long());
  }

  uint_t operator+(unsigned long long v) { return uint_t(v_ + v); }

  uint_t operator-(unsigned long long v) { return uint_t(v_ - v); }

  uint_t operator*(unsigned long long v) { return uint_t(v_ * v); }

  uint_t operator/(unsigned long long v) { return uint_t(v_ / v); }

  uint_t &operator++() {
    v_++;
    return *this;
  }

  uint_t operator++(int) { return uint_t(v_++); }

  uint_t &operator--() {
    v_--;
    return *this;
  }

  uint_t operator--(int) { return uint_t(v_--); }

  uint_t &operator=(unsigned long long rhs) {
    v_ = rhs;
    return *this;
  }

  std::string get_type() const noexcept override {
    return typeid(decltype(*this)).name();
  }
};

/**
 * @brief Represents a signed integer value
 * @details
 * - Wraps a `long long`
 * - Supports arithmetic operators
 * - Enables direct assignment from `long long`
 */
class int_t : public pod_value_t<long long> {
public:
  int_t(long long v) : pod_value_t<long long>(v) {}

  int_t operator+(int_t v) { return int_t(v_ + v.operator long long()); }

  int_t operator-(int_t v) { return int_t(v_ - v.operator long long()); }

  int_t operator*(int_t v) { return int_t(v_ * v.operator long long()); }

  int_t operator/(int_t v) { return int_t(v_ / v.operator long long()); }

  int_t operator+(long long v) { return int_t(v_ + v); }

  int_t operator-(long long v) { return int_t(v_ - v); }

  int_t operator*(long long v) { return int_t(v_ * v); }

  int_t operator/(long long v) { return int_t(v_ / v); }

  int_t &operator++() {
    v_++;
    return *this;
  }

  int_t operator++(int) { return int_t(v_++); }

  int_t &operator--() {
    v_--;
    return *this;
  }

  int_t operator--(int) { return int_t(v_--); }

  int_t &operator=(long long rhs) {
    v_ = rhs;
    return *this;
  }

  std::string get_type() const noexcept override {
    return typeid(decltype(*this)).name();
  }
};

/**
 * @brief Represents a 32-bit floating-point value
 * @details
 * - Wraps a `float`
 * - Supports basic arithmetic and assignment
 */
class float_t : public pod_value_t<float> {
public:
  float_t(float v) : pod_value_t<float>(v) {}

  float_t operator+(float_t v) { return float_t(v_ + v.operator float()); }

  float_t operator-(float_t v) { return float_t(v_ - v.operator float()); }

  float_t operator*(float_t v) { return float_t(v_ * v.operator float()); }

  float_t operator/(float_t v) { return float_t(v_ / v.operator float()); }

  float_t operator+(float v) { return float_t(v_ + v); }

  float_t operator-(float v) { return float_t(v_ - v); }

  float_t operator*(float v) { return float_t(v_ * v); }

  float_t operator/(float v) { return float_t(v_ / v); }

  float_t &operator=(float rhs) {
    v_ = rhs;
    return *this;
  }

  std::string get_type() const noexcept override {
    return typeid(decltype(*this)).name();
  }
};

/**
 * @brief Represents a 64-bit floating-point value
 * @details
 * - Wraps a `double`
 * - Supports basic arithmetic and assignment
 */
class double_t : public pod_value_t<double> {
public:
  double_t(double v) : pod_value_t<double>(v) {}

  double_t operator+(double_t v) { return double_t(v_ + v.operator double()); }

  double_t operator-(double_t v) { return double_t(v_ - v.operator double()); }

  double_t operator*(double_t v) { return double_t(v_ * v.operator double()); }

  double_t operator/(double_t v) { return double_t(v_ / v.operator double()); }

  double_t operator+(double v) { return double_t(v_ + v); }

  double_t operator-(double v) { return double_t(v_ - v); }

  double_t operator*(double v) { return double_t(v_ * v); }

  double_t operator/(double v) { return double_t(v_ / v); }

  double_t &operator=(double rhs) {
    v_ = rhs;
    return *this;
  }

  std::string get_type() const noexcept override {
    return typeid(decltype(*this)).name();
  }
};

/**
 * @brief Represents a string
 * @details
 * - Inherits from both `value_t` and `std::string`
 * - Supports concatenation with `std::string` and `string_t`
 */
class string_t : public value_t, public std::string {
public:
  template <typename... args_t>
  string_t(args_t... args) : std::string(args...), value_t() {}

  string_t operator+(std::string v) { return string_t(*this + v); }

  string_t operator+(string_t v) { return string_t(*this + std::string(v)); }

  nlohmann::json to_nlohmann_json() const override {
    return nlohmann::json(*this);
  }

  std::string get_type() const noexcept override {
    return typeid(decltype(*this)).name();
  }
};

/**
 * @brief Represents a JSON-compatible array of values
 * @details
 * - Inherits from `value_t` and `std::vector<std::shared_ptr<value_t>>`
 * - Provides type-safe access via `.at<t>(index)`
 */
class array_t : public value_t, public std::vector<std::shared_ptr<value_t>> {
public:
  template <typename... args_t>
  array_t(args_t... args)
      : value_t(), std::vector<std::shared_ptr<value_t>>(args...) {}

  nlohmann::json to_nlohmann_json() const override {
    nlohmann::json::array_t rv = nlohmann::json::array();
    for (const auto &elem : *this)
      rv.push_back(elem->to_nlohmann_json());
    return rv;
  }

  std::string get_type() const noexcept override {
    return typeid(decltype(*this)).name();
  }

  template <typename t = value_t>
    requires std::is_base_of_v<value_t, t>
  std::shared_ptr<const t> at(size_t idx) const {
    auto array = as<array_t>();
    if (idx >= array->size())
      throw ailoy::exception(std::format("out of range: {}", idx));
    return array->std::vector<std::shared_ptr<value_t>>::at(idx)->as<t>();
  }

  template <typename t = value_t>
    requires std::is_base_of_v<value_t, t>
  std::shared_ptr<t> at(size_t idx) {
    auto array = as<array_t>();
    if (idx >= array->size())
      throw ailoy::exception(std::format("out of range: {}", idx));
    return array->std::vector<std::shared_ptr<value_t>>::at(idx)->as<t>();
  }
};

/**
 * @brief Represents a JSON-compatible key-value object
 * @details
 * - Inherits from `value_t` and `std::unordered_map<std::string,
 * std::shared_ptr<value_t>>`
 * - Supports safe key access with optional default value
 * - Enables type-safe casting via `.at<t>(key)`
 */
class map_t : public value_t,
              public std::unordered_map<std::string, std::shared_ptr<value_t>> {
public:
  template <typename... args_t>
  map_t(args_t... args)
      : value_t(),
        std::unordered_map<std::string, std::shared_ptr<value_t>>(args...) {}

  nlohmann::json to_nlohmann_json() const override {
    nlohmann::json::object_t rv = nlohmann::json::object();
    for (const auto [key, elem] : *this)
      rv.insert_or_assign(key, elem->to_nlohmann_json());
    return rv;
  }

  std::string get_type() const noexcept override {
    return typeid(decltype(*this)).name();
  }

  template <typename t = value_t>
    requires std::is_base_of_v<value_t, t>
  std::shared_ptr<const t> at(const std::string &key) const {
    auto map = as<map_t>();
    if (!map->contains(key))
      throw ailoy::exception(std::format("key not exists: {}", key));
    return map
        ->std::unordered_map<std::string, std::shared_ptr<value_t>>::at(key)
        ->as<t>();
  }

  template <typename t = value_t>
    requires std::is_base_of_v<value_t, t>
  std::shared_ptr<const t> at(std::string key, t default_value) const {
    auto map = as<map_t>();
    if (!map->contains(key))
      return std::make_shared<t>(default_value);
    return map
        ->std::unordered_map<std::string, std::shared_ptr<value_t>>::at(key)
        ->as<t>();
  }

  template <typename t = value_t>
    requires std::is_base_of_v<value_t, t>
  std::shared_ptr<t> at(std::string key) {
    auto map = as<map_t>();
    if (!map->contains(key))
      throw ailoy::exception(std::format("key not exists: {}", key));
    return map
        ->std::unordered_map<std::string, std::shared_ptr<value_t>>::at(key)
        ->as<t>();
  }

  template <typename t = value_t>
    requires std::is_base_of_v<value_t, t>
  std::shared_ptr<t> at(std::string key, t default_value) {
    auto map = as<map_t>();
    if (!map->contains(key))
      return std::make_shared<t>(default_value);
    return map
        ->std::unordered_map<std::string, std::shared_ptr<value_t>>::at(key)
        ->as<t>();
  }
};

/**
 * @brief Represents a multi-dimensional tensor (NumPy-like array)
 * @details
 * - Stores shape, DLDataType, and raw data buffer
 * - Provides serialization to/from binary JSON
 * - Allows reinterpretation as `std::vector<T>`
 */
class ndarray_t : public value_t {
public:
  ndarray_t() = default;

  ndarray_t(std::vector<size_t> shape, DLDataType dtype, const uint8_t *data,
            size_t size)
      : dtype(dtype), data(data, data + size) {
    this->shape.clear();
    for (size_t dim : shape)
      this->shape.push_back(dim);
  }

  nlohmann::json to_nlohmann_json() const override;

  static std::shared_ptr<ndarray_t>
  from_nlohmann_json(const nlohmann::json::binary_t &j);

  std::string get_type() const noexcept override {
    return typeid(decltype(*this)).name();
  }

  const size_t itemsize() const { return (dtype.bits * dtype.lanes + 7) / 8; }

  const size_t size() const {
    size_t nbytes =
        std::reduce(shape.begin(), shape.end(), 1, std::multiplies<int>());
    nbytes *= itemsize();
    return nbytes;
  }

  template <typename t> operator std::vector<t>() const {
    const t *casted_data = reinterpret_cast<const t *>(data.data());
    size_t len = size() / sizeof(t);
    return std::vector<t>(casted_data, casted_data + len);
  }

  std::string shape_str() const;

  std::vector<size_t> shape;
  DLDataType dtype;
  bytes_t data;
};

std::shared_ptr<value_t> from_nlohmann_json(const nlohmann::json &j);

std::shared_ptr<value_t> decode(std::shared_ptr<bytes_t> bytes,
                                encoding_method_t method);

std::shared_ptr<value_t> decode(const std::string &bytes,
                                encoding_method_t method);

} // namespace ailoy
