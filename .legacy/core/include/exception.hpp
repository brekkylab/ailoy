#pragma once

#include <exception>
#include <format>
#include <string>

namespace ailoy {

std::string build_errstr(const char *what);

template <typename T>
concept is_exception_reason = requires(T t) {
  requires noexcept(t.what());
  { t.what() } -> std::same_as<const char *>;
} || requires() {
  requires noexcept(T::what());
  { T::what() } -> std::same_as<const char *>;
};

struct runtime_error {
  runtime_error() : errstr_("Internal error") {}

  runtime_error(const std::string &what) : errstr_(what) {}

  const char *what() const noexcept { return errstr_.c_str(); }

  operator std::string() const { return errstr_; }

  std::string errstr_;
};

static_assert(is_exception_reason<runtime_error>);

struct type_error {
  type_error() : errstr_("Type error") {}

  type_error(const std::string &what) : errstr_("Type error: " + what) {}

  type_error(const std::string &context, const std::string &name)
      : errstr_(std::format("[{}] Type error: {}", context, name)) {}

  type_error(const std::string &context, const std::string &name,
             const std::string &expected, const std::string &actual)
      : errstr_(std::format(
            "[{}] Type error:\n - name: {}\n - expected: {}\n - actual: {}",
            context, name, expected, actual)) {}

  const char *what() const noexcept { return errstr_.c_str(); }

  operator std::string() const { return errstr_; }

  std::string errstr_;
};

static_assert(is_exception_reason<type_error>);

struct range_error {
  range_error() : range_error("Range error") {}

  range_error(const std::string &what) : errstr_(what) {}

  range_error(const std::string &context, const std::string &name)
      : errstr_(std::format("[{}] Range error: {}", context, name)) {}

  const char *what() const noexcept { return errstr_.c_str(); }

  operator std::string() const { return errstr_; }

  std::string errstr_;
};

static_assert(is_exception_reason<range_error>);

struct value_error {
  value_error() : value_error("Value error") {}

  value_error(const std::string &what) : errstr_(what) {}

  value_error(const std::string &context, const std::string &name)
      : errstr_(std::format("[{}] Value error: {}", context, name)) {}

  value_error(const std::string &context, const std::string &name,
              const std::string &expected, const std::string &actual)
      : errstr_(std::format(
            "[{}] Value error:\n - name: {}\n - expected: {}\n - actual: {}",
            context, name, expected, actual)) {}

  const char *what() const noexcept { return errstr_.c_str(); }

  operator std::string() const { return errstr_; }

  std::string errstr_;
};

static_assert(is_exception_reason<value_error>);

struct not_implemented {
  not_implemented();

  not_implemented(const std::string &what);

  const char *what() const noexcept;

  operator std::string() const { return errstr_; }

  std::string errstr_;
};

static_assert(is_exception_reason<not_implemented>);

template <typename T = runtime_error>
  requires is_exception_reason<T>
class exception_t : public std::exception {
public:
  template <typename... TArgs> exception_t(TArgs... args) : std::exception() {
    if constexpr (requires { T::what(); }) {
      errstr = build_errstr(T::what());
    } else {
      errstr = build_errstr(T(std::forward<TArgs>(args)...).what());
    }
  }

  const char *what() const noexcept override { return errstr.c_str(); }

private:
  std::string errstr;
};

template <typename T = runtime_error, typename... TArgs>
exception_t<T> exception(TArgs... args) {
  return exception_t<T>(std::forward<TArgs>(args)...);
}

} // namespace ailoy
