#pragma once

#include <memory>
#include <type_traits>

namespace ailoy {

class object_t;

template <typename base_t>
  requires std::is_base_of_v<object_t, std::remove_cv_t<base_t>>
struct object_cast_t {
  template <typename derived_t>
    requires std::is_base_of_v<std::remove_cv_t<base_t>,
                               std::remove_cv_t<derived_t>>
  static std::shared_ptr<derived_t> downcast(std::shared_ptr<base_t> obj) {
    // @jhlee: TODO use dynamic_pointer_cast for developing,
    // static_pointer_cast for production mode
    // uses RTTI for checking downcast safety
    return std::dynamic_pointer_cast<derived_t>(obj);
  }
};

class object_t : public std::enable_shared_from_this<object_t> {
public:
  object_t() = default;

  object_t(const object_t &) = default;

  object_t(object_t &&) = default;

  virtual ~object_t() = default;

  template <typename derived_t>
    requires std::is_base_of_v<object_t, derived_t>
  std::shared_ptr<const derived_t> as() const {
    using base_t = std::decay_t<decltype(*this)>;
    return object_cast_t<const base_t>::downcast<const derived_t>(
        shared_from_this());
  }

  template <typename derived_t>
    requires std::is_base_of_v<object_t, derived_t>
  std::shared_ptr<derived_t> as() {
    using base_t = std::decay_t<decltype(*this)>;
    return object_cast_t<base_t>::downcast<derived_t>(shared_from_this());
  }
};

template <typename t, typename... args_t>
std::shared_ptr<t> create(args_t... args) {
  return std::make_shared<t>(std::forward<args_t>(args)...);
}

template <typename t, typename... args_t>
concept is_object = requires(args_t... args) {
  // `ailoy::create` function can create an instance
  { ailoy::create<t>(args...) } -> std::same_as<std::shared_ptr<t>>;
} && std::is_base_of_v<object_t, t>;

template <typename t, typename setter_t = t::setter_t>
std::unique_ptr<setter_t> create_setter(std::shared_ptr<t> value) {
  return std::make_unique<setter_t>(value);
}

using url_t = std::string;

template <typename T> struct is_shared_ptr : std::false_type {};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;

template <typename T> inline constexpr bool false_type_v = false;

} // namespace ailoy
