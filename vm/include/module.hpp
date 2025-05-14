/**
 * @file module.hpp
 * @brief Element that provides functionalities to VM by being loaded into it.
 * @details
 *
 * Module defines some `operator_t`s and `component_t`s for functionalities.
 * Modules can be loaded into VM in form of operators and component factories.
 *
 * ## `operator_t`
 *
 * `operator_t` is a single function that can be run in VM.
 * `operator_t` has `initialize()` and `step()` that calls the function.
 * For a single call of operator, `initialize()` is called first with
 * parameters, then `step()` is called several times to proceed the function.
 *
 * `module_` contains `operator_t`s to make VMs can call them.
 *
 * ### `method_operator_t`
 *
 * `method_operator_t` is an `operator_t` that is enclosed `component_t`.
 * `method_operator_t` has `bind()` to bind itself to a `component_t` instance.
 *
 * ### instant/iterative
 *
 * The result of an operator call is usually obtained instantly,
 * but can be obtained through multiple steps, such as language model infer.
 * The formers are called 'instant', while the latters are called `iterative`.
 *
 * ## `component_t`
 *
 * `component_t` is an `object_t` that has operators and objects as members.
 * operators are the `method_operator_` instances bound to this `component_t`.
 * objects are the `object_t` instances included in this  `component_t`.
 * the operators and the objects interact with each other to perform a
 * funtionality in a `component_t` instance.
 *
 * In short, `component_t` is similar to classes in programming languages.
 *
 * ### `component_factory_t`
 *
 * A function that creates `component_t` instances.
 * `module_` contains `component_factory_t`s to make VMs can create components
 *
 * ```cpp
 * {
 *   auto create_my_component = get_my_module()->factories.at("my_component");
 *
 *   auto attrs = pakky::create<pakky::map_t>();
 *   auto my_component = std::get<0>(create_my_component(attrs));
 *   auto my_op = my_component->get_operator("my_operator");
 *
 *   auto inputs = pakky::create<pakky::map_t>();
 *   my_op->initialize(inputs);
 *   auto out = std::get<0>(my_op->step()).val->as<pakky::map_t>();
 * }
 * ```
 */
#pragma once

#include <functional>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>

#include "value.hpp"

namespace ailoy {

class component_t;

/**
 * @brief Operator output when operator success
 */
struct ok_output_t {
  ok_output_t() : val(nullptr), finish(true) {}

  ok_output_t(std::shared_ptr<value_t> v) : val(v), finish(true) {}

  ok_output_t(std::shared_ptr<value_t> v, bool fin) : val(v), finish(fin) {}

  bool finish;

  std::shared_ptr<value_t> val;
};

/**
 * @brief Operator output when operator fails
 */
struct error_output_t {
  error_output_t(const std::string &reason) : reason(reason) {}

  std::string reason;
};

using output_t = std::variant<ok_output_t, error_output_t>;

using value_or_error_t = std::variant<std::shared_ptr<value_t>, error_output_t>;

using component_or_error_t =
    std::variant<std::shared_ptr<component_t>, error_output_t>;

/**
 * @brief Abstract base class for VM-running function object
 */
class operator_t : public object_t {
public:
  operator_t() : in_(nullptr) {}

  virtual std::optional<error_output_t>
  initialize(std::shared_ptr<const value_t> in = nullptr) {
    in_ = in;
    return std::nullopt;
  }

  virtual void reset_input() {
    if (in_ != nullptr)
      in_.reset();
  }

  std::shared_ptr<const value_t> get_input() const noexcept { return in_; }

  virtual output_t step() = 0;

private:
  std::shared_ptr<const value_t> in_;
};

/**
 * @brief VM-running function object with instant output
 */
class instant_operator_t : public operator_t {
public:
  instant_operator_t(
      std::function<value_or_error_t(std::shared_ptr<const value_t>)> f)
      : operator_t(), f_(f) {}

  output_t step() override {
    auto output = f_(get_input());
    reset_input();
    if (output.index() == 0)
      return ok_output_t(std::get<0>(output));
    else
      return std::get<1>(output);
  }

private:
  std::function<value_or_error_t(std::shared_ptr<const value_t>)> f_;
};

/**
 * @brief VM-running function object with iterative output
 */
class iterative_operator_t : public operator_t {
public:
  iterative_operator_t(
      std::function<value_or_error_t(std::shared_ptr<const value_t>)> finit,
      std::function<output_t(std::shared_ptr<value_t>)> fstep)
      : operator_t(), finit_(finit), fstep_(fstep) {}

  std::optional<error_output_t>
  initialize(std::shared_ptr<const value_t> in) override {
    auto parent_rv = operator_t::initialize(in);
    if (parent_rv.has_value())
      return parent_rv;
    auto state_or_error = finit_(in);
    if (state_or_error.index() == 0) {
      state_ = std::get<0>(state_or_error);
      return std::nullopt;
    } else {
      return std::get<1>(state_or_error);
    }
  }

  output_t step() override {
    auto output = fstep_(state_);
    ailoy::ok_output_t *ok_output = std::get_if<ok_output_t>(&output);
    if (ok_output)
      if (ok_output->finish)
        reset_input();
    return output;
  }

  void reset_input() override {
    operator_t::reset_input();
    state_ = nullptr;
  }

private:
  std::function<value_or_error_t(std::shared_ptr<const value_t>)> finit_;
  std::function<output_t(std::shared_ptr<value_t>)> fstep_;
  std::shared_ptr<value_t> state_;
};

/**
 * @brief Abstract base class for VM-running component method object
 */
class method_operator_t : public operator_t {
public:
  method_operator_t() : operator_t() {}

  virtual std::optional<error_output_t>
  initialize(std::shared_ptr<const value_t> in) override {
    return operator_t::initialize(in);
  }

  void bind(std::shared_ptr<component_t> comp) { comp_ = comp; }

protected:
  std::weak_ptr<component_t> comp_;
};

/**
 * @brief VM-running component method object with instant output
 */
class instant_method_operator_t : public method_operator_t {
public:
  instant_method_operator_t(
      std::function<value_or_error_t(std::shared_ptr<component_t>,
                                     std::shared_ptr<const value_t>)>
          f)
      : method_operator_t(), f_(f) {}

  output_t step() override {
    auto output = f_(comp_.lock(), get_input());
    reset_input();
    if (output.index() == 0)
      return ok_output_t(std::get<0>(output));
    else
      return std::get<1>(output);
  }

private:
  std::function<value_or_error_t(std::shared_ptr<component_t>,
                                 std::shared_ptr<const value_t>)>
      f_;
};

/**
 * @brief VM-running component method object with iterative output
 */
class iterative_method_operator_t : public method_operator_t {
public:
  iterative_method_operator_t(
      std::function<value_or_error_t(std::shared_ptr<component_t>,
                                     std::shared_ptr<const value_t>)>
          finit,
      std::function<output_t(std::shared_ptr<component_t>,
                             std::shared_ptr<value_t>)>
          fstep)
      : method_operator_t(), finit_(finit), fstep_(fstep) {}

  std::optional<error_output_t>
  initialize(std::shared_ptr<const value_t> in) override {
    auto parent_rv = method_operator_t::initialize(in);
    if (parent_rv.has_value())
      return parent_rv;
    auto state_or_error = finit_(comp_.lock(), in);
    if (state_or_error.index() == 0) {
      state_ = std::get<0>(state_or_error);
      return std::nullopt;
    } else {
      return std::get<1>(state_or_error);
    }
  }

  output_t step() override {
    auto output = fstep_(comp_.lock(), state_);
    ailoy::ok_output_t *ok_output = std::get_if<ok_output_t>(&output);
    if (ok_output)
      if (ok_output->finish)
        reset_input();
    return output;
  }

  void reset_input() override {
    operator_t::reset_input();
    state_ = nullptr;
  }

private:
  std::function<value_or_error_t(std::shared_ptr<component_t>,
                                 std::shared_ptr<const value_t>)>
      finit_;
  std::function<output_t(std::shared_ptr<component_t>,
                         std::shared_ptr<value_t>)>
      fstep_;
  std::shared_ptr<value_t> state_;
};

/**
 * @brief Component consisting a functionality of modules
 */
class component_t : public object_t {
public:
  component_t() {}

  template <std::ranges::range range_type>
  component_t(range_type ops) : ops_(ops) {}

  const std::unordered_map<std::string, std::shared_ptr<method_operator_t>> &
  get_operators() const {
    return ops_;
  }

  std::shared_ptr<operator_t> get_operator(const std::string &name) {
    if (!ops_.contains(name)) {
      return nullptr;
    }

    auto op = ops_.at(name);
    op->bind(this->as<component_t>());
    return op;
  }

  template <typename t = object_t>
    requires std::is_base_of_v<object_t, t>
  std::shared_ptr<t> get_obj(const std::string &name) const {
    if (!objs_.contains(name)) {
      return nullptr;
    }
    return objs_.at(name)->as<t>();
  }

  template <typename t = object_t>
    requires std::is_base_of_v<object_t, t>
  std::shared_ptr<t> get_obj(const std::string &name) {
    if (!objs_.contains(name)) {
      return nullptr;
    }
    return objs_.at(name)->as<t>();
  }

  void set_obj(const std::string &name, std::shared_ptr<object_t> obj) {
    objs_.insert_or_assign(name, obj);
  }

private:
  std::unordered_map<std::string, std::shared_ptr<method_operator_t>> ops_;
  std::unordered_map<std::string, std::shared_ptr<object_t>> objs_;
};

/**
 * @brief Function that creates component instance
 */
using component_factory_t =
    std::function<component_or_error_t(std::shared_ptr<const value_t>)>;

/**
 * @brief Module that provides functionalities to VM.
 */
struct module_t : public object_t {
  std::unordered_map<std::string, std::shared_ptr<operator_t>> ops;

  std::unordered_map<std::string, component_factory_t> factories;
};

/**
 * @brief Get default module with basic operators & components
 */
std::shared_ptr<const module_t> get_default_module();

} // namespace ailoy
