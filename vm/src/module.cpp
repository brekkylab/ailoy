#include "module.hpp"

#include "calculator.hpp"
#include "http_request.hpp"

namespace ailoy {

static std::shared_ptr<module_t> default_module = ailoy::create<module_t>();

std::shared_ptr<const module_t> get_default_module() {
  auto &default_ops = default_module->ops;
  auto &default_factories = default_module->factories;

  // Add Operator Echo
  if (default_ops.find("echo") == default_ops.end()) {
    auto f = [](std::shared_ptr<const value_t> inputs) -> value_or_error_t {
      if (!inputs->is_type_of<string_t>())
        return error_output_t(
            type_error("Echo", "inputs", "string_t", inputs->get_type()));
      auto out = create<string_t>(*inputs->as<string_t>());
      return out;
    };
    default_ops.insert_or_assign("echo", create<instant_operator_t>(f));
  }

  // Add Operator Spell
  if (default_ops.find("spell") == default_ops.end()) {
    auto finit = [](std::shared_ptr<const value_t> inputs) -> value_or_error_t {
      if (!inputs->is_type_of<string_t>())
        return error_output_t(
            type_error("Spell", "inputs", "string_t", inputs->get_type()));
      auto initial_state = create<map_t>();
      initial_state->insert_or_assign(
          "message", ailoy::create<string_t>(*inputs->as<string_t>()));
      initial_state->insert_or_assign("index", create<uint_t>(0));
      auto rv = std::static_pointer_cast<value_t>(initial_state);
      return rv;
    };
    auto fstep = [](std::shared_ptr<value_t> state) -> output_t {
      auto msg = state->as<map_t>()->at<string_t>("message");
      auto idx = state->as<map_t>()->at<uint_t>("index");
      if (*idx > msg->size())
        return error_output_t(range_error("Spell", "Index overflow"));

      auto next_char = create<string_t>(msg->substr((*idx)++, 1));
      return ok_output_t(next_char, *idx == msg->size());
    };
    default_ops.insert_or_assign("spell",
                                 create<iterative_operator_t>(finit, fstep));
  }

  // Create accumulate
  if (default_factories.find("accumulator") == default_factories.end()) {
    component_factory_t fcreate = [](std::shared_ptr<const value_t> inputs)
        -> std::shared_ptr<component_t> {
      auto put = [](std::shared_ptr<component_t> component,
                    std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        auto outputs = create<ailoy::map_t>();
        auto s = inputs->as<string_t>();
        auto base = component->get_obj<string_t>("base");
        auto count = component->get_obj<uint_t>("count");
        *base += *s;
        ++(*count);
        return outputs;
      };
      auto get = [](std::shared_ptr<component_t> component,
                    std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        auto outputs =
            create<ailoy::string_t>(*component->get_obj<string_t>("base"));
        return outputs;
      };
      auto count =
          [](std::shared_ptr<component_t> component,
             std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        auto outputs =
            create<ailoy::uint_t>(*component->get_obj<uint_t>("count"));
        return outputs;
      };
      auto rv = create<component_t>(
          std::initializer_list<
              std::pair<const std::string, std::shared_ptr<method_operator_t>>>{
              {"put", create<instant_method_operator_t>(put)},
              {"get", create<instant_method_operator_t>(get)},
              {"count", create<instant_method_operator_t>(count)},
          });
      rv->set_obj("count", create<uint_t>(0));
      rv->set_obj("base", create<string_t>(*inputs->as<string_t>()));
      return rv;
    };
    default_factories.insert_or_assign("accumulator", fcreate);
  }

  // Add Operator HTTP Request
  if (default_ops.find("http_request") == default_ops.end()) {
    default_ops.insert_or_assign("http_request",
                                 create<instant_operator_t>(http_request_op));
  }

  // Add Operator Calculator
  if (!default_ops.contains("calculator")) {
    default_ops.insert_or_assign("calculator",
                                 create<instant_operator_t>(calculator_op));
  }

  return default_module;
}

} // namespace ailoy
