#include "module.hpp"

namespace ailoy {

static std::shared_ptr<module_t> debug_module = ailoy::create<module_t>();

std::shared_ptr<const module_t> get_debug_module() {
  auto &debug_ops = debug_module->ops;
  auto &debug_factories = debug_module->factories;

  // Add Operator Echo
  if (debug_ops.find("echo") == debug_ops.end()) {
    auto f = [](std::shared_ptr<const value_t> inputs) -> value_or_error_t {
      if (!inputs->is_type_of<map_t>())
        return error_output_t(
            type_error("echo", "inputs", "map_t", inputs->get_type()));

      auto inputs_map = inputs->as<map_t>();
      if (!inputs_map->contains("text"))
        return error_output_t(range_error("echo", "text"));
      auto text = inputs_map->at("text");
      if (!text->is_type_of<string_t>())
        return error_output_t(
            type_error("echo", "text", "string_t", text->get_type()));

      auto outputs = create<map_t>();
      outputs->insert_or_assign("text",
                                create<string_t>(*text->as<string_t>()));
      return outputs;
    };
    debug_ops.insert_or_assign("echo", create<instant_operator_t>(f));
  }

  // Add Operator Spell
  if (debug_ops.find("spell") == debug_ops.end()) {
    auto finit = [](std::shared_ptr<const value_t> inputs) -> value_or_error_t {
      if (!inputs->is_type_of<map_t>())
        return error_output_t(
            type_error("spell", "inputs", "map_t", inputs->get_type()));

      auto inputs_map = inputs->as<map_t>();
      if (!inputs_map->contains("text"))
        return error_output_t(range_error("spell", "text"));
      auto text = inputs_map->at("text");
      if (!text->is_type_of<string_t>())
        return error_output_t(
            type_error("spell", "text", "string_t", text->get_type()));

      auto initial_state = create<map_t>();
      initial_state->insert_or_assign(
          "message", ailoy::create<string_t>(*text->as<string_t>()));
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

      auto outputs = create<map_t>();
      outputs->insert_or_assign("text", next_char);

      return ok_output_t(outputs, *idx == msg->size());
    };
    debug_ops.insert_or_assign("spell",
                               create<iterative_operator_t>(finit, fstep));
  }

  // Create accumulate
  if (debug_factories.find("accumulator") == debug_factories.end()) {
    component_factory_t fcreate =
        [](std::shared_ptr<const value_t> attrs) -> component_or_error_t {
      if (!attrs->is_type_of<map_t>())
        return error_output_t(
            type_error("accumulator", "attrs", "map_t", attrs->get_type()));

      auto attrs_map = attrs->as<map_t>();
      if (!attrs_map->contains("base"))
        return error_output_t(range_error("accumulator", "base"));
      auto base = attrs_map->at("base");
      if (!base->is_type_of<string_t>())
        return error_output_t(
            type_error("accumulator", "base", "string_t", base->get_type()));

      auto put = [](std::shared_ptr<component_t> component,
                    std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        if (!inputs->is_type_of<map_t>())
          return error_output_t(type_error("accumulator.put", "inputs", "map_t",
                                           inputs->get_type()));

        auto inputs_map = inputs->as<map_t>();
        if (!inputs_map->contains("s"))
          return error_output_t(range_error("accumulator.put", "s"));
        auto s = inputs_map->at("s");
        if (!s->is_type_of<string_t>())
          return error_output_t(
              type_error("accumulator.put", "s", "string_t", s->get_type()));

        auto outputs = create<ailoy::map_t>();
        auto base = component->get_obj<string_t>("base");
        auto count = component->get_obj<uint_t>("count");
        *base += *s->as<string_t>();
        ++(*count);
        return outputs;
      };
      auto get = [](std::shared_ptr<component_t> component,
                    std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        auto text =
            create<ailoy::string_t>(*component->get_obj<string_t>("base"));
        auto outputs = create<ailoy::map_t>();
        outputs->insert_or_assign("text", text);
        return outputs;
      };
      auto count =
          [](std::shared_ptr<component_t> component,
             std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        auto count =
            create<ailoy::uint_t>(*component->get_obj<uint_t>("count"));
        auto outputs = create<ailoy::map_t>();
        outputs->insert_or_assign("count", count);
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
      rv->set_obj("base", create<string_t>(*base->as<string_t>()));
      return rv;
    };
    debug_factories.insert_or_assign("accumulator", fcreate);
  }

  return debug_module;
}

} // namespace ailoy
