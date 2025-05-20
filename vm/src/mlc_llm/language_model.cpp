#include "language_model.hpp"

#include "mlc_llm_engine.hpp"
#include "module.hpp"
#include "uuid.hpp"

namespace ailoy {

std::optional<error_output_t>
_validate_language_model_messages(const std::string &context,
                                  std::shared_ptr<const map_t> input_map) {
  if (!input_map->contains("messages"))
    return error_output_t(range_error(context, "messages"));
  if (!input_map->at("messages")->is_type_of<array_t>())
    return error_output_t(type_error(context, "messages", "array_t",
                                     input_map->at("messages")->get_type()));
  auto messages = input_map->at("messages")->as<array_t>();
  for (auto msg_val : *messages) {
    if (!msg_val->is_type_of<map_t>())
      return error_output_t(
          type_error(context, "messages.*", "map_t", msg_val->get_type()));
    auto msg = msg_val->as<map_t>();
    if (!msg->contains("role"))
      return error_output_t(range_error(context, "role"));
    if (!msg->at("role")->is_type_of<string_t>())
      return error_output_t(
          type_error(context, "role", "string_t", msg->at("role")->get_type()));
    if (*msg->at<string_t>("role") != "system" &&
        *msg->at<string_t>("role") != "user" &&
        *msg->at<string_t>("role") != "assistant" &&
        *msg->at<string_t>("role") != "tool")
      return error_output_t(value_error(context, "role",
                                        "system | user | assistant | tool",
                                        *msg->at<string_t>("role")));
    if (!(msg->contains("content") || msg->contains("tool_calls")))
      return error_output_t("Invalid msg schema");
    if (msg->contains("content") &&
        !(msg->at("content")->is_type_of<string_t>() ||
          msg->at("content")->is_type_of<null_t>()))
      return error_output_t("Invalid msg schema");
    if (msg->contains("tool_calls") &&
        !msg->at("tool_calls")->is_type_of<array_t>())
      return error_output_t(type_error(context, "tool_calls", "string_t",
                                       msg->at("content")->get_type()));
  }
  return std::nullopt;
}

component_or_error_t
create_tvm_language_model_component(std::shared_ptr<const value_t> inputs) {
  if (!inputs->is_type_of<map_t>())
    return error_output_t(type_error("TVM Language Model: create", "inputs",
                                     "map_t", inputs->get_type()));

  auto input_map = inputs->as<map_t>();

  // Parse model name
  if (!input_map->contains("model"))
    return error_output_t(range_error("TVM Language Model: create", "model"));
  if (!input_map->at("model")->is_type_of<string_t>())
    return error_output_t(type_error("TVM Language Model: create", "model",
                                     "string_t",
                                     input_map->at("model")->get_type()));
  std::string model = *input_map->at<string_t>("model");

  // Parse quantization(optional)
  std::string quantization;
  if (input_map->contains("quantization")) {
    if (input_map->at("quantization")->is_type_of<string_t>())
      quantization = *input_map->at<string_t>("quantization");
    else
      return error_output_t(
          type_error("TVM Language Model: create", "quantization", "string_t",
                     input_map->at("quantization")->get_type()));
  } else
    quantization = "q4f16_1";

  // Parse mode(optional)
  std::string mode;
  if (input_map->contains("mode")) {
    if (input_map->at("mode")->is_type_of<string_t>()) {
      mode = *input_map->at<string_t>("mode");
      if (mode != "local" && mode != "interactive" && mode != "server")
        return error_output_t(value_error("TVM Language Model: create", "mode",
                                          "local | interactive | server",
                                          mode));
    } else {
      return error_output_t(type_error("TVM Language Model: create", "mode",
                                       "string_t",
                                       input_map->at("mode")->get_type()));
    }
  } else
    mode = "interactive";

// Parse device
// @jhlee: TODO implement other environment
#if defined(USE_METAL)
  auto device = DLDevice{kDLMetal, 0};
#elif defined(USE_VULKAN)
  auto device = DLDevice{kDLVulkan, 0};
#else
  auto device = DLDevice{kDLCPU, 0};
#endif

  // Get engine
  std::shared_ptr<mlc_llm_engine_t> engine;
  try {
    engine = get_mlc_llm_engine(model, quantization, device, mode);
  } catch (const ailoy::runtime_error e) {
    return error_output_t(e.what());
  }

  if (engine->get_last_error().has_value())
    return error_output_t(engine->get_last_error().value());

  // Get template engine
  auto template_engine = chat_template_engine_t::make_from_config_file(
      engine->get_model_path() / "chat-template-config.json");

  engine->set_chat_template_engine(template_engine);

  // Define inference op
  auto infer = ailoy::create<iterative_method_operator_t>(
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        if (!inputs->is_type_of<map_t>())
          return error_output_t(type_error("TVM Language Model: infer",
                                           "inputs", "map_t",
                                           inputs->get_type()));

        auto input_map = inputs->as<map_t>();

        // Get input messages
        auto error = _validate_language_model_messages(
            "TVM Language Model: infer", input_map);
        if (error.has_value())
          return error.value();
        auto messages = input_map->at<array_t>("messages")->to_nlohmann_json();

        // Get tools (optional)
        nlohmann::json tools;
        if (input_map->contains("tools")) {
          if (input_map->at("tools")->is_type_of<string_t>()) {
            const std::string tools_str = *input_map->at<string_t>("tools");
            if (!nlohmann::json::accept(tools_str))
              return error_output_t(
                  value_error("[TVM Language Model: infer] Invalid JSON "
                              "string in tools: " +
                              tools_str));
            tools = nlohmann::json::parse(tools_str);
          } else if (input_map->at("tools")->is_type_of<array_t>()) {
            tools = input_map->at<array_t>("tools")->operator nlohmann::json();
          } else {
            return error_output_t(type_error(
                "TVM Language Model: infer", "tools", "string_t | array_t",
                input_map->at("tools")->get_type()));
          }
        }

        // Get reasoning (optional)
        bool enable_reasoning = false;
        if (input_map->contains("enable_reasoning") &&
            input_map->at("enable_reasoning")->is_type_of<bool_t>())
          enable_reasoning = *input_map->at<bool_t>("enable_reasoning");

        // Get ignore_reasoning (optional)
        bool ignore_reasoning_messages = false;
        if (input_map->contains("ignore_reasoning_messages") &&
            input_map->at("ignore_reasoning_messages")->is_type_of<bool_t>())
          ignore_reasoning_messages =
              *input_map->at<bool_t>("ignore_reasoning_messages");
        component->set_obj("ignore_reasoning_messages",
                           create<ailoy::bool_t>(ignore_reasoning_messages));

        // Apply chat template on messages
        auto prompt =
            component->get_obj("template_engine")
                ->as<chat_template_engine_t>()
                ->apply_chat_template(messages, tools, enable_reasoning);

        // Generate request ID
        uuid_t request_id = generate_uuid();

        // Insert request
        component->get_obj("engine")
            ->as<mlc_llm_engine_t>()
            ->initialize_chat_completion(request_id, prompt);

        return create<string_t>(request_id);
      },
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<value_t> state) -> output_t {
        // Get saved values & objects
        auto request_id = state->as<string_t>();
        auto engine = component->get_obj("engine")->as<mlc_llm_engine_t>();
        const bool ignore_reasoning_messages =
            *component->get_obj("ignore_reasoning_messages")
                 ->as<ailoy::bool_t>();

        // check if delta needs to be dismissed
        auto dismiss_delta =
            [ignore_reasoning_messages](nlohmann::json delta) -> bool {
          // case 1) dismiss if ignore option is set and output is reasoning
          bool dismiss_reasoning =
              (ignore_reasoning_messages && delta.value("reasoning", false));
          // case 2) wait during content is empty and no tool calls exist
          bool wait_valid_tool_call_output =
              ((delta.value("content", "") == "") &&
               !delta.contains("tool_calls"));
          return dismiss_reasoning || wait_valid_tool_call_output;
        };

        // Repeat steps until valid output comes or it finished.
        mlc_llm_engine_t::response_state_t response;
        while (true) {
          auto step_output = engine->step_chat_completion(*request_id);
          if (engine->get_last_error().has_value())
            return error_output_t(engine->get_last_error().value());

          // If prefill is not finished, `step_output` can be empty.
          // Step again until prefill is finished.
          if (!step_output.has_value())
            continue;

          response = step_output.value();
          // break if finish_reason is defined
          if (response.finish_reason.has_value())
            break;
          // break if delta is not dismissed
          if (!dismiss_delta(response.message))
            break;
        }

        auto out = create<map_t>();
        out->insert_or_assign(
            "message", from_nlohmann_json(response.message)->as<map_t>());
        if (response.finish_reason.has_value())
          out->insert_or_assign(
              "finish_reason",
              create<string_t>(response.finish_reason.value()));
        else
          out->insert_or_assign("finish_reason", create<null_t>());
        return ok_output_t(out, (response.finish_reason.has_value() &&
                                 response.finish_reason.value() == "stop"));
      });

  // Define inference op
  auto apply_chat_template = ailoy::create<instant_method_operator_t>(
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        // Validate inputs
        if (!inputs->is_type_of<map_t>())
          return error_output_t(
              type_error("TVM Language Model: apply_chat_template", "inputs",
                         "map_t", inputs->get_type()));

        auto input_map = inputs->as<map_t>();

        // Get input messages
        auto error = _validate_language_model_messages(
            "TVM Language Model: apply_chat_template", input_map);
        if (error.has_value())
          return error.value();
        auto messages = input_map->at<array_t>("messages")->to_nlohmann_json();

        // Get tools (optional)
        nlohmann::json tools;
        if (input_map->contains("tools")) {
          if (input_map->at("tools")->is_type_of<string_t>()) {
            const std::string tools_str = *input_map->at<string_t>("tools");
            if (!nlohmann::json::accept(tools_str))
              return error_output_t(
                  value_error("[TVM Language Model: apply_chat_template] "
                              "Invalid JSON string in tools: " +
                              tools_str));
            tools = nlohmann::json::parse(tools_str);
          } else if (input_map->at("tools")->is_type_of<array_t>()) {
            tools = input_map->at<array_t>("tools")->operator nlohmann::json();
          } else {
            return error_output_t(type_error(
                "TVM Language Model: apply_chat_template", "tools",
                "string_t | array_t", input_map->at("tools")->get_type()));
          }
        }

        // Get reasoning (optional)
        bool enable_reasoning = false;
        if (input_map->contains("enable_reasoning"))
          if (input_map->at("enable_reasoning")->is_type_of<bool_t>())
            enable_reasoning = *input_map->at<bool_t>("enable_reasoning");

        // Apply chat template on messages
        auto prompt =
            component->get_obj("template_engine")
                ->as<chat_template_engine_t>()
                ->apply_chat_template(messages, tools, enable_reasoning);

        auto outputs = create<map_t>();
        outputs->insert_or_assign("prompt", create<string_t>(prompt));
        return outputs;
      });

  // Create component
  auto ops = std::initializer_list<
      std::pair<const std::string, std::shared_ptr<method_operator_t>>>{
      {"infer", infer}, {"apply_chat_template", apply_chat_template}};
  auto rv = create<component_t>(ops);
  rv->set_obj("engine", engine);
  rv->set_obj("template_engine", template_engine);
  return rv;
};

} // namespace ailoy
