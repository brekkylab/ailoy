#include "language.hpp"

#include "chromadb_vector_store.hpp"
#include "faiss/faiss_vector_store.hpp"
#include "mlc_llm/embedding_model.hpp"
#include "mlc_llm/mlc_llm_engine.hpp"
#include "openai.hpp"
#include "split_text.hpp"
#include "uuid.hpp"

namespace ailoy {

static std::shared_ptr<module_t> language_module = create<module_t>();

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

std::shared_ptr<const module_t> get_language_module() {
  if (!language_module->factories.contains("tvm_embedding_model")) {
    language_module->factories.insert_or_assign(
        "tvm_embedding_model", create_tvm_embedding_model_component);
  }
  if (!language_module->factories.contains("tvm_language_model")) {
    component_factory_t fcreate =
        [](std::shared_ptr<const value_t> inputs) -> component_or_error_t {
      if (!inputs->is_type_of<map_t>())
        return error_output_t(type_error("TVM Language Model: create", "inputs",
                                         "map_t", inputs->get_type()));

      auto input_map = inputs->as<map_t>();

      // Parse model name
      if (!input_map->contains("model"))
        return error_output_t(
            range_error("TVM Language Model: create", "model"));
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
          return error_output_t(type_error(
              "TVM Language Model: create", "quantization", "string_t",
              input_map->at("quantization")->get_type()));
      } else
        quantization = "q4f16_1";

      // Parse mode(optional)
      std::string mode;
      if (input_map->contains("mode")) {
        if (input_map->at("mode")->is_type_of<string_t>()) {
          mode = *input_map->at<string_t>("mode");
          if (mode != "local" && mode != "interactive" && mode != "server")
            return error_output_t(
                value_error("TVM Language Model: create", "mode",
                            "local | interactive | server", mode));
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
      auto engine = get_mlc_llm_engine(model, quantization, device, mode);
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
            auto messages =
                input_map->at<array_t>("messages")->to_nlohmann_json();

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
                tools =
                    input_map->at<array_t>("tools")->operator nlohmann::json();
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
                input_map->at("ignore_reasoning_messages")
                    ->is_type_of<bool_t>())
              ignore_reasoning_messages =
                  *input_map->at<bool_t>("ignore_reasoning_messages");
            component->set_obj(
                "ignore_reasoning_messages",
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
              bool dismiss_reasoning = (ignore_reasoning_messages &&
                                        delta.value("reasoning", false));
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
                  type_error("TVM Language Model: apply_chat_template",
                             "inputs", "map_t", inputs->get_type()));

            auto input_map = inputs->as<map_t>();

            // Get input messages
            auto error = _validate_language_model_messages(
                "TVM Language Model: apply_chat_template", input_map);
            if (error.has_value())
              return error.value();
            auto messages =
                input_map->at<array_t>("messages")->to_nlohmann_json();

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
                tools =
                    input_map->at<array_t>("tools")->operator nlohmann::json();
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

    language_module->factories.insert_or_assign("tvm_language_model", fcreate);
  }
  if (!language_module->ops.contains("split_text_by_separator")) {
    auto split_text_by_separator_op =
        [](std::shared_ptr<const value_t> inputs) -> value_or_error_t {
      // Get input parameters
      if (!inputs->is_type_of<map_t>())
        return error_output_t(
            type_error("Split Text", "inputs", "map_t", inputs->get_type()));
      auto input_map = inputs->as<map_t>();

      // Parse text
      if (!input_map->contains("text"))
        return error_output_t(range_error("Split Text", "text"));
      if (!input_map->at("text")->is_type_of<string_t>())
        return error_output_t(type_error("Split Text", "text", "string_t",
                                         input_map->at("text")->get_type()));
      const std::string &text = *input_map->at<string_t>("text");

      // Parse chunk size
      size_t chunk_size = 4000;
      if (input_map->contains("chunk_size")) {
        if (input_map->at("chunk_size")->is_type_of<uint_t>())
          chunk_size = *input_map->at<uint_t>("chunk_size");
        else if (input_map->at("chunk_size")->is_type_of<int_t>())
          chunk_size = *input_map->at<int_t>("chunk_size");
        else
          return error_output_t(
              type_error("Split Text", "chunk_size", "uint_t | int_t",
                         input_map->at("chunk_size")->get_type()));
        if (chunk_size < 1)
          return error_output_t(value_error("Split Text", "chunk_size", ">= 1",
                                            std::to_string(chunk_size)));
      }

      // Parse chunk overlap
      size_t chunk_overlap = 200;
      if (input_map->contains("chunk_overlap")) {
        if (input_map->at("chunk_overlap")->is_type_of<uint_t>())
          chunk_overlap = *input_map->at<uint_t>("chunk_overlap");
        else if (input_map->at("chunk_overlap")->is_type_of<int_t>())
          chunk_overlap = *input_map->at<int_t>("chunk_overlap");
        else
          return error_output_t(
              type_error("Split Text", "chunk_overlap", "uint_t | int_t",
                         input_map->at("chunk_overlap")->get_type()));
        if (chunk_overlap < 1)
          return error_output_t(value_error("Split Text", "chunk_overlap",
                                            ">= 1",
                                            std::to_string(chunk_overlap)));
      }

      // Parse separator
      std::string separator = "\n\n";
      if (input_map->contains("separator")) {
        if (!input_map->at("separator")->is_type_of<string_t>())
          return error_output_t(
              type_error("Split Text", "separator", "string_t",
                         input_map->at("separator")->get_type()));
        separator = *input_map->at<string_t>("separator");
      }

      // Parse length_function
      std::string length_function = "default";
      if (input_map->contains("length_function")) {
        if (!input_map->at("length_function")->is_type_of<string_t>())
          return error_output_t(
              type_error("Split Text", "length_function", "string_t",
                         input_map->at("length_function")->get_type()));
        length_function = *input_map->at<string_t>("length_function");
        if (!(length_function == "default" || length_function == "string"))
          return error_output_t(value_error("Split Text", "length_function",
                                            "default | string",
                                            length_function));
      }

      // Split text into chunks
      auto chunks = split_text_by_separator(text, chunk_size, chunk_overlap,
                                            separator, length_function);

      // Return output
      auto outputs = create<map_t>();
      outputs->insert_or_assign("chunks", create<array_t>());
      for (const auto &chunk : chunks)
        outputs->at<array_t>("chunks")->push_back(create<string_t>(chunk));
      return outputs;
    };
    language_module->ops.insert_or_assign(
        "split_text_by_separator",
        create<instant_operator_t>(split_text_by_separator_op));
  }
  if (!language_module->ops.contains("split_text") ||
      !language_module->ops.contains("split_text_separators_recursively")) {
    auto split_text_by_separators_recursively_op =
        [](std::shared_ptr<const value_t> inputs) -> value_or_error_t {
      // Get input parameters
      if (!inputs->is_type_of<map_t>())
        return error_output_t(
            type_error("Split Text", "inputs", "map_t", inputs->get_type()));
      auto input_map = inputs->as<map_t>();

      // Parse text
      if (!input_map->contains("text"))
        return error_output_t(range_error("Split Text", "text"));
      if (!input_map->at("text")->is_type_of<string_t>())
        return error_output_t(type_error("Split Text", "text", "string_t",
                                         input_map->at("text")->get_type()));
      const std::string &text = *input_map->at<string_t>("text");

      // Parse chunk size
      size_t chunk_size = 4000;
      if (input_map->contains("chunk_size")) {
        if (input_map->at("chunk_size")->is_type_of<uint_t>())
          chunk_size = *input_map->at<uint_t>("chunk_size");
        else if (input_map->at("chunk_size")->is_type_of<int_t>())
          chunk_size = *input_map->at<int_t>("chunk_size");
        else
          return error_output_t(
              type_error("Split Text", "chunk_size", "uint_t | int_t",
                         input_map->at("chunk_size")->get_type()));
        if (chunk_size < 1)
          return error_output_t(value_error("Split Text", "chunk_size", ">= 1",
                                            std::to_string(chunk_size)));
      }

      // Parse chunk overlap
      size_t chunk_overlap = 200;
      if (input_map->contains("chunk_overlap")) {
        if (input_map->at("chunk_overlap")->is_type_of<uint_t>())
          chunk_overlap = *input_map->at<uint_t>("chunk_overlap");
        else if (input_map->at("chunk_overlap")->is_type_of<int_t>())
          chunk_overlap = *input_map->at<int_t>("chunk_overlap");
        else
          return error_output_t(
              type_error("Split Text", "chunk_overlap", "uint_t | int_t",
                         input_map->at("chunk_overlap")->get_type()));
        if (chunk_overlap < 1)
          return error_output_t(value_error("Split Text", "chunk_overlap",
                                            ">= 1",
                                            std::to_string(chunk_overlap)));
      }

      // Parse separators
      std::vector<std::string> separators;
      if (input_map->contains("separator")) {
        if (!input_map->at("separators")->is_type_of<array_t>())
          return error_output_t(
              type_error("Split Text", "separators", "array_t",
                         input_map->at("separators")->get_type()));
        for (const auto sep : *input_map->at<array_t>("separators")) {
          if (!sep->is_type_of<string_t>())
            return error_output_t(type_error("Split Text", "separators.*",
                                             "string_t", sep->get_type()));
          separators.push_back(*sep->as<string_t>());
        }
      } else {
        separators.push_back("\n\n");
        separators.push_back("\n");
        separators.push_back(" ");
        separators.push_back("");
      }

      // Parse length_function
      std::string length_function = "default";
      if (input_map->contains("length_function")) {
        if (!input_map->at("length_function")->is_type_of<string_t>())
          return error_output_t(
              type_error("Split Text", "length_function", "string_t",
                         input_map->at("length_function")->get_type()));
        length_function = *input_map->at<string_t>("length_function");
        if (!(length_function == "default" || length_function == "string"))
          return error_output_t(value_error("Split Text", "length_function",
                                            "default | string",
                                            length_function));
      }

      // Split text into chunks
      auto chunks = split_text_by_separators_recursively(
          text, chunk_size, chunk_overlap, separators, length_function);

      // Return output
      auto outputs = create<map_t>();
      outputs->insert_or_assign("chunks", create<array_t>());
      for (const auto &chunk : chunks)
        outputs->at<array_t>("chunks")->push_back(create<string_t>(chunk));
      return outputs;
    };
    language_module->ops.insert_or_assign(
        "split_text_separators_recursively",
        create<instant_operator_t>(split_text_by_separators_recursively_op));
    language_module->ops.insert_or_assign(
        "split_text",
        create<instant_operator_t>(split_text_by_separators_recursively_op));
  }
  if (!language_module->factories.contains("faiss_vector_store")) {
    language_module->factories.insert_or_assign(
        "faiss_vector_store",
        create_vector_store_component<faiss_vector_store_t>);
  }
  if (!language_module->factories.contains("chromadb_vector_store")) {
    language_module->factories.insert_or_assign(
        "chromadb_vector_store",
        create_vector_store_component<chromadb_vector_store_t>);
  }
  if (!language_module->factories.contains("openai")) {
    language_module->factories.insert_or_assign("openai",
                                                create_openai_component);
  }

  return language_module;
}

} // namespace ailoy
