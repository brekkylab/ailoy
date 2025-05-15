#include "openai.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace ailoy {
/*
  OpenAI returns function call arguments as string by default,
  and it is also expected to be a string when getting messages from user.
  But we don't want to return it as string in our results,
  so it should be converted in a different way depending on the situation.
  This thread_local variable controls how to convert the arguments.
*/
thread_local bool g_dump_function_call_arguments_as_string = false;
} // namespace ailoy

namespace nlohmann {

void to_json(json &j, const ailoy::openai_chat_function_call_t &obj) {
  j = json{{"name", obj.name}};
  if (obj.arguments.has_value()) {
    if (ailoy::g_dump_function_call_arguments_as_string) {
      j["arguments"] = obj.arguments.value().dump();
    } else {
      j["arguments"] = obj.arguments.value();
    }
  }
}

void from_json(const json &j, ailoy::openai_chat_function_call_t &obj) {
  j.at("name").get_to(obj.name);
  if (j.contains("arguments")) {
    if (j["arguments"].is_string()) {
      obj.arguments = nlohmann::json::parse(j["arguments"].get<std::string>());
    } else {
      obj.arguments = j["arguments"];
    }
  }
}

void to_json(json &j, const ailoy::openai_chat_tool_call_t &obj) {
  j = json{{"id", obj.id}, {"type", obj.type}, {"function", obj.function}};
}

void from_json(const json &j, ailoy::openai_chat_tool_call_t &obj) {
  j.at("id").get_to(obj.id);
  j.at("type").get_to(obj.type);
  obj.function = j.at("function");
}

void to_json(json &j, const ailoy::openai_chat_completion_message_t &obj) {
  j = json{{"role", obj.role}};
  j["content"] = obj.content;
  j["name"] = obj.name;
  j["tool_calls"] = obj.tool_calls;
  j["tool_call_id"] = obj.tool_call_id;
}

void from_json(const json &j, ailoy::openai_chat_completion_message_t &obj) {
  j.at("role").get_to(obj.role);
  if (j.contains("content") && !j["content"].is_null())
    obj.content = j.at("content");
  if (j.contains("name") && !j["name"].is_null())
    obj.name = j.at("name");
  if (j.contains("tool_calls") && !j["tool_calls"].is_null())
    obj.tool_calls =
        j.at("tool_calls").get<std::vector<ailoy::openai_chat_tool_call_t>>();
  if (j.contains("tool_call_id") && !j["tool_call_id"].is_null())
    obj.tool_call_id = j.at("tool_call_id");
}

void to_json(json &j,
             const ailoy::openai_chat_completion_response_choice_t &obj) {
  j = json{{"index", obj.index},
           {"finish_reason", obj.finish_reason},
           {"message", obj.message}};
}

void from_json(const json &j,
               ailoy::openai_chat_completion_response_choice_t &obj) {
  j.at("index").get_to(obj.index);
  j.at("finish_reason").get_to(obj.finish_reason);
  j.at("message").get_to(obj.message);
}

void to_json(json &j, const ailoy::openai_chat_function_t &obj) {
  j = json{{"name", obj.name}, {"parameters", obj.parameters}};
  j["description"] = obj.description;
}

void from_json(const json &j, ailoy::openai_chat_function_t &obj) {
  j.at("name").get_to(obj.name);
  j.at("parameters").get_to(obj.parameters);
  if (j.contains("description") && !j["description"].is_null())
    obj.description = j.at("description");
}

void to_json(json &j, const ailoy::openai_chat_tool_t &obj) {
  j = json{{"type", obj.type}, {"function", obj.function}};
}

void from_json(const json &j, ailoy::openai_chat_tool_t &obj) {
  j.at("type").get_to(obj.type);
  j.at("function").get_to(obj.function);
}

void to_json(json &j, const ailoy::openai_chat_completion_request_t &obj) {
  j = json{{"messages", obj.messages}};
  j["model"] = obj.model;
  j["tools"] = obj.tools;
}

void from_json(const json &j, ailoy::openai_chat_completion_request_t &obj) {
  j.at("messages").get_to(obj.messages);
  if (j.contains("model") && !j["model"].is_null())
    obj.model = j.at("model");
  if (j.contains("tools") && !j["tools"].is_null())
    obj.tools = j.at("tools").get<std::vector<ailoy::openai_chat_tool_t>>();
}

void to_json(json &j, const ailoy::openai_response_delta_t &obj) {
  j = json{{"message", obj.message}, {"finish_reason", obj.finish_reason}};
}

void from_json(const json &j, ailoy::openai_response_delta_t &obj) {
  j.at("message").get_to(obj.message);
  j.at("finish_reason").get_to(obj.finish_reason);
}

} // namespace nlohmann

namespace ailoy {

std::unique_ptr<openai_chat_completion_request_t>
convert_request_input(std::shared_ptr<const value_t> inputs) {
  if (!inputs->is_type_of<map_t>())
    throw ailoy::exception("[OpenAI] input should be a map");
  // throw ailoy::exception<type_error>("OpenAI", "inputs", "map_t",
  // inputs->get_type());

  auto request = std::make_unique<openai_chat_completion_request_t>();

  auto input_map = inputs->as<map_t>();
  if (!input_map->contains("messages") ||
      !input_map->at("messages")->is_type_of<array_t>()) {
    throw ailoy::exception(
        "[OpenAI] input should have array type field 'messages'");
  }

  auto messages = input_map->at<array_t>("messages");
  for (const auto &msg_val : *messages) {
    request->messages.push_back(msg_val->to_nlohmann_json());
  }

  if (input_map->contains("tools")) {
    auto tools = input_map->at("tools");
    if (!tools->is_type_of<array_t>()) {
      throw ailoy::exception("[OpenAI] tools should be an array type");
    }
    request->tools = std::vector<openai_chat_tool_t>{};
    for (const auto &tool_val : *tools->as<array_t>()) {
      request->tools.value().push_back(tool_val->to_nlohmann_json());
    }
  }

  return request;
}

openai_llm_engine_t::openai_llm_engine_t(const std::string &api_key,
                                         const std::string &model)
    : api_key_(api_key), model_(model) {}

openai_response_delta_t openai_llm_engine_t::infer(
    std::unique_ptr<openai_chat_completion_request_t> request) {
  httplib::Client client("https://api.openai.com");
  std::unordered_map<std::string, std::string> headers = {
      {"Authorization", "Bearer " + api_key_},
      {"Content-Type", "application/json"},
      {"Cache-Control", "no-cache"},
  };
  httplib::Headers httplib_headers;
  for (const auto &[key, value] : headers) {
    httplib_headers.emplace(key, value);
  }

  request->model = model_;
  // convert function call arguments as string if exists
  g_dump_function_call_arguments_as_string = true;
  nlohmann::json body = *request;
  g_dump_function_call_arguments_as_string = false;

  httplib::Request http_req;
  std::stringstream response_body;

  http_req.method = "POST";
  http_req.path = "/v1/chat/completions";
  http_req.headers = httplib_headers;
  http_req.body = body.dump();

  auto result = client.send(http_req);

  if (!result)
    throw ailoy::runtime_error("[OpenAI] Request failed: " +
                               std::string(httplib::to_string(result.error())));

  if (result->status != httplib::OK_200) {
    std::cout << result->status << " " << result->body << std::endl;
    throw ailoy::runtime_error(std::format("[OpenAI] Request failed: [{}] {}",
                                           result->status, result->body));
  }

  auto j = nlohmann::json::parse(result->body);
  openai_chat_completion_response_choice_t choice = j["choices"][0];
  auto delta = openai_response_delta_t{.message = choice.message,
                                       .finish_reason = choice.finish_reason};
  return delta;
}

// TODO: enable iterative infer
// void infer_stream(const openai_request_t &req) {
//   httplib::Client client("https://api.openai.com");
//   std::unordered_map<std::string, std::string> headers = {
//       {"Authorization", "Bearer " + req.api_key},
//       {"Content-Type", "application/json"},
//       {"Accept", "text/event-stream"},
//       {"Cache-Control", "no-cache"},
//   };
//   httplib::Headers httplib_headers;
//   for (const auto &[key, value] : headers) {
//     httplib_headers.emplace(key, value);
//   }

//   nlohmann::json body = {{"model", req.model},
//                          {"messages", nlohmann::json::array()},
//                          {"stream", true}};
//   for (const auto &msg : req.messages) {
//     body["messages"].push_back({{"role", msg.role}, {"content",
//     msg.content}});
//   }

//   httplib::Request http_req;
//   std::stringstream response_body;

//   http_req.method = "POST";
//   http_req.path = "/v1/chat/completions";
//   http_req.headers = httplib_headers;
//   http_req.body = body.dump();

//   std::string buffer;
//   http_req.content_receiver = [&](const char *data, size_t data_length,
//                                   uint64_t, uint64_t) {
//     std::string chunk(data, data_length);
//     response_body << chunk;
//     buffer += chunk;

//     // Process SSE events
//     size_t pos = 0;
//     while (pos < buffer.length()) {
//       size_t event_end = buffer.find("\n\n", pos);
//       if (event_end == std::string::npos)
//         break;

//       std::string event_block = buffer.substr(pos, event_end - pos);
//       pos = event_end + 2;

//       if (event_block.rfind("data:", 0) != 0)
//         continue;
//       std::string event_data = event_block.substr(5);
//       event_data.erase(0, event_data.find_first_not_of(" \t"));
//       event_data.erase(event_data.find_last_not_of(" \t") + 1);

//       if (event_data == "[DONE]") {
//         break;
//       }

//       try {
//         auto j = nlohmann::json::parse(event_data);
//         if (j.contains("choices") && !j["choices"].empty()) {
//           auto &_delta = j["choices"][0]["delta"];
//           auto &_finish_reason = j["choices"][0]["finish_reason"];
//           std::optional<std::string> content;
//           std::optional<std::string> finish_reason;

//           if (_delta.contains("content") && !_delta["content"].is_null()) {
//             content = _delta["content"].get<std::string>();
//           }
//           if (!_finish_reason.is_null()) {
//             finish_reason = _finish_reason.get<std::string>();
//           }

//           // callback
//           std::cout << "content: " << content.value_or("") << std::endl;
//         }
//       } catch (const std::exception &e) {
//         throw ailoy::runtime_error("[OpenAI] Error parsing SSE data: " +
//                                    std::string(e.what()));
//       }
//     }
//     buffer = buffer.substr(pos);
//     return true;
//   };

//   httplib::Response res;
//   httplib::Error err;
//   bool request_succeeded = client.send(http_req, res, err);
//   if (!request_succeeded) {
//     throw ailoy::runtime_error("[OpenAI] Request failed: " +
//                                std::string(httplib::to_string(err)));
//   }
// }

component_or_error_t
create_openai_component(std::shared_ptr<const value_t> attrs) {
  if (!attrs->is_type_of<map_t>())
    throw runtime_error("[OpenAI] Invalid input");

  auto data = attrs->as<map_t>();
  std::string api_key = *data->at<string_t>("api_key");
  std::string model = *data->at<string_t>("model", "gpt-4o");
  auto engine = ailoy::create<openai_llm_engine_t>(api_key, model);

  auto infer = ailoy::create<instant_method_operator_t>(
      [&](std::shared_ptr<component_t> component,
          std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        auto request = convert_request_input(inputs);
        auto delta =
            component->get_obj("engine")->as<openai_llm_engine_t>()->infer(
                std::move(request));
        auto rv = ailoy::from_nlohmann_json(delta);
        return rv;
      });

  auto ops = std::initializer_list<
      std::pair<const std::string, std::shared_ptr<method_operator_t>>>{
      {"infer", infer}};
  auto rv = ailoy::create<component_t>(ops);
  rv->set_obj("engine", engine);
  return rv;
}

} // namespace ailoy