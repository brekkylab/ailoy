#include "openai_schema.hpp"

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
  } else {
    j["arguments"] = nlohmann::json::object();
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
  if (obj.content.has_value()) {
    j["content"] = nlohmann::json::array();
    j["content"].push_back(nlohmann::json::object());
    j["content"][0]["type"] = "text";
    j["content"][0]["text"] = obj.content;
  }
  if (obj.name.has_value())
    j["name"] = obj.name;
  if (obj.tool_calls.has_value())
    j["tool_calls"] = obj.tool_calls;
  if (obj.tool_call_id.has_value())
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

nlohmann::json openai_chat_completion_request_t::to_json(
    bool function_call_arguments_as_string) {
  if (function_call_arguments_as_string) {
    g_dump_function_call_arguments_as_string = true;
    nlohmann::json j = *this;
    g_dump_function_call_arguments_as_string = false;
    return j;
  } else {
    return *this;
  }
}

} // namespace ailoy