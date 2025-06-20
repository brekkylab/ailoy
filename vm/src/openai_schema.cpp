#include "openai_schema.hpp"

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

/* OpenAI API content parts */

void to_json(json &j, const ailoy::openai_chat_text_content_t &obj) {
  j = json{{"type", obj.type}, {"text", obj.text}};
}
void from_json(const json &j, ailoy::openai_chat_text_content_t &obj) {
  j.at("type").get_to(obj.type);
  j.at("text").get_to(obj.text);
}

void to_json(json &j, const ailoy::openai_chat_image_content_t &obj) {
  j = json{{"type", obj.type}, {"image_url", json{{"url", obj.image_url.url}}}};
}
void from_json(const json &j, ailoy::openai_chat_image_content_t &obj) {
  j.at("type").get_to(obj.type);
  j.at("image_url").at("url").get_to(obj.image_url.url);
}

void to_json(json &j, const ailoy::openai_chat_audio_content_t &obj) {
  j = json{{"type", obj.type},
           {"input_audio", json{{"data", obj.input_audio.data},
                                {"format", obj.input_audio.format}}}};
}
void from_json(const json &j, ailoy::openai_chat_audio_content_t &obj) {
  j.at("type").get_to(obj.type);
  j.at("input_audio").at("data").get_to(obj.input_audio.data);
  j.at("input_audio").at("format").get_to(obj.input_audio.format);
}

/* OpenAI API Tools */

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

void to_json(json &j, const ailoy::openai_chat_function_call_t &obj) {
  j = json{{"name", obj.name}};
  if (obj.arguments.has_value()) {
    if (ailoy::g_dump_function_call_arguments_as_string) {
      j["arguments"] = obj.arguments.value().dump();
    } else {
      j["arguments"] = obj.arguments.value();
    }
  } else {
    j["arguments"] = json::object();
  }
}
void from_json(const json &j, ailoy::openai_chat_function_call_t &obj) {
  j.at("name").get_to(obj.name);
  if (j.contains("arguments")) {
    if (j["arguments"].is_string()) {
      obj.arguments = json::parse(j["arguments"].get<std::string>());
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

/* OpenAI API messages */

void to_json(json &j,
             const ailoy::openai_chat_completion_system_message_t &obj) {
  j = json{{"role", obj.role}};
  std::visit([&j](const auto &v) { j["content"] = v; }, obj.content);
}
void from_json(const json &j,
               ailoy::openai_chat_completion_system_message_t &obj) {
  j.at("role").get_to(obj.role);
  if (j.at("content").is_string()) {
    obj.content = j.at("content").get<std::string>();
  } else {
    obj.content =
        j.at("content").get<std::vector<ailoy::openai_chat_text_content_t>>();
  }
}

void to_json(json &j, const ailoy::openai_chat_completion_user_message_t &obj) {
  j = json{{"role", obj.role}};
  j["content"] = json::array();
  if (obj.content.index() == 0) {
    j["content"] = std::get<0>(obj.content);
  } else if (obj.content.index() == 1) {
    for (auto &item : std::get<1>(obj.content)) {
      std::visit([&j](const auto &v) { j["content"].push_back(v); }, item);
    }
  }
  if (obj.name.has_value())
    j["name"] = obj.name;
}
void from_json(const json &j,
               ailoy::openai_chat_completion_user_message_t &obj) {
  j.at("role").get_to(obj.role);
  if (j.at("content").is_string()) {
    obj.content = j.at("content").get<std::string>();
  } else {
    std::vector<std::variant<ailoy::openai_chat_text_content_t,
                             ailoy::openai_chat_image_content_t,
                             ailoy::openai_chat_audio_content_t>>
        content;
    for (auto &item : j.at("content").get<std::vector<json>>()) {
      std::string type = item.at("type");
      if (type == "text") {
        content.push_back(item.get<ailoy::openai_chat_text_content_t>());
      } else if (type == "image_url") {
        content.push_back(item.get<ailoy::openai_chat_image_content_t>());
      } else if (type == "input_audio") {
        content.push_back(item.get<ailoy::openai_chat_audio_content_t>());
      } else {
        throw std::runtime_error("invalid content type: " + type);
      }
    }
    obj.content = content;
  }
  if (j.contains("name") && !j["name"].is_null())
    obj.name = j.at("name");
}

void to_json(json &j,
             const ailoy::openai_chat_completion_assistant_message_t &obj) {
  j = json{{"role", obj.role}};
  if (obj.content.has_value()) {
    std::visit([&j](const auto &v) { j["content"] = v; }, obj.content.value());
  }
  if (obj.name.has_value())
    j["name"] = obj.name;
  if (obj.tool_calls.has_value()) {
    j["tool_calls"] = obj.tool_calls.value();
  }
}
void from_json(const json &j,
               ailoy::openai_chat_completion_assistant_message_t &obj) {
  j.at("role").get_to(obj.role);
  if (j.contains("content") && !j["content"].is_null()) {
    auto content = j.at("content");
    if (content.is_string()) {
      obj.content = content.get<std::string>();
    } else {
      obj.content =
          content.get<std::vector<ailoy::openai_chat_text_content_t>>();
    }
  }
  if (j.contains("name") && !j["name"].is_null())
    obj.name = j.at("name");
  if (j.contains("tool_calls") && !j["tool_calls"].is_null()) {
    obj.tool_calls = j.at("tool_calls");
  }
}

void to_json(json &j, const ailoy::openai_chat_completion_tool_message_t &obj) {
  j = json{{"role", obj.role}, {"tool_call_id", obj.tool_call_id}};
  std::visit([&j](const auto &v) { j["content"] = v; }, obj.content);
}
void from_json(const json &j,
               ailoy::openai_chat_completion_tool_message_t &obj) {
  j.at("role").get_to(obj.role);
  j.at("tool_call_id").get_to(obj.tool_call_id);
  if (j.at("content").is_string()) {
    obj.content = j.at("content").get<std::string>();
  } else {
    obj.content =
        j.at("content").get<std::vector<ailoy::openai_chat_text_content_t>>();
  }
}

/* OpenAI API requests */

void to_json(json &j, const ailoy::openai_chat_completion_request_t &obj) {
  j["messages"] = json::array();
  for (auto &message : obj.messages) {
    std::visit([&j](const auto &v) { j["messages"].push_back(v); }, message);
  }
  j["model"] = obj.model;
  j["tools"] = obj.tools;
}
void from_json(const json &j, ailoy::openai_chat_completion_request_t &obj) {
  obj.messages = {};
  for (auto &message : j.at("messages").get<std::vector<json>>()) {
    std::string role = message.at("role");
    if (role == "system") {
      obj.messages.push_back(
          message.get<ailoy::openai_chat_completion_system_message_t>());
    } else if (role == "user") {
      obj.messages.push_back(
          message.get<ailoy::openai_chat_completion_user_message_t>());
    } else if (role == "assistant") {
      obj.messages.push_back(
          message.get<ailoy::openai_chat_completion_assistant_message_t>());
    } else if (role == "tool") {
      obj.messages.push_back(
          message.get<ailoy::openai_chat_completion_tool_message_t>());
    } else {
      throw std::runtime_error("invalid role: " + role);
    }
  }
  if (j.contains("model") && !j["model"].is_null())
    obj.model = j.at("model");
  if (j.contains("tools") && !j["tools"].is_null())
    obj.tools = j.at("tools").get<std::vector<ailoy::openai_chat_tool_t>>();
}

/* OpenAI API responses */

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

/* Ailoy OpenAI component */

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