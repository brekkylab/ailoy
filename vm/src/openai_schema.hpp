#pragma once

#include <optional>
#include <vector>

#include <nlohmann/json.hpp>

namespace ailoy {

/* Structs for OpenAI API responses */

struct openai_chat_function_call_t {
  std::string name;
  std::optional<nlohmann::json> arguments = std::nullopt;
};

struct openai_chat_tool_call_t {
  std::string id;
  std::string type = "function";
  openai_chat_function_call_t function;
};

struct openai_chat_completion_message_t {
  std::string role;
  std::optional<std::string> content = std::nullopt;
  std::optional<std::string> name = std::nullopt;
  std::optional<std::vector<openai_chat_tool_call_t>> tool_calls = std::nullopt;
  std::optional<std::string> tool_call_id = std::nullopt;
};

struct openai_chat_completion_response_choice_t {
  int index;
  std::string finish_reason;
  openai_chat_completion_message_t message;
};

struct openai_chat_completion_stream_response_choice_t {
  int index;
  openai_chat_completion_message_t delta;
  std::optional<std::string> finish_reason;
};

/* Structs for OpenAI API requests */

struct openai_chat_function_t {
  std::string name;
  std::optional<std::string> description = std::nullopt;
  nlohmann::json parameters;
};

struct openai_chat_tool_t {
  std::string type = "function";
  openai_chat_function_t function;
};

struct openai_chat_completion_request_t {
  std::vector<openai_chat_completion_message_t> messages;
  std::optional<std::string> model = std::nullopt;
  std::optional<std::vector<openai_chat_tool_t>> tools = std::nullopt;

  nlohmann::json to_json(bool function_call_arguments_as_string = false);
};

/* Structs for OpenAI component */

struct openai_response_delta_t {
  openai_chat_completion_message_t message;
  std::string finish_reason;
};

} // namespace ailoy

namespace nlohmann {

void to_json(json &j, const ailoy::openai_chat_function_call_t &obj);
void from_json(const json &j, ailoy::openai_chat_function_call_t &obj);

void to_json(json &j, const ailoy::openai_chat_tool_call_t &obj);
void from_json(const json &j, ailoy::openai_chat_tool_call_t &obj);

void to_json(json &j, const ailoy::openai_chat_completion_message_t &obj);
void from_json(const json &j, ailoy::openai_chat_completion_message_t &obj);

void to_json(json &j,
             const ailoy::openai_chat_completion_response_choice_t &obj);
void from_json(const json &j,
               ailoy::openai_chat_completion_response_choice_t &obj);

void to_json(json &j, const ailoy::openai_chat_function_t &obj);
void from_json(const json &j, ailoy::openai_chat_function_t &obj);

void to_json(json &j, const ailoy::openai_chat_tool_t &obj);
void from_json(const json &j, ailoy::openai_chat_tool_t &obj);

void to_json(json &j, const ailoy::openai_chat_completion_request_t &obj);
void from_json(const json &j, ailoy::openai_chat_completion_request_t &obj);

void to_json(json &j, const ailoy::openai_response_delta_t &obj);
void from_json(const json &j, ailoy::openai_response_delta_t &obj);

} // namespace nlohmann
