#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace ailoy {

/* Structs for OpenAI API content parts */

struct openai_chat_text_content_t {
  std::string type = "text";
  std::string text;
};

struct openai_chat_image_url_content_t {
  std::string url;
};
struct openai_chat_image_content_t {
  std::string type = "image_url";
  openai_chat_image_url_content_t image_url;
};

struct openai_chat_audio_content_data_t {
  std::string data;
  std::string format;
};
struct openai_chat_audio_content_t {
  std::string type = "input_audio";
  openai_chat_audio_content_data_t input_audio;
};

/* Structs for OpenAI API Tools */

struct openai_chat_function_t {
  std::string name;
  std::optional<std::string> description = std::nullopt;
  nlohmann::json parameters;
};

struct openai_chat_tool_t {
  std::string type = "function";
  openai_chat_function_t function;
};

struct openai_chat_function_call_t {
  std::string name;
  std::optional<nlohmann::json> arguments = std::nullopt;
};

struct openai_chat_tool_call_t {
  std::string id;
  std::string type = "function";
  openai_chat_function_call_t function;
};

/* Structs for OpenAI API messages */

struct openai_chat_completion_system_message_t {
  std::string role = "system";
  std::variant<std::string, std::vector<openai_chat_text_content_t>> content;
};

struct openai_chat_completion_user_message_t {
  std::string role = "user";
  std::variant<std::string,
               std::vector<std::variant<openai_chat_text_content_t,
                                        openai_chat_image_content_t,
                                        openai_chat_audio_content_t>>>
      content;
  std::optional<std::string> name = std::nullopt;
};

struct openai_chat_completion_assistant_message_t {
  std::string role = "assistant";
  std::optional<
      std::variant<std::string, std::vector<openai_chat_text_content_t>>>
      content = std::nullopt;
  std::optional<std::string> name = std::nullopt;
  std::optional<std::vector<openai_chat_tool_call_t>> tool_calls = std::nullopt;
};

struct openai_chat_completion_tool_message_t {
  std::string role = "tool";
  std::variant<std::string, std::vector<openai_chat_text_content_t>> content;
  std::string tool_call_id;
};

/* Structs for OpenAI API requests */

struct openai_chat_completion_request_t {
  std::vector<std::variant<openai_chat_completion_system_message_t,
                           openai_chat_completion_user_message_t,
                           openai_chat_completion_assistant_message_t,
                           openai_chat_completion_tool_message_t>>
      messages;
  std::optional<std::string> model = std::nullopt;
  std::optional<std::vector<openai_chat_tool_t>> tools = std::nullopt;

  nlohmann::json to_json(bool function_call_arguments_as_string = false);
};

/* Structs for OpenAI API responses */

struct openai_chat_completion_response_choice_t {
  int index;
  std::string finish_reason;
  openai_chat_completion_assistant_message_t message;
};

struct openai_chat_completion_stream_response_choice_t {
  int index;
  openai_chat_completion_assistant_message_t delta;
  std::optional<std::string> finish_reason;
};

/* Structs for Ailoy OpenAI component */

struct openai_response_delta_t {
  openai_chat_completion_assistant_message_t message;
  std::string finish_reason;
};

} // namespace ailoy

namespace nlohmann {

/* OpenAI API content parts */
void to_json(json &j, const ailoy::openai_chat_text_content_t &obj);
void from_json(const json &j, ailoy::openai_chat_text_content_t &obj);

void to_json(json &j, const ailoy::openai_chat_image_content_t &obj);
void from_json(const json &j, ailoy::openai_chat_image_content_t &obj);

void to_json(json &j, const ailoy::openai_chat_audio_content_t &obj);
void from_json(const json &j, ailoy::openai_chat_audio_content_t &obj);

/* OpenAI API Tools */

void to_json(json &j, const ailoy::openai_chat_function_t &obj);
void from_json(const json &j, ailoy::openai_chat_function_t &obj);

void to_json(json &j, const ailoy::openai_chat_tool_t &obj);
void from_json(const json &j, ailoy::openai_chat_tool_t &obj);

void to_json(json &j, const ailoy::openai_chat_function_call_t &obj);
void from_json(const json &j, ailoy::openai_chat_function_call_t &obj);

void to_json(json &j, const ailoy::openai_chat_tool_call_t &obj);
void from_json(const json &j, ailoy::openai_chat_tool_call_t &obj);

/* OpenAI API messages */

void to_json(json &j,
             const ailoy::openai_chat_completion_system_message_t &obj);
void from_json(const json &j,
               ailoy::openai_chat_completion_system_message_t &obj);

void to_json(json &j, const ailoy::openai_chat_completion_user_message_t &obj);
void from_json(const json &j,
               ailoy::openai_chat_completion_user_message_t &obj);

void to_json(json &j,
             const ailoy::openai_chat_completion_assistant_message_t &obj);
void from_json(const json &j,
               ailoy::openai_chat_completion_assistant_message_t &obj);

void to_json(json &j, const ailoy::openai_chat_completion_tool_message_t &obj);
void from_json(const json &j,
               ailoy::openai_chat_completion_tool_message_t &obj);

/* OpenAI API requests */

void to_json(json &j, const ailoy::openai_chat_completion_request_t &obj);
void from_json(const json &j, ailoy::openai_chat_completion_request_t &obj);

/* OpenAI API responses */

void to_json(json &j,
             const ailoy::openai_chat_completion_response_choice_t &obj);
void from_json(const json &j,
               ailoy::openai_chat_completion_response_choice_t &obj);

/* Ailoy OpenAI component */

void to_json(json &j, const ailoy::openai_response_delta_t &obj);
void from_json(const json &j, ailoy::openai_response_delta_t &obj);

} // namespace nlohmann
