#pragma once

#include <optional>
#include <vector>

#include <nlohmann/json.hpp>

#include "module.hpp"
#include "value.hpp"

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
};

/* Structs for OpenAI component */

struct openai_response_delta_t {
  openai_chat_completion_message_t message;
  std::string finish_reason;
};

class openai_llm_engine_t : public object_t {
public:
  openai_llm_engine_t(const std::string &api_key, const std::string &model);

  openai_response_delta_t
  infer(std::unique_ptr<openai_chat_completion_request_t> request);

private:
  std::string api_key_;
  std::string model_;
};

component_or_error_t
create_openai_component(std::shared_ptr<const value_t> attrs);

} // namespace ailoy
