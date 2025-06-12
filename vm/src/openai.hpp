#pragma once

#include <optional>
#include <vector>

#include <nlohmann/json.hpp>

#include "module.hpp"
#include "openai_schema.hpp"
#include "value.hpp"

namespace ailoy {

class openai_llm_engine_t : public object_t {
public:
  openai_llm_engine_t(const std::string &api_key, const std::string &model);

  std::unique_ptr<openai_chat_completion_request_t>
  convert_request_input(std::shared_ptr<const value_t> inputs);

  openai_response_delta_t
  infer(std::unique_ptr<openai_chat_completion_request_t> request);

protected:
  const std::string name_ = "OpenAI";
  const std::string api_url_ = "https://api.openai.com";
  const std::string api_path_ = "/v1/chat/completions";
  std::string api_key_;
  std::string model_;
};

template <typename engine_t>
  requires std::is_base_of_v<openai_llm_engine_t, engine_t>
component_or_error_t
create_openai_component(std::shared_ptr<const value_t> attrs);

class gemini_llm_engine_t : public openai_llm_engine_t {
private:
  const std::string name_ = "Gemini";
  const std::string api_url_ = "https://generativelanguage.googleapis.com";
  const std::string api_path_ = "/v1beta/openai/chat/completions";
};

} // namespace ailoy
