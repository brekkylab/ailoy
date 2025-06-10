#pragma once

#include <nlohmann/json.hpp>

#include "openai.hpp"

namespace ailoy {

class gemini_llm_engine_t : public object_t {
public:
  gemini_llm_engine_t(const std::string &api_key, const std::string &model);

  openai_response_delta_t
  infer(std::unique_ptr<openai_chat_completion_request_t> request);

private:
  std::string api_key_;
  std::string model_;
};

component_or_error_t
create_gemini_component(std::shared_ptr<const value_t> attrs);

} // namespace ailoy