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
  virtual std::string name() const { return "OpenAI"; }
  virtual std::string api_url() const { return "https://api.openai.com"; }
  virtual std::string api_path() const { return "/v1/chat/completions"; }
  std::string api_key_;
  std::string model_;
};

class gemini_llm_engine_t : public openai_llm_engine_t {
public:
  gemini_llm_engine_t(const std::string &api_key, const std::string &model)
      : openai_llm_engine_t(api_key, model) {}

private:
  std::string name() const override { return "Gemini"; }
  std::string api_url() const override {
    return "https://generativelanguage.googleapis.com";
  }
  std::string api_path() const override {
    return "/v1beta/openai/chat/completions";
  }
};

template <typename engine_t>
  requires std::is_base_of_v<openai_llm_engine_t, engine_t>
component_or_error_t
create_openai_component(std::shared_ptr<const value_t> attrs) {
  if (!attrs->is_type_of<map_t>())
    throw runtime_error("Invalid input");

  auto data = attrs->as<map_t>();
  std::string api_key = *data->at<string_t>("api_key");
  std::string model = *data->at<string_t>("model");
  auto engine = ailoy::create<engine_t>(api_key, model);

  auto infer = ailoy::create<instant_method_operator_t>(
      [&](std::shared_ptr<component_t> component,
          std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        auto engine = component->get_obj("engine")->as<engine_t>();
        auto request = engine->convert_request_input(inputs);
        try {
          auto resp = engine->infer(std::move(request));
          auto rv = ailoy::from_nlohmann_json(resp)->template as<map_t>();
          return rv;
        } catch (const ailoy::runtime_error &e) {
          return error_output_t(e.what());
        }
      });

  auto ops = std::initializer_list<
      std::pair<const std::string, std::shared_ptr<method_operator_t>>>{
      {"infer", infer}};
  auto rv = ailoy::create<component_t>(ops);
  rv->set_obj("engine", engine);
  return rv;
}

} // namespace ailoy
