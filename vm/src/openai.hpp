#pragma once

#include <nlohmann/json.hpp>

#include "module.hpp"
#include "openai_schema.hpp"
#include "value.hpp"

namespace ailoy {

class openai_llm_engine_t : public object_t {
public:
  openai_llm_engine_t(const std::string &api_key, const std::string &model);

  openai_response_delta_t infer(std::shared_ptr<const value_t> input);

protected:
  virtual inline std::string name() const { return "OpenAI"; }
  virtual inline std::string api_url() const {
    return "https://api.openai.com";
  }
  virtual inline std::string api_path() const { return "/v1/chat/completions"; }
  virtual inline std::unordered_map<std::string, std::string> headers() {
    return {
        {"Authorization", "Bearer " + api_key_},
        {"Content-Type", "application/json"},
        {"Cache-Control", "no-cache"},
    };
  }
  virtual nlohmann::json
  convert_request_body(std::shared_ptr<const value_t> inputs);

  virtual void postprocess_response_body(nlohmann::json &body);

  std::string api_key_;
  std::string model_;
};

class gemini_llm_engine_t : public openai_llm_engine_t {
public:
  gemini_llm_engine_t(const std::string &api_key, const std::string &model)
      : openai_llm_engine_t(api_key, model) {}

private:
  inline std::string name() const override { return "Gemini"; }
  inline std::string api_url() const override {
    return "https://generativelanguage.googleapis.com";
  }
  inline std::string api_path() const override {
    return "/v1beta/openai/chat/completions";
  }
};

class claude_llm_engine_t : public openai_llm_engine_t {
public:
  claude_llm_engine_t(const std::string &api_key, const std::string &model)
      : openai_llm_engine_t(api_key, model) {}

private:
  inline std::string name() const override { return "Claude"; }
  inline std::string api_url() const override {
    return "https://api.anthropic.com";
  }
  inline std::unordered_map<std::string, std::string> headers() override {
    return {
        {"x-api-key", api_key_},
        {"anthropic-version", "2023-06-01"},
        {"anthropic-dangerous-direct-browser-access",
         "true"}, // necessary for requests from browser
        {"Content-Type", "application/json"},
        {"Cache-Control", "no-cache"},
    };
  }
  nlohmann::json
  convert_request_body(std::shared_ptr<const value_t> inputs) override;
};

class grok_llm_engine_t : public openai_llm_engine_t {
public:
  grok_llm_engine_t(const std::string &api_key, const std::string &model)
      : openai_llm_engine_t(api_key, model) {}

private:
  inline std::string name() const override { return "Grok"; }
  inline std::string api_url() const override { return "https://api.x.ai"; }

  void postprocess_response_body(nlohmann::json &body) override;
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

        try {
          auto resp = engine->infer(inputs);
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
