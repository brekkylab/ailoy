#include "gemini.hpp"

#include <format>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "logging.hpp"

namespace ailoy {

gemini_llm_engine_t::gemini_llm_engine_t(const std::string &api_key,
                                         const std::string &model)
    : api_key_(api_key), model_(model) {}

openai_response_delta_t gemini_llm_engine_t::infer(
    std::unique_ptr<openai_chat_completion_request_t> request) {
  httplib::Client client("https://generativelanguage.googleapis.com");
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
  nlohmann::json body = request->to_json();

  httplib::Request http_req;
  std::stringstream response_body;

  http_req.method = "POST";
  http_req.path = "/v1beta/openai/chat/completions";
  http_req.headers = httplib_headers;
  http_req.body = body.dump();

  auto result = client.send(http_req);

  if (!result)
    throw ailoy::runtime_error("[Gemini] Request failed: " +
                               std::string(httplib::to_string(result.error())));

  if (result->status != httplib::OK_200) {
    debug("[{}] {}", result->status, result->body);
    throw ailoy::runtime_error(std::format("[Gemini] Request failed: [{}] {}",
                                           result->status, result->body));
  }

  auto j = nlohmann::json::parse(result->body);
  auto choice =
      from_json_to_openai_chat_completion_response_choice_t(j["choices"][0]);
  auto delta = openai_response_delta_t{.message = choice.message,
                                       .finish_reason = choice.finish_reason};
  return delta;
}

component_or_error_t
create_gemini_component(std::shared_ptr<const value_t> attrs) {
  if (!attrs->is_type_of<map_t>())
    throw runtime_error("[Gemini] Invalid input");

  auto data = attrs->as<map_t>();
  std::string api_key = *data->at<string_t>("api_key");
  std::string model = *data->at<string_t>("model", "gemini-2.0-flash");
  auto engine = ailoy::create<gemini_llm_engine_t>(api_key, model);

  auto infer = ailoy::create<instant_method_operator_t>(
      [&](std::shared_ptr<component_t> component,
          std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        auto request = convert_request_input(inputs);
        auto engine = component->get_obj("engine")->as<gemini_llm_engine_t>();
        try {
          auto resp = engine->infer(std::move(request));
          auto rv = ailoy::from_nlohmann_json(resp.to_json())->as<map_t>();
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