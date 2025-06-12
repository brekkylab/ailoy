#include "openai.hpp"

#include <format>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "chat_manager.hpp"
#include "logging.hpp"

namespace ailoy {

openai_llm_engine_t::openai_llm_engine_t(const std::string &api_key,
                                         const std::string &model)
    : api_key_(api_key), model_(model) {}

openai_response_delta_t openai_llm_engine_t::infer(
    std::unique_ptr<openai_chat_completion_request_t> request) {
  httplib::Client client(api_url_);
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
  nlohmann::json body = request->to_json(true);

  httplib::Request http_req;
  std::stringstream response_body;

  http_req.method = "POST";
  http_req.path = api_path_;
  http_req.headers = httplib_headers;
  http_req.body = body.dump();

  auto result = client.send(http_req);

  if (!result)
    throw ailoy::runtime_error(
        std::format("[{}] Request failed: {}", name_,
                    std::string(httplib::to_string(result.error()))));

  if (result->status != httplib::OK_200) {
    debug("[{}] {}", result->status, result->body);
    throw ailoy::runtime_error(std::format(
        "[{}] Request failed: [{}] {}", name_, result->status, result->body));
  }

  auto j = nlohmann::json::parse(result->body);
  openai_chat_completion_response_choice_t choice = j["choices"][0];
  auto delta = openai_response_delta_t{.message = choice.message,
                                       .finish_reason = choice.finish_reason};
  return delta;
}

std::unique_ptr<openai_chat_completion_request_t>
openai_llm_engine_t::convert_request_input(
    std::shared_ptr<const value_t> inputs) {
  if (!inputs->is_type_of<map_t>())
    throw ailoy::exception(std::format("[{}] input should be a map", name_));

  auto request = std::make_unique<openai_chat_completion_request_t>();

  auto input_map = inputs->as<map_t>();
  if (!input_map->contains("messages") ||
      !input_map->at("messages")->is_type_of<array_t>()) {
    throw ailoy::exception(std::format(
        "[{}] input should have array type field 'messages'", name_));
  }

  // Apply input conversion
  auto messages = input_map->at<array_t>("messages");
  messages = melt_content_text(messages)->as<array_t>();

  for (const auto &msg_val : *messages) {
    request->messages.push_back(msg_val->to_nlohmann_json());
  }

  if (input_map->contains("tools")) {
    auto tools = input_map->at("tools");
    if (!tools->is_type_of<array_t>()) {
      throw ailoy::exception(
          std::format("[{}] tools should be an array type", name_));
    }
    request->tools = std::vector<openai_chat_tool_t>{};
    for (const auto &tool_val : *tools->as<array_t>()) {
      request->tools.value().push_back(tool_val->to_nlohmann_json());
    }
  }

  return request;
}

// TODO: enable iterative infer
// void infer_stream(const openai_request_t &req) {
//   httplib::Client client("https://api.openai.com");
//   std::unordered_map<std::string, std::string> headers = {
//       {"Authorization", "Bearer " + req.api_key},
//       {"Content-Type", "application/json"},
//       {"Accept", "text/event-stream"},
//       {"Cache-Control", "no-cache"},
//   };
//   httplib::Headers httplib_headers;
//   for (const auto &[key, value] : headers) {
//     httplib_headers.emplace(key, value);
//   }

//   nlohmann::json body = {{"model", req.model},
//                          {"messages", nlohmann::json::array()},
//                          {"stream", true}};
//   for (const auto &msg : req.messages) {
//     body["messages"].push_back({{"role", msg.role}, {"content",
//     msg.content}});
//   }

//   httplib::Request http_req;
//   std::stringstream response_body;

//   http_req.method = "POST";
//   http_req.path = "/v1/chat/completions";
//   http_req.headers = httplib_headers;
//   http_req.body = body.dump();

//   std::string buffer;
//   http_req.content_receiver = [&](const char *data, size_t data_length,
//                                   uint64_t, uint64_t) {
//     std::string chunk(data, data_length);
//     response_body << chunk;
//     buffer += chunk;

//     // Process SSE events
//     size_t pos = 0;
//     while (pos < buffer.length()) {
//       size_t event_end = buffer.find("\n\n", pos);
//       if (event_end == std::string::npos)
//         break;

//       std::string event_block = buffer.substr(pos, event_end - pos);
//       pos = event_end + 2;

//       if (event_block.rfind("data:", 0) != 0)
//         continue;
//       std::string event_data = event_block.substr(5);
//       event_data.erase(0, event_data.find_first_not_of(" \t"));
//       event_data.erase(event_data.find_last_not_of(" \t") + 1);

//       if (event_data == "[DONE]") {
//         break;
//       }

//       try {
//         auto j = nlohmann::json::parse(event_data);
//         if (j.contains("choices") && !j["choices"].empty()) {
//           auto &_delta = j["choices"][0]["delta"];
//           auto &_finish_reason = j["choices"][0]["finish_reason"];
//           std::optional<std::string> content;
//           std::optional<std::string> finish_reason;

//           if (_delta.contains("content") && !_delta["content"].is_null()) {
//             content = _delta["content"].get<std::string>();
//           }
//           if (!_finish_reason.is_null()) {
//             finish_reason = _finish_reason.get<std::string>();
//           }

//           // callback
//           std::cout << "content: " << content.value_or("") << std::endl;
//         }
//       } catch (const std::exception &e) {
//         throw ailoy::runtime_error("[OpenAI] Error parsing SSE data: " +
//                                    std::string(e.what()));
//       }
//     }
//     buffer = buffer.substr(pos);
//     return true;
//   };

//   httplib::Response res;
//   httplib::Error err;
//   bool request_succeeded = client.send(http_req, res, err);
//   if (!request_succeeded) {
//     throw ailoy::runtime_error("[OpenAI] Request failed: " +
//                                std::string(httplib::to_string(err)));
//   }
// }

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