#include "openai.hpp"

#include <format>

#include "http.hpp"
#include "logging.hpp"

namespace ailoy {

openai_llm_engine_t::openai_llm_engine_t(const std::string &api_key,
                                         const std::string &model)
    : api_key_(api_key), model_(model) {}

openai_response_delta_t
openai_llm_engine_t::infer(std::shared_ptr<const value_t> input) {
  nlohmann::json body = convert_request_body(input);
  debug("[{}] Request body: {}", name(), body.dump());

  auto res = ailoy::http::request({
      .url = std::format("{}{}", api_url(), api_path()),
      .method = ailoy::http::method_t::POST,
      .headers = headers(),
      .body = body.dump(),
  });

  if (!res)
    throw ailoy::runtime_error(
        std::format("[{}] Request failed: {}", name(), res.error()));

  if (res->status_code != ailoy::http::OK_200) {
    debug("[{}] {}", res->status_code, res->body);
    throw ailoy::runtime_error(std::format(
        "[{}] Request failed: [{}] {}", name(), res->status_code, res->body));
  }

  auto j = nlohmann::json::parse(res->body);
  postprocess_response_body(j);
  debug("[{}] Processed Response body: {}", name(), j.dump());

  openai_chat_completion_response_choice_t choice = j["choices"][0];
  auto delta = openai_response_delta_t{.message = choice.message,
                                       .finish_reason = choice.finish_reason};
  return delta;
}

nlohmann::json openai_llm_engine_t::convert_request_body(
    std::shared_ptr<const value_t> inputs) {
  if (!inputs->is_type_of<map_t>())
    throw ailoy::exception(std::format("[{}] input should be a map", name()));

  auto input_map = inputs->as<map_t>();
  if (!input_map->contains("messages") ||
      !input_map->at("messages")->is_type_of<array_t>()) {
    throw ailoy::exception(std::format(
        "[{}] input should have array type field 'messages'", name()));
  }

  auto request = std::make_unique<openai_chat_completion_request_t>(
      input_map->to_nlohmann_json());
  request->model = model_;

  // Dump with function call arguments as string
  nlohmann::json body = request->to_json(true);
  return body;
}

void openai_llm_engine_t::postprocess_response_body(nlohmann::json &body) {
  // do nothing
}

nlohmann::json claude_llm_engine_t::convert_request_body(
    std::shared_ptr<const value_t> inputs) {
  auto body = openai_llm_engine_t::convert_request_body(inputs);

  // Claude does not take empty tools list.
  // We need to remove it if it's empty.
  if (body["tools"].get<std::vector<nlohmann::json>>().size() == 0) {
    body.erase("tools");
  }

  // Claude takes the tool result content as a string,
  // not an array of {"type": "text", "text": "..."}
  for (auto &message : body["messages"]) {
    if (message["role"] == "tool") {
      if (message["content"].is_string())
        continue;
      if (message["content"].size() == 0)
        continue;
      if (message["content"][0]["type"] != "text")
        continue;
      message["content"] = message["content"][0]["text"];
    }
  }

  return body;
}

void grok_llm_engine_t::postprocess_response_body(nlohmann::json &body) {
  // Grok returns finish_reason as "stop" even tool calling has been invoked.
  // So we modify finish_reason to "tool_calls" to make it compatible to other
  // API services.
  for (auto &choice : body["choices"]) {
    if (choice["message"].contains("tool_calls")) {
      choice["finish_reason"] = "tool_calls";
    }
  }
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

} // namespace ailoy