#include "http_request.hpp"

#include <magic_enum/magic_enum.hpp>

#include "exception.hpp"
#include "http.hpp"

namespace ailoy {

value_or_error_t http_request_op(std::shared_ptr<const value_t> inputs) {
  if (!inputs->is_type_of<map_t>())
    return error_output_t(
        type_error("http_request", "inputs", "map_t", inputs->get_type()));

  auto input_map = inputs->as<map_t>();
  if (!input_map->contains("url"))
    return error_output_t(range_error("http_request", "url"));
  if (!input_map->at("url")->is_type_of<string_t>())
    return error_output_t(type_error("http_request", "url", "string_t",
                                     input_map->at("url")->get_type()));
  auto url = input_map->at<string_t>("url");

  if (!input_map->contains("method"))
    return error_output_t(range_error("http_request", "method"));
  if (!input_map->at("method")->is_type_of<string_t>())
    return error_output_t(type_error("http_request", "method", "string_t",
                                     input_map->at("method")->get_type()));
  auto method = input_map->at<string_t>("method");
  if (!(*method == "GET" || *method == "POST" || *method == "PUT" ||
        *method == "PATCH" || *method == "DELETE")) {
    return error_output_t(value_error("HTTP Request", "method",
                                      "GET | POST | PUT | PATCH | DELETE",
                                      *method));
  }
  std::shared_ptr<const map_t> headers = create<map_t>();
  if (input_map->contains("headers")) {
    if (!input_map->at("headers")->is_type_of<map_t>())
      return error_output_t(type_error("http_request", "headers", "map_t",
                                       input_map->at("headers")->get_type()));
    headers = input_map->at<map_t>("headers");
  }
  std::shared_ptr<const string_t> body = create<string_t>();
  if (input_map->contains("body")) {
    if (!input_map->at("body")->is_type_of<string_t>())
      return error_output_t(type_error("http_request", "body", "string_t",
                                       input_map->at("body")->get_type()));
    body = input_map->at<string_t>("body");
  }

  ailoy::http::headers_t req_headers;
  for (auto it = headers->begin(); it != headers->end(); it++) {
    std::string key = it->first;
    std::string value = *it->second->as<string_t>();
    req_headers.emplace(key, value);
  }

  auto resp = ailoy::http::request({
      .url = *url,
      .method = magic_enum::enum_cast<ailoy::http::method_t>(*method).value(),
      .headers = req_headers,
      .body = body->data(),
  });

  if (!resp) {
    return error_output_t(resp.error());
  }

  auto resp_headers = create<map_t>();
  for (const auto &[key, value] : resp->headers) {
    resp_headers->insert_or_assign(key, create<string_t>(value));
  }

  auto outputs = create<map_t>();
  outputs->insert_or_assign("status_code", create<uint_t>(resp->status_code));
  outputs->insert_or_assign("headers", resp_headers);
  outputs->insert_or_assign("body", create<bytes_t>(resp->body));
  return outputs;
}

} // namespace ailoy