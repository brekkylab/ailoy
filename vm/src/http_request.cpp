#include "http_request.hpp"

#include <httplib.h>

#include "exception.hpp"

namespace ailoy {

http_response_t run_http_request(const http_request_t &req) {
  auto url = req.url;
  auto method = req.method;
  auto headers = req.headers;
  auto body = req.body;

  std::string host, path;
  size_t pos = url.find("://");
  if (pos != std::string::npos) {
    pos += 3;
    size_t slash_pos = url.find("/", pos);
    if (slash_pos == std::string::npos) {
      host = url.substr(pos);
      path = "/";
    } else {
      host = url.substr(pos, slash_pos - pos);
      path = url.substr(slash_pos);
    }
  } else {
    return http_response_t{
        .status_code = 400,
        .headers = {},
        .body = "Invalid URL format",
    };
  }

  httplib::Client client(host);
  client.set_follow_location(true);
  client.set_read_timeout(5, 0);
  client.set_connection_timeout(5, 0);

  httplib::Headers httplib_headers;
  for (const auto &[key, value] : req.headers) {
    httplib_headers.emplace(key, value);
  }

  http_response_t response;
  httplib::Result result;
  std::string body_str = body.has_value() ? body->data() : "";

  std::string content_type;
  if (req.headers.contains("Content-Type")) {
    content_type = req.headers.find("Content-Type")->second;
  } else {
    content_type = "text/plain";
  }

  if (req.method == "GET") {
    result = client.Get(path, httplib_headers);
  } else if (req.method == "POST") {
    result = client.Post(path, httplib_headers, body_str, content_type);
  } else if (req.method == "PUT") {
    result = client.Put(path, httplib_headers, body_str, content_type);
  } else if (req.method == "DELETE") {
    result = client.Delete(path, httplib_headers);
  } else {
    return http_response_t{
        .status_code = 400,
        .headers = {},
        .body = "Unsupported HTTP method",
    };
  }

  if (result) {
    response.status_code = result->status;
    response.body = result->body;

    for (const auto &[key, value] : result->headers) {
      response.headers[key] = value;
    }
  } else {
    response.status_code = 500;
    response.body = "Request Failed";
  }

  return response;
}

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
        *method == "DELETE")) {
    return error_output_t(value_error("HTTP Request", "method",
                                      "GET | POST | PUT | DELETE", *method));
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

  std::unordered_map<std::string, std::string> req_headers_map;
  for (auto it = headers->begin(); it != headers->end(); it++) {
    std::string key = it->first;
    std::string value = *it->second->as<string_t>();
    req_headers_map.emplace(key, value);
  }

  ailoy::http_request_t req = {
      .url = *url,
      .method = *method,
      .headers = req_headers_map,
      .body = body->data(),
  };
  auto resp = ailoy::run_http_request(req);

  auto resp_headers_map = create<map_t>();
  for (const auto &[key, value] : resp.headers) {
    resp_headers_map->insert_or_assign(key, create<string_t>(value));
  }

  auto outputs = create<map_t>();
  outputs->insert_or_assign("status_code", create<uint_t>(resp.status_code));
  outputs->insert_or_assign("headers", resp_headers_map);
  outputs->insert_or_assign("body", create<bytes_t>(resp.body));
  return outputs;
}

} // namespace ailoy