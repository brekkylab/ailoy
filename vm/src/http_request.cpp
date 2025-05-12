#include <httplib.h>

#include "exception.hpp"
#include "http_request.hpp"
#include "module.hpp"

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

} // namespace ailoy