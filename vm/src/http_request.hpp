#pragma once

#include <optional>
#include <unordered_map>

#include "module.hpp"
#include "value.hpp"

namespace ailoy {

struct http_request_t {
  std::string url;
  std::string method;
  std::unordered_map<std::string, std::string> headers;
  std::optional<std::string> body = std::nullopt;
};

struct http_response_t {
  int status_code;
  std::unordered_map<std::string, std::string> headers;
  std::string body;
};

http_response_t run_http_request(const http_request_t &req);

value_or_error_t http_request_op(std::shared_ptr<const value_t> inputs);

} // namespace ailoy
