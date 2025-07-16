#include <format>

#if defined(EMSCRIPTEN)
#include <emscripten.h>
#include <emscripten/fetch.h>
#else
#include <httplib.h>
#endif

#include <magic_enum/magic_enum.hpp>
#include <nlohmann/json.hpp>

#include "exception.hpp"
#include "http.hpp"

namespace ailoy {
namespace http {

struct url_t {
  std::string scheme;
  std::string host;
  std::string path;
};

std::variant<url_t, std::string> parse_url(const std::string &url) {
  url_t result;

  // 1) Extract scheme (e.g., 'http://')
  std::size_t scheme_end = url.find("://");
  if (scheme_end == std::string::npos) {
    return "Missing scheme";
  }
  result.scheme = url.substr(0, scheme_end);
  scheme_end += 3; // Skip past "://"

  // 2) Extract host and path
  if (scheme_end < url.size()) {
    std::size_t slash_pos = url.find('/', scheme_end);
    if (slash_pos == std::string::npos) {
      result.host = url.substr(scheme_end);
      result.path = "/";
    } else {
      result.host = url.substr(scheme_end, slash_pos - scheme_end);
      result.path = url.substr(slash_pos);
    }
  }
  if (result.host.empty()) {
    return "Host is empty";
  }
  return result;
}

#if defined(EMSCRIPTEN)

EM_ASYNC_JS(char *, do_fetch_async,
            (const char *url, const char *method, const char *body,
             const char *headers_json),
            {
              // clang-format off
              const urlStr = UTF8ToString(url);
              const methodStr = UTF8ToString(method);
              const bodyStr = UTF8ToString(body);
              const headersJsonStr = UTF8ToString(headers_json);

              try {
                // Parse headers
                let headers = {};
                if (headersJsonStr.length > 0) {
                  headers = JSON.parse(headersJsonStr);
                }

                // Prepare fetch options
                let fetchOptions = {
                  method: methodStr, 
                  headers: headers,
                };

                // Add body if provided and method supports it
                if (bodyStr.length > 0 && methodStr !== 'GET') {
                  fetchOptions.body = bodyStr;
                }

                // Perform fetch
                const response = await fetch(urlStr, fetchOptions);

                // Read response body
                const responseBody = await response.text();

                // Extract headers
                const responseHeaders = {};
                for (let [key, value] of response.headers) {
                  responseHeaders[key] = value;
                }

                const result = {
                  status_code: response.status,
                  body: responseBody,
                  headers: responseHeaders,
                  error: null,
                };

                const resultJson = JSON.stringify(result);
                return stringToNewUTF8(resultJson);

              } catch (error) {
                const errorResult = {
                  status_code: -1,
                  body: "",
                  headers: {},
                  error: error.toString(),
                };

                const resultJson = JSON.stringify(errorResult);
                return stringToNewUTF8(resultJson);
              }
              // clang-format on
            });

// Helper function to convert headers map to JSON string
std::string headers_to_json(const headers_t &headers) {
  nlohmann::json json_obj(headers);
  return json_obj.dump();
}

result_t request(const request_t &req) {
  std::string method_str = std::string(magic_enum::enum_name(req.method));
  std::string headers_json = headers_to_json(req.headers);
  std::string body_str = req.body.value_or("");

  // Call async fetch function - this will block until complete due to ASYNCIFY
  char *result_json_ptr =
      do_fetch_async(req.url.c_str(), method_str.c_str(), body_str.c_str(),
                     headers_json.c_str());

  if (!result_json_ptr) {
    return result_t(nullptr, "Failed to get result from JavaScript");
  }

  // Parse the JSON result
  std::string result_json(result_json_ptr);
  free(result_json_ptr); // Free the memory allocated by stringToNewUTF8

  try {
    nlohmann::json json_result = nlohmann::json::parse(result_json);

    // Check for error
    if (!json_result["error"].is_null()) {
      std::string error = json_result["error"].get<std::string>();
      return result_t(nullptr, std::format("Fetch failed: {}", error));
    }

    // Create response
    response_t response;
    response.status_code = json_result["status_code"].get<int>();
    response.body = json_result["body"].get<std::string>();
    response.headers = json_result["headers"].get<headers_t>();

    return result_t(std::make_unique<response_t>(response));

  } catch (const nlohmann::json::exception &e) {
    return result_t(nullptr, std::format("JSON parsing error: {}", e.what()));
  }
}

#else

result_t request(const request_t &req) {
  auto parse_result = parse_url(req.url);
  if (parse_result.index() != 0) {
    return result_t(nullptr, std::format("URL parsing failed: {}",
                                         std::get<1>(parse_result)));
  }
  auto url_parsed = std::get<0>(parse_result);

  std::variant<std::shared_ptr<httplib::Client>,
               std::shared_ptr<httplib::SSLClient>>
      client;

  if (url_parsed.scheme == "https") {
    auto client_ = std::make_shared<httplib::SSLClient>(url_parsed.host);
    client_->set_follow_location(true);
    client_->set_connection_timeout(5, 0);
    client_->set_read_timeout(60, 0);
    client_->enable_server_certificate_verification(false);
    client_->enable_server_hostname_verification(false);
    client = client_;
  } else {
    auto client_ = std::make_shared<httplib::Client>(url_parsed.host);
    client_->set_follow_location(true);
    client_->set_connection_timeout(5, 0);
    client_->set_read_timeout(60, 0);
    client = client_;
  }

  httplib::Headers httplib_headers;
  for (const auto &[key, value] : req.headers) {
    httplib_headers.emplace(key, value);
  }

  httplib::Result result;
  std::string body_str = req.body.has_value() ? req.body->data() : "";

  std::string content_type;
  if (req.headers.contains("Content-Type")) {
    content_type = req.headers.find("Content-Type")->second;
  } else {
    content_type = "text/plain";
  }

  if (req.method == method_t::GET) {
    if (req.data_callback.has_value()) {
      // If data_callback is provided, the response body becomes empty.
      // The data_callback is responsible for handling the arrived partial data.
      std::visit(
          [&](auto &&client_) {
            result = client_->Get(
                url_parsed.path, httplib_headers,
                [&](const char *data, size_t data_length) {
                  return req.data_callback.value()(data, data_length);
                },
                [&](uint64_t current, uint64_t total) {
                  if (req.progress_callback.has_value()) {
                    return req.progress_callback.value()(current, total);
                  }
                  return true;
                });
          },
          client);
    } else {
      std::visit(
          [&](auto &&client_) {
            result = client_->Get(url_parsed.path, httplib_headers,
                                  [&](uint64_t current, uint64_t total) {
                                    if (req.progress_callback.has_value()) {
                                      return req.progress_callback.value()(
                                          current, total);
                                    }
                                    return true;
                                  });
          },
          client);
    }
  } else if (req.method == method_t::POST) {
    std::visit(
        [&](auto &&client_) {
          result = client_->Post(
              url_parsed.path, httplib_headers, body_str, content_type,
              [&](uint64_t current, uint64_t total) {
                if (req.progress_callback.has_value()) {
                  return req.progress_callback.value()(current, total);
                }
                return true;
              });
        },
        client);
  } else if (req.method == method_t::PUT) {
    std::visit(
        [&](auto &&client_) {
          result = client_->Put(
              url_parsed.path, httplib_headers, body_str, content_type,
              [&](uint64_t current, uint64_t total) {
                if (req.progress_callback.has_value()) {
                  return req.progress_callback.value()(current, total);
                }
                return true;
              });
        },
        client);
  } else if (req.method == method_t::PATCH) {
    std::visit(
        [&](auto &&client_) {
          result = client_->Patch(
              url_parsed.path, httplib_headers, body_str, content_type,
              [&](uint64_t current, uint64_t total) {
                if (req.progress_callback.has_value()) {
                  return req.progress_callback.value()(current, total);
                }
                return true;
              });
        },
        client);
  } else if (req.method == method_t::DELETE) {
    std::visit(
        [&](auto &&client_) {
          result = client_->Delete(
              url_parsed.path, httplib_headers, body_str, content_type,
              [&](uint64_t current, uint64_t total) {
                if (req.progress_callback.has_value()) {
                  return req.progress_callback.value()(current, total);
                }
                return true;
              });
        },
        client);
  } else {
    return result_t(nullptr, std::format("Unsupported HTTP method: {}",
                                         magic_enum::enum_name(req.method)));
  }

  if (!result) {
    return result_t(nullptr, httplib::to_string(result.error()));
  }

  response_t response;
  response.status_code = result->status;
  response.body = result->body;
  for (const auto &[key, value] : result->headers) {
    response.headers[key] = value;
  }

  return result_t(std::make_unique<response_t>(response));
}

#endif

} // namespace http
} // namespace ailoy