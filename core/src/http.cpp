#include <format>
#include <variant>

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

struct EmscriptenRequestContext {
  std::function<bool(const char *, size_t)> data_callback;
  std::function<bool(uint64_t, uint64_t)> progress_callback;
  std::unique_ptr<response_t> response;
  std::string error_message;
  bool completed = false;
};

// Emscripten fetch callbacks
void on_fetch_success(emscripten_fetch_t *fetch) {
  EmscriptenRequestContext *ctx =
      static_cast<EmscriptenRequestContext *>(fetch->userData);

  // Create response
  ctx->response = std::make_unique<response_t>();
  ctx->response->status_code = fetch->status;
  ctx->response->body = std::string(fetch->data, fetch->numBytes);

  size_t headers_len = emscripten_fetch_get_response_headers_length(fetch);
  std::vector<char> headers_buf(headers_len + 1);
  emscripten_fetch_get_response_headers(fetch, headers_buf.data(),
                                        headers_len + 1);
  char **unpacked_headers =
      emscripten_fetch_unpack_response_headers(headers_buf.data());
  if (unpacked_headers) {
    for (int i = 0; unpacked_headers[i]; i += 2) {
      std::string key = unpacked_headers[i];
      std::string value = unpacked_headers[i + 1];
      ctx->response->headers[key] = value;
    }
  }
  emscripten_fetch_free_unpacked_response_headers(unpacked_headers);

  ctx->completed = true;
  emscripten_fetch_close(fetch);
}

void on_fetch_error(emscripten_fetch_t *fetch) {
  EmscriptenRequestContext *ctx =
      static_cast<EmscriptenRequestContext *>(fetch->userData);
  ctx->error_message =
      std::format("Fetch failed with status: {}", fetch->status);
  ctx->completed = true;
  emscripten_fetch_close(fetch);
}

void on_fetch_progress(emscripten_fetch_t *fetch) {
  EmscriptenRequestContext *ctx =
      static_cast<EmscriptenRequestContext *>(fetch->userData);

  if (ctx->progress_callback) {
    ctx->progress_callback(fetch->dataOffset + fetch->numBytes,
                           fetch->totalBytes);
  }
}

result_t request(const request_t &req) {
  // Create context
  auto ctx = std::make_unique<EmscriptenRequestContext>();
  if (req.data_callback.has_value()) {
    ctx->data_callback = req.data_callback.value();
  }
  if (req.progress_callback.has_value()) {
    ctx->progress_callback = req.progress_callback.value();
  }

  // Setup fetch attributes
  emscripten_fetch_attr_t attr;
  emscripten_fetch_attr_init(&attr);

  // Set method
  std::string method_str = std::string(magic_enum::enum_name(req.method));
  strcpy(attr.requestMethod, method_str.c_str());

  // Set callbacks
  attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;

  // For GET requests with progress callback, enable onprogress.
  if (req.method == method_t::GET && req.progress_callback.has_value()) {
    attr.onprogress = on_fetch_progress;
  }

  attr.onsuccess = on_fetch_success;
  attr.onerror = on_fetch_error;
  attr.userData = ctx.get();

  // Set headers
  std::vector<const char *> headers;
  for (const auto &[key, value] : req.headers) {
    headers.push_back(key.c_str());
    headers.push_back(value.c_str());
  }
  headers.push_back(nullptr); // Null terminate
  attr.requestHeaders = headers.data();

  // Set body for non-GET requests
  if (req.body.has_value() && req.method != method_t::GET) {
    attr.requestData = req.body->c_str();
    attr.requestDataSize = req.body->size();
  }

  // Start fetch
  emscripten_fetch_t *fetch = emscripten_fetch(&attr, req.url.c_str());

  if (!fetch) {
    return result_t(nullptr, "Failed to start fetch");
  }

  // Wait for completion (this will block due to ASYNCIFY)
  while (!ctx->completed) {
    emscripten_sleep(1);
  }

  // NOTE: data_callback is not available inside onprogress due to the "memory
  // access out of bounds" error. So we call it after Emscripten fetch is
  // entirely finished.
  if (req.data_callback.has_value()) {
    ctx->data_callback(ctx->response->body.data(), ctx->response->body.size());
    ctx->response->body.clear();
  }

  return result_t(std::move(ctx->response));
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
  } else if (req.method == method_t::DELETE_) {
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