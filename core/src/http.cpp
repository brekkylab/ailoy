#if defined(EMSCRIPTEN)
#include <emscripten.h>
#include <emscripten/fetch.h>
#else
#include <httplib.h>
#endif

#include <format>

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

EM_ASYNC_JS(void, do_fetch, (const char *url, const char *method), {
  const url = UTF8ToString(url);
  const method = UTF8ToString(method);
  out("waiting for a fetch");
  const response = await fetch(url);
  out("got the fetch response");
  // (normally you would do something with the fetch here)
});

response_t request(const request_t &req) {
  do_fetch(req.url.c_str(), "GET");
  // switch (req.method) {
  // case method_t::GET:
  //   strcpy(attr.requestMethod, "GET");
  //   break;
  // case method_t::POST:
  //   strcpy(attr.requestMethod, "POST");
  //   break;
  // case method_t::PUT:
  //   strcpy(attr.requestMethod, "PUT");
  //   break;
  // case method_t::DELETE:
  //   strcpy(attr.requestMethod, "DELETE");
  //   break;
  // default:
  //   return response_t{
  //       .status_code = 400, .headers = {}, .body = "Unsupported HTTP
  //       method"};
  // }

  // attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;

  // std::string body_str = req.body.value_or("");
  // if (!body_str.empty()) {
  //   attr.requestData = body_str.c_str();
  //   attr.requestDataSize = body_str.size();
  // }

  // std::string header_str;
  // for (const auto &[key, value] : req.headers) {
  //   header_str += key + ": " + value + "\n";
  // }
  // attr.requestHeaders = header_str.c_str();

  // emscripten_fetch_t *fetch = emscripten_fetch(&attr, full_url.c_str());
  // emscripten_fetch_wait(fetch); // ❗ blocks until fetch completes

  // response_t res;
  // res.status_code = fetch->status;
  // res.body = std::string(fetch->data, fetch->numBytes);
  // // NOTE: fetch->responseHeaders not available unless using JS bridge

  // emscripten_fetch_close(fetch);
  // return res;
}

#else

std::unique_ptr<response_t> request(const std::unique_ptr<request_t> &req) {
  auto parse_result = parse_url(req->url);
  if (parse_result.index() != 0) {
    return std::make_unique<response_t>(
        response_t{.status_code = 400,
                   .headers = {},
                   .body = std::format("URL parsing failed: {}",
                                       std::get<1>(parse_result))});
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
  for (const auto &[key, value] : req->headers) {
    httplib_headers.emplace(key, value);
  }

  httplib::Result result;
  std::string body_str = req->body.has_value() ? req->body->data() : "";

  std::string content_type;
  if (req->headers.contains("Content-Type")) {
    content_type = req->headers.find("Content-Type")->second;
  } else {
    content_type = "text/plain";
  }

  if (req->method == method_t::GET) {
    if (req->data_callback.has_value()) {
      // If data_callback is provided, the response body becomes empty.
      // The data_callback is responsible for handling the arrived partial data.
      std::visit(
          [&](auto &&client_) {
            result = client_->Get(
                url_parsed.path, httplib_headers,
                [&](const char *data, size_t data_length) {
                  return req->data_callback.value()(data, data_length);
                },
                [&](uint64_t current, uint64_t total) {
                  if (req->progress_callback.has_value()) {
                    return req->progress_callback.value()(current, total);
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
                                    if (req->progress_callback.has_value()) {
                                      return req->progress_callback.value()(
                                          current, total);
                                    }
                                    return true;
                                  });
          },
          client);
    }
  } else if (req->method == method_t::POST) {
    std::visit(
        [&](auto &&client_) {
          result = client_->Post(
              url_parsed.path, httplib_headers, body_str, content_type,
              [&](uint64_t current, uint64_t total) {
                if (req->progress_callback.has_value()) {
                  return req->progress_callback.value()(current, total);
                }
                return true;
              });
        },
        client);
  } else if (req->method == method_t::PUT) {
    std::visit(
        [&](auto &&client_) {
          result = client_->Put(
              url_parsed.path, httplib_headers, body_str, content_type,
              [&](uint64_t current, uint64_t total) {
                if (req->progress_callback.has_value()) {
                  return req->progress_callback.value()(current, total);
                }
                return true;
              });
        },
        client);
  } else if (req->method == method_t::PATCH) {
    std::visit(
        [&](auto &&client_) {
          result = client_->Patch(
              url_parsed.path, httplib_headers, body_str, content_type,
              [&](uint64_t current, uint64_t total) {
                if (req->progress_callback.has_value()) {
                  return req->progress_callback.value()(current, total);
                }
                return true;
              });
        },
        client);
  } else if (req->method == method_t::DELETE) {
    std::visit(
        [&](auto &&client_) {
          result = client_->Delete(
              url_parsed.path, httplib_headers, body_str, content_type,
              [&](uint64_t current, uint64_t total) {
                if (req->progress_callback.has_value()) {
                  return req->progress_callback.value()(current, total);
                }
                return true;
              });
        },
        client);
  } else {
    return std::make_unique<response_t>(response_t{
        .status_code = 400,
        .headers = {},
        .body = "Unsupporting HTTP method",
    });
  }

  response_t response;
  if (result) {
    response.status_code = result->status;
    response.body = result->body;

    for (const auto &[key, value] : result->headers) {
      response.headers[key] = value;
    }
  } else {
    response.status_code = -1;
    response.body = "Request Failed";
    response.error = httplib::to_string(result.error());
  }

  return std::make_unique<response_t>(response);
}

#endif

} // namespace http
} // namespace ailoy