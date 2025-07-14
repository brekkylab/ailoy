#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace ailoy {

namespace http {

enum class method_t { GET, POST, PUT, PATCH, DELETE };

using request_data_callback_t = std::function<bool(const char *, size_t)>;
using request_progress_callback_t = std::function<bool(uint64_t, uint64_t)>;

struct request_t {
  std::string url;
  method_t method;
  std::unordered_map<std::string, std::string> headers;
  std::optional<std::string> body = std::nullopt;
  std::optional<request_data_callback_t> data_callback = std::nullopt;
  std::optional<request_progress_callback_t> progress_callback = std::nullopt;
};

struct response_t {
  int status_code;
  std::unordered_map<std::string, std::string> headers;
  std::string body;
  std::optional<std::string> error;
};

std::unique_ptr<response_t> request(const std::unique_ptr<request_t> &req);

} // namespace http

} // namespace ailoy