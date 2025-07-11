#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace ailoy {

namespace http {

enum class method_t { GET, PUT, POST, DELETE };

struct request_t {
  std::string url;
  method_t method;
  std::unordered_map<std::string, std::string> headers;
  std::optional<std::string> body = std::nullopt;
};

struct response_t {
  int status_code;
  std::unordered_map<std::string, std::string> headers;
  std::string body;
};

response_t request(const request_t &req);

} // namespace http

} // namespace ailoy