#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace ailoy {

namespace http {

enum class method_t {
  GET,
  POST,
  PUT,
  PATCH,
  DELETE_, // On Windows, DELETE is defined as a macro in `winnt.h`
  HEAD
};

// Copied from cpp-httplib
enum status_code_t {
  // Information responses
  Continue_100 = 100,
  SwitchingProtocol_101 = 101,
  Processing_102 = 102,
  EarlyHints_103 = 103,

  // Successful responses
  OK_200 = 200,
  Created_201 = 201,
  Accepted_202 = 202,
  NonAuthoritativeInformation_203 = 203,
  NoContent_204 = 204,
  ResetContent_205 = 205,
  PartialContent_206 = 206,
  MultiStatus_207 = 207,
  AlreadyReported_208 = 208,
  IMUsed_226 = 226,

  // Redirection messages
  MultipleChoices_300 = 300,
  MovedPermanently_301 = 301,
  Found_302 = 302,
  SeeOther_303 = 303,
  NotModified_304 = 304,
  UseProxy_305 = 305,
  unused_306 = 306,
  TemporaryRedirect_307 = 307,
  PermanentRedirect_308 = 308,

  // Client error responses
  BadRequest_400 = 400,
  Unauthorized_401 = 401,
  PaymentRequired_402 = 402,
  Forbidden_403 = 403,
  NotFound_404 = 404,
  MethodNotAllowed_405 = 405,
  NotAcceptable_406 = 406,
  ProxyAuthenticationRequired_407 = 407,
  RequestTimeout_408 = 408,
  Conflict_409 = 409,
  Gone_410 = 410,
  LengthRequired_411 = 411,
  PreconditionFailed_412 = 412,
  PayloadTooLarge_413 = 413,
  UriTooLong_414 = 414,
  UnsupportedMediaType_415 = 415,
  RangeNotSatisfiable_416 = 416,
  ExpectationFailed_417 = 417,
  ImATeapot_418 = 418,
  MisdirectedRequest_421 = 421,
  UnprocessableContent_422 = 422,
  Locked_423 = 423,
  FailedDependency_424 = 424,
  TooEarly_425 = 425,
  UpgradeRequired_426 = 426,
  PreconditionRequired_428 = 428,
  TooManyRequests_429 = 429,
  RequestHeaderFieldsTooLarge_431 = 431,
  UnavailableForLegalReasons_451 = 451,

  // Server error responses
  InternalServerError_500 = 500,
  NotImplemented_501 = 501,
  BadGateway_502 = 502,
  ServiceUnavailable_503 = 503,
  GatewayTimeout_504 = 504,
  HttpVersionNotSupported_505 = 505,
  VariantAlsoNegotiates_506 = 506,
  InsufficientStorage_507 = 507,
  LoopDetected_508 = 508,
  NotExtended_510 = 510,
  NetworkAuthenticationRequired_511 = 511,
};

using headers_t = std::unordered_map<std::string, std::string>;

using request_data_callback_t = std::function<bool(const char *, size_t)>;
using request_progress_callback_t = std::function<bool(uint64_t, uint64_t)>;

struct request_t {
  std::string url;
  method_t method;
  headers_t headers;
  std::optional<std::string> body = std::nullopt;
  std::optional<request_data_callback_t> data_callback = std::nullopt;
  std::optional<request_progress_callback_t> progress_callback = std::nullopt;
};

struct response_t {
  int status_code = -1;
  headers_t headers;
  std::string body;

  response_t() = default;
  response_t(const response_t &) = default;
  response_t &operator=(const response_t &) = default;
  response_t(response_t &&) = default;
  response_t &operator=(response_t &&) = default;
};

class result_t {
public:
  result_t() = default;
  result_t(std::unique_ptr<response_t> &&res,
           std::optional<std::string> error = std::nullopt)
      : res_(std::move(res)), error_(error) {}

  // response
  operator bool() const { return res_ != nullptr; }
  bool operator==(std::nullptr_t) const { return res_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return res_ != nullptr; }
  const response_t &value() const { return *res_; }
  response_t &value() { return *res_; }
  const response_t &operator*() const { return *res_; }
  response_t &operator*() { return *res_; }
  const response_t *operator->() const { return res_.get(); }
  response_t *operator->() { return res_.get(); }

  // error
  std::string error() const { return error_.value_or(""); }

private:
  std::unique_ptr<response_t> res_;
  std::optional<std::string> error_ = std::nullopt;
};

result_t request(const request_t &req);

} // namespace http

} // namespace ailoy