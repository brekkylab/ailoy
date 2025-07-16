#include "http.hpp"

#include <iostream>

#ifdef EMSCRIPTEN
#include <emscripten.h>
#else
#include <gtest/gtest.h>
#endif

// Test functions
void test_get_request() {
  std::cout << "=== Testing GET Request ===" << std::endl;

  ailoy::http::request_t req;
  req.url = "https://jsonplaceholder.typicode.com/posts/1";
  req.method = ailoy::http::method_t::GET;
  req.headers["User-Agent"] = "EmscriptenTest/1.0";

  auto result = ailoy::http::request(req);

  if (!result) {
    std::cerr << result.error() << std::endl;
    return;
  }

  std::cout << "Status: " << result->status_code << std::endl;
  std::cout << "Body (first 200 chars): " << result->body.substr(0, 200)
            << std::endl;
  std::cout << "Headers:" << std::endl;
  for (const auto &[key, value] : result->headers) {
    std::cout << "  " << key << ": " << value << std::endl;
  }
  std::cout << std::endl;
}

void test_post_request() {
  std::cout << "=== Testing POST Request ===" << std::endl;

  ailoy::http::request_t req;
  req.url = "https://jsonplaceholder.typicode.com/posts";
  req.method = ailoy::http::method_t::POST;
  req.headers["Content-Type"] = "application/json";
  req.headers["User-Agent"] = "EmscriptenTest/1.0";
  req.body = R"({
        "title": "Test Post",
        "body": "This is a test post from Emscripten",
        "userId": 1
    })";

  auto result = ailoy::http::request(req);

  if (!result) {
    std::cerr << result.error() << std::endl;
  }

  std::cout << "Status: " << result->status_code << std::endl;
  std::cout << "Body: " << result->body << std::endl;
  std::cout << "Content-Type: " << result->headers["content-type"] << std::endl;
  std::cout << std::endl;
}

void test_headers() {
  std::cout << "=== Testing Headers ===" << std::endl;

  ailoy::http::request_t req;
  req.url = "https://httpbin.org/headers";
  req.method = ailoy::http::method_t::GET;
  req.headers["X-Custom-Header"] = "TestValue";
  req.headers["Authorization"] = "Bearer test-token";

  auto result = ailoy::http::request(req);

  if (!result) {
    std::cerr << result.error() << std::endl;
  }

  std::cout << "Status: " << result->status_code << std::endl;
  std::cout << "Response: " << result->body << std::endl;
  std::cout << std::endl;
}

void test_error_handling() {
  std::cout << "=== Testing Error Handling ===" << std::endl;

  ailoy::http::request_t req;
  req.url = "https://httpbin.org/status/404";
  req.method = ailoy::http::method_t::GET;

  auto result = ailoy::http::request(req);

  if (!result) {
    std::cerr << result.error() << std::endl;
  }

  std::cerr << "Status: " << result->status_code << std::endl;
  std::cerr << "Body: " << result->body << std::endl;
  std::cout << std::endl;
}

#ifdef EMSCRIPTEN

extern "C" {
EMSCRIPTEN_KEEPALIVE
void run_tests() {
  std::cout << "Starting HTTP tests..." << std::endl;

  test_get_request();
  test_post_request();
  test_headers();
  test_error_handling();

  std::cout << "All tests completed!" << std::endl;
}
}

#else

TEST(AiloyHTTPTest, GetRequest) { test_get_request(); }
TEST(AiloyHTTPTest, PostRequest) { test_post_request(); }
TEST(AiloyHTTPTest, Headers) { test_headers(); }
TEST(AiloyHTTPTest, ErrorHandling) { test_error_handling(); }

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif
