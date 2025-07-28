#include <thread>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "chat_template.hpp"
#include "message.hpp"

using namespace ailoy;

TEST(TestMessage, TestMessage) {
  messages_t msgs;
  msgs.push_back(message_t{role_t::system, "You are a helpful assistant."});
  msgs.push_back(message_t{role_t::user, "Hi what's your name?"});
  msgs.push_back(message_t{role_t::assistant, content_category_t::reasoning,
                           "Thinking about what is my name..."});
  msgs.push_back(message_t{role_t::assistant, "You can call me Jaden."});
  msgs.push_back(message_t{role_t::user, "Are you exist?"});
  std::cout << msgs << std::endl;
}

TEST(TestMessage, TestChatTemplate) {
  ailoy_add_chat_template("qwen3", "hello {{v}}");
  char *tmpl;
  ailoy_get_chat_template("qwen3", &tmpl);
  std::cout << tmpl << std::endl;
  free(tmpl);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
