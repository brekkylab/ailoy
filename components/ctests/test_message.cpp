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
  msgs.push_back(message_t{role_t::user, "Are you existing?"});
  std::cout << msgs << std::endl;
}

TEST(TestMessage, TestChatTemplate) {
  messages_t msgs;
  msgs.push_back(message_t{role_t::system, "You are a helpful assistant."});
  msgs.push_back(message_t{role_t::user, "Hi what's your name?"});
  nlohmann::json v = msgs;

  ailoy_add_chat_template("qwen3", "hello {{messages[0].role}}");
  char *out = nullptr;
  ailoy_apply_chat_template("qwen3", v.dump().c_str(), &out);
  if (*out) {
    ASSERT_EQ(std::string(out), "hello system");
    free(out);
  } else {
    throw std::runtime_error("No output");
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
