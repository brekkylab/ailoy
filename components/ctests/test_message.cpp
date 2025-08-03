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

  ailoy_chat_template_t *tmpl;
  char *out = nullptr;
  ASSERT_EQ(ailoy_chat_template_create("hello {{messages[0].role}}", &tmpl), 0);
  ASSERT_EQ(ailoy_chat_template_apply(tmpl, v.dump().c_str(), &out), 0);
  if (*out) {
    ASSERT_EQ(std::string(out), "hello system");
    free(out);
  } else {
    throw std::runtime_error("No output");
  }
  ASSERT_EQ(ailoy_chat_template_destroy(tmpl), 0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
