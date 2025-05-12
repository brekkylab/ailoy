#include <gtest/gtest.h>
#include <httplib.h>

#include <iostream>

#include "language.hpp"
#include "module.hpp"

std::string get_file_content(const std::string &url, const std::string &path) {
  httplib::Client cli(url);
  std::string rv;

  if (auto res = cli.Get(path)) {
    if (res->status == httplib::StatusCode::OK_200)
      rv = res->body;
  } else {
    std::cout << "HTTP error: " << httplib::to_string(res.error()) << std::endl;
  }

  return rv;
}

TEST(TextSplitterTest, TestSplitBySeparator) {

  auto language_module = ailoy::get_language_module();
  auto split_text_by_separator =
      language_module->ops.at("split_text_by_separator");
  auto in = ailoy::create<ailoy::map_t>();

  auto input_str = get_file_content("https://raw.githubusercontent.com",
                                    "/hwchase17/chat-your-data/refs/heads/"
                                    "master/state_of_the_union.txt");

  in->insert_or_assign("text", ailoy::create<ailoy::string_t>(input_str));
  in->insert_or_assign("chunk_size", ailoy::create<ailoy::uint_t>(500));
  in->insert_or_assign("chunk_overlap", ailoy::create<ailoy::uint_t>(200));
  auto v = split_text_by_separator->initialize(in);
  if (v.has_value()) {
    std::cerr << v.value().reason << std::endl;
    ASSERT_FALSE(true);
  }
  auto chunks = std::get<0>(split_text_by_separator->step())
                    .val->as<ailoy::map_t>()
                    ->at<ailoy::array_t>("chunks");
  for (auto c : *chunks) {
    // if a chunk contains >500 characters
    // without separator, this can fail.
    ASSERT_GE(500, c->as<ailoy::string_t>()->size());

    // For printing
    // std::cout << "===(" << c->as<ailoy::string_t>()->size() << ")" <<
    // std::endl
    //           << c << std::endl
    //           << "===" << std::endl;
  }
}

TEST(TextSplitterTest, TestSplitBySeparatorRecursively) {

  auto language_module = ailoy::get_language_module();
  auto split_text_by_separator = language_module->ops.at("split_text");
  auto in = ailoy::create<ailoy::map_t>();

  auto input_str = get_file_content("https://raw.githubusercontent.com",
                                    "/hwchase17/chat-your-data/refs/heads/"
                                    "master/state_of_the_union.txt");
  in->insert_or_assign("text", ailoy::create<ailoy::string_t>(input_str));
  in->insert_or_assign("chunk_size", ailoy::create<ailoy::uint_t>(500));
  in->insert_or_assign("chunk_overlap", ailoy::create<ailoy::uint_t>(200));
  split_text_by_separator->initialize(in);
  auto chunks = std::get<0>(split_text_by_separator->step())
                    .val->as<ailoy::map_t>()
                    ->at<ailoy::array_t>("chunks");
  for (auto c : *chunks) {
    // if a chunk contains >500 characters without separator, this can fail.
    ASSERT_GE(500, c->as<ailoy::string_t>()->size());

    // For printing
    // std::cout << "===(" << c.size() << ")" << std::endl
    //           << c << std::endl
    //           << "===" << std::endl;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
