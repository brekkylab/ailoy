#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "module.hpp"

TEST(HttpRequestTest, Get_Frankfurter) {
  auto default_operators = ailoy::get_default_module()->ops;
  auto http_request_op = default_operators.at("http_request");

  auto input = ailoy::create<ailoy::map_t>();
  input->insert_or_assign(
      "url", ailoy::create<ailoy::string_t>(
                 "https://api.frankfurter.dev/v1/latest?base=USD&symbols=KRW"));
  input->insert_or_assign("method", ailoy::create<ailoy::string_t>("GET"));

  http_request_op->initialize(input);
  auto output_opt = http_request_op->step();
  ASSERT_EQ(output_opt.index(), 0);

  auto output = std::get<0>(output_opt).val->as<ailoy::map_t>();
  ASSERT_EQ(*output->at<ailoy::uint_t>("status_code"), 200);

  auto j = nlohmann::json::parse(*output->at<ailoy::bytes_t>("body"));
  // body is like:
  // {"amount":1.0,"base":"USD","date":"2025-04-17","rates":{"KRW":1416.48}}
  ASSERT_EQ(j["amount"], 1.0);
  ASSERT_EQ(j["base"], "USD");
  ASSERT_EQ(j["rates"].contains("KRW"), true);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
