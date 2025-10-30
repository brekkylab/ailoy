#include <iostream>

#include <gtest/gtest.h>

#include "packet.hpp"

std::string txid = "1b22da6e-a0e3-405e-93ed-a2de78e45b66";

TEST(TestPacket, TestConnectPacket) {
  std::shared_ptr<ailoy::bytes_t> serialized =
      ailoy::dump_packet<ailoy::packet_type::connect>(txid);

  std::shared_ptr<ailoy::packet_t> packet = ailoy::load_packet(serialized);
  ASSERT_EQ(packet->ptype, ailoy::packet_type::connect);
  ASSERT_FALSE(packet->itype.has_value());
  ASSERT_TRUE(packet->headers);
  ASSERT_EQ(packet->headers->size(), 2);

  // First header field == tx_id
  ASSERT_TRUE(packet->headers->at(0)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(0)->as<ailoy::string_t>().operator*(), txid);

  // Second header field == version
  ASSERT_TRUE(packet->headers->at(1)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(1)->as<ailoy::string_t>().operator*(), "1");

  // std::cout << *serialized << std::endl;
}

TEST(TestPacket, TestSubscribeCallFunctionPacket) {
  auto opname = "foo";

  std::shared_ptr<ailoy::bytes_t> serialized =
      ailoy::dump_packet<ailoy::packet_type::subscribe,
                         ailoy::instruction_type::call_function>(txid, opname);
  std::shared_ptr<ailoy::packet_t> packet = ailoy::load_packet(serialized);
  ASSERT_EQ(packet->ptype, ailoy::packet_type::subscribe);
  ASSERT_TRUE(packet->itype.has_value());
  ASSERT_EQ(packet->itype.value(), ailoy::instruction_type::call_function);
  ASSERT_TRUE(packet->headers);
  ASSERT_EQ(packet->headers->size(), 2);

  // First header field == tx_id
  ASSERT_TRUE(packet->headers->at(0)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(0)->as<ailoy::string_t>().operator*(), txid);

  // Second header field == opname
  ASSERT_TRUE(packet->headers->at(1)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(1)->as<ailoy::string_t>().operator*(), opname);

  // std::cout << *serialized << std::endl;
}

TEST(TestPacket, TestUnsubscribeCallFunctionPacket) {
  auto opname = "foo";

  std::shared_ptr<ailoy::bytes_t> serialized =
      ailoy::dump_packet<ailoy::packet_type::unsubscribe,
                         ailoy::instruction_type::call_function>(txid, opname);
  std::shared_ptr<ailoy::packet_t> packet = ailoy::load_packet(serialized);
  ASSERT_EQ(packet->ptype, ailoy::packet_type::unsubscribe);
  ASSERT_TRUE(packet->itype.has_value());
  ASSERT_EQ(packet->itype.value(), ailoy::instruction_type::call_function);
  ASSERT_TRUE(packet->headers);
  ASSERT_EQ(packet->headers->size(), 2);

  // First header field == tx_id
  ASSERT_TRUE(packet->headers->at(0)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(0)->as<ailoy::string_t>().operator*(), txid);

  // Second header field == opname
  ASSERT_TRUE(packet->headers->at(1)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(1)->as<ailoy::string_t>().operator*(), opname);

  // std::cout << *serialized << std::endl;
}

TEST(TestPacket, TestExecuteCallFunctionPacket) {
  auto opname = "foo";
  auto opargs = ailoy::create<ailoy::map_t>();
  opargs->insert_or_assign("arg0", ailoy::create<ailoy::string_t>("arg0value"));
  opargs->insert_or_assign("arg1", ailoy::create<ailoy::uint_t>(100));
  opargs->insert_or_assign("arg2", ailoy::create<ailoy::array_t>());

  std::shared_ptr<ailoy::bytes_t> serialized =
      ailoy::dump_packet<ailoy::packet_type::execute,
                         ailoy::instruction_type::call_function>(txid, opname,
                                                                 opargs);
  std::shared_ptr<ailoy::packet_t> packet = ailoy::load_packet(serialized);
  ASSERT_EQ(packet->ptype, ailoy::packet_type::execute);
  ASSERT_TRUE(packet->itype.has_value());
  ASSERT_EQ(packet->itype.value(), ailoy::instruction_type::call_function);
  ASSERT_TRUE(packet->headers);
  ASSERT_EQ(packet->headers->size(), 2);

  // First header field == tx_id
  ASSERT_TRUE(packet->headers->at(0)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(0)->as<ailoy::string_t>().operator*(), txid);

  // Second header field == opname
  ASSERT_TRUE(packet->headers->at(1)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(1)->as<ailoy::string_t>().operator*(), opname);

  // Body == {"in": opargs}
  ASSERT_TRUE(packet->body->is_type_of<ailoy::map_t>());
  auto body = packet->body->as<ailoy::map_t>();
  ASSERT_TRUE(body->find("in") != body->end());
  ASSERT_TRUE(body->at("in")->is_type_of<ailoy::map_t>());
  auto body_in = body->at("in")->as<ailoy::map_t>();
  ASSERT_TRUE(body_in->find("arg0") != body_in->end());
  ASSERT_TRUE(body_in->find("arg1") != body_in->end());
  ASSERT_TRUE(body_in->find("arg2") != body_in->end());

  // std::cout << *serialized << std::endl;
}

TEST(TestPacket, TestSubscribeDefineComponentPacket) {
  auto comptype = "foobar";

  std::shared_ptr<ailoy::bytes_t> serialized =
      ailoy::dump_packet<ailoy::packet_type::subscribe,
                         ailoy::instruction_type::define_component>(txid,
                                                                    comptype);
  std::shared_ptr<ailoy::packet_t> packet = ailoy::load_packet(serialized);
  ASSERT_EQ(packet->ptype, ailoy::packet_type::subscribe);
  ASSERT_TRUE(packet->itype.has_value());
  ASSERT_EQ(packet->itype.value(), ailoy::instruction_type::define_component);
  ASSERT_TRUE(packet->headers);
  ASSERT_EQ(packet->headers->size(), 2);

  // First header field == tx_id
  ASSERT_TRUE(packet->headers->at(0)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(0)->as<ailoy::string_t>().operator*(), txid);

  // Second header field == comptype
  ASSERT_TRUE(packet->headers->at(1)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(1)->as<ailoy::string_t>().operator*(),
            comptype);

  // std::cout << *serialized << std::endl;
}

TEST(TestPacket, TestUnsubscribeDefineComponentPacket) {
  auto comptype = "foobar";

  std::shared_ptr<ailoy::bytes_t> serialized =
      ailoy::dump_packet<ailoy::packet_type::unsubscribe,
                         ailoy::instruction_type::define_component>(txid,
                                                                    comptype);
  std::shared_ptr<ailoy::packet_t> packet = ailoy::load_packet(serialized);
  ASSERT_EQ(packet->ptype, ailoy::packet_type::unsubscribe);
  ASSERT_TRUE(packet->itype.has_value());
  ASSERT_EQ(packet->itype.value(), ailoy::instruction_type::define_component);
  ASSERT_TRUE(packet->headers);
  ASSERT_EQ(packet->headers->size(), 2);

  // First header field == tx_id
  ASSERT_TRUE(packet->headers->at(0)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(0)->as<ailoy::string_t>().operator*(), txid);

  // Second header field == comptype
  ASSERT_TRUE(packet->headers->at(1)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(1)->as<ailoy::string_t>().operator*(),
            comptype);

  // std::cout << *serialized << std::endl;
}

TEST(TestPacket, TestExecuteDefineComponentPacket) {
  auto comptype = "foobar";
  auto compname = "foo";
  auto compargs = ailoy::create<ailoy::map_t>();
  compargs->insert_or_assign("arg0",
                             ailoy::create<ailoy::string_t>("arg0value"));

  std::shared_ptr<ailoy::bytes_t> serialized =
      ailoy::dump_packet<ailoy::packet_type::execute,
                         ailoy::instruction_type::define_component>(
          txid, comptype, compname, compargs);
  std::shared_ptr<ailoy::packet_t> packet = ailoy::load_packet(serialized);
  ASSERT_EQ(packet->ptype, ailoy::packet_type::execute);
  ASSERT_TRUE(packet->itype.has_value());
  ASSERT_EQ(packet->itype.value(), ailoy::instruction_type::define_component);
  ASSERT_TRUE(packet->headers);
  ASSERT_EQ(packet->headers->size(), 2);

  // First header field == tx_id
  ASSERT_TRUE(packet->headers->at(0)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(0)->as<ailoy::string_t>().operator*(), txid);

  // Second header field == comptype
  ASSERT_TRUE(packet->headers->at(1)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(1)->as<ailoy::string_t>().operator*(),
            comptype);

  // Body == {"name": compname, "in": compargs}
  ASSERT_TRUE(packet->body->is_type_of<ailoy::map_t>());
  auto body = packet->body->as<ailoy::map_t>();
  ASSERT_TRUE(body->find("name") != body->end());
  ASSERT_TRUE(body->at("name")->is_type_of<ailoy::string_t>());
  ASSERT_EQ(body->at<ailoy::string_t>("name").operator*(), compname);
  ASSERT_TRUE(body->find("in") != body->end());

  // std::cout << *serialized << std::endl;
}

TEST(TestPacket, TestRespondOkPacket) {
  std::shared_ptr<ailoy::bytes_t> serialized =
      ailoy::dump_packet<ailoy::packet_type::respond, true>(txid);
  std::shared_ptr<ailoy::packet_t> packet = ailoy::load_packet(serialized);
  ASSERT_EQ(packet->ptype, ailoy::packet_type::respond);
  ASSERT_FALSE(packet->itype.has_value());
  ASSERT_TRUE(packet->headers);
  ASSERT_EQ(packet->headers->size(), 1);

  // First header field == tx_id
  ASSERT_TRUE(packet->headers->at(0)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(0)->as<ailoy::string_t>().operator*(), txid);

  // Body == {"status": true}
  ASSERT_TRUE(packet->body->is_type_of<ailoy::map_t>());
  auto body = packet->body->as<ailoy::map_t>();
  ASSERT_TRUE(body->find("status") != body->end());
  ASSERT_TRUE(body->at("status")->is_type_of<ailoy::bool_t>());
  ASSERT_TRUE(body->at<ailoy::bool_t>("status").operator*());

  // std::cout << *serialized << std::endl;
}

TEST(TestPacket, TestRespondexecuteOkPacket) {
  std::shared_ptr<ailoy::map_t> out = ailoy::create<ailoy::map_t>();
  out->insert_or_assign("message",
                        ailoy::create<ailoy::string_t>("hello world"));
  std::shared_ptr<ailoy::bytes_t> serialized =
      ailoy::dump_packet<ailoy::packet_type::respond_execute, true>(txid, 0,
                                                                    true, out);
  std::shared_ptr<ailoy::packet_t> packet = ailoy::load_packet(serialized);
  ASSERT_EQ(packet->ptype, ailoy::packet_type::respond_execute);
  ASSERT_FALSE(packet->itype.has_value());
  ASSERT_TRUE(packet->headers);
  ASSERT_EQ(packet->headers->size(), 3);

  // First header field == tx_id
  ASSERT_TRUE(packet->headers->at(0)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(0)->as<ailoy::string_t>().operator*(), txid);

  // Second header field == sequence
  ASSERT_TRUE(packet->headers->at(1)->is_type_of<ailoy::uint_t>());
  ASSERT_EQ(packet->headers->at(1)->as<ailoy::uint_t>().operator*(), 0);

  // Third header field == finish
  ASSERT_TRUE(packet->headers->at(2)->is_type_of<ailoy::bool_t>());
  ASSERT_TRUE(packet->headers->at(2)->as<ailoy::bool_t>().operator*());

  // Body == {"status": true, "out": data}
  ASSERT_TRUE(packet->body->is_type_of<ailoy::map_t>());
  auto body = packet->body->as<ailoy::map_t>();
  ASSERT_TRUE(body->find("status") != body->end());
  ASSERT_TRUE(body->at("status")->is_type_of<ailoy::bool_t>());
  ASSERT_TRUE(body->at<ailoy::bool_t>("status").operator*());
  ASSERT_TRUE(body->find("out") != body->end());
  ASSERT_TRUE(body->at("out")->is_type_of<ailoy::map_t>());
  ASSERT_TRUE(body->at<ailoy::map_t>("out")->find("message") !=
              body->at<ailoy::map_t>("out")->end());
  ASSERT_TRUE(body->at<ailoy::map_t>("out")
                  ->at("message")
                  ->is_type_of<ailoy::string_t>());
  ASSERT_EQ(
      body->at<ailoy::map_t>("out")->at<ailoy::string_t>("message").operator*(),
      "hello world");

  // std::cout << *serialized << std::endl;
}

TEST(TestPacket, TestRespondErrorPacket) {
  auto reason = "This is the test";

  std::shared_ptr<ailoy::bytes_t> serialized =
      ailoy::dump_packet<ailoy::packet_type::respond, false>(txid, reason);
  std::shared_ptr<ailoy::packet_t> packet = ailoy::load_packet(serialized);
  ASSERT_EQ(packet->ptype, ailoy::packet_type::respond);
  ASSERT_FALSE(packet->itype.has_value());
  ASSERT_TRUE(packet->headers);
  ASSERT_EQ(packet->headers->size(), 1);

  // First header field == tx_id
  ASSERT_TRUE(packet->headers->at(0)->is_type_of<ailoy::string_t>());
  ASSERT_EQ(packet->headers->at(0)->as<ailoy::string_t>().operator*(), txid);

  // Body == {"status": false, "reason": reason}
  ASSERT_TRUE(packet->body->is_type_of<ailoy::map_t>());
  auto body = packet->body->as<ailoy::map_t>();
  ASSERT_TRUE(body->find("status") != body->end());
  ASSERT_TRUE(body->at("status")->is_type_of<ailoy::bool_t>());
  ASSERT_FALSE(body->at<ailoy::bool_t>("status").operator*());
  ASSERT_TRUE(body->find("reason") != body->end());
  ASSERT_TRUE(body->at("reason")->is_type_of<ailoy::string_t>());
  ASSERT_EQ(body->at<ailoy::string_t>("reason").operator*(), reason);

  // std::cout << *serialized << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
