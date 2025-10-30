#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "broker.hpp"
#include "broker_client.hpp"
#include "uuid.hpp"

using namespace std::chrono_literals;

static bool stoppable_passed = false;
static bool connect_and_disconnect_passed = false;

TEST(AiloyBrokerTest, Stoppable) {
  std::thread t =
      std::thread([] { ailoy::broker_start("inproc://stoppable"); });
  std::this_thread::sleep_for(100ms);
  ailoy::broker_stop("inproc://stoppable");
  t.join();
  stoppable_passed = true;
}

TEST(AiloyBrokerTest, ConnectAndDisconnect) {
  if (!stoppable_passed)
    GTEST_SKIP() << "skipping because AiloyBrokerTest.Stoppable did not pass";

  std::string url = "inproc://connect_and_disconnect";
  std::thread tserver = std::thread([url] {
    auto remaining_connection = ailoy::broker_start(url);
    ASSERT_EQ(remaining_connection, 0);
  });
  auto tclient = std::thread([url]() {
    std::this_thread::sleep_for(10ms);
    auto client = ailoy::create<ailoy::broker_client_t>(url);
    ASSERT_TRUE(client);

    // Send connect packet
    {
      auto uuid = ailoy::generate_uuid();
      ASSERT_TRUE(client->send<ailoy::packet_type::connect>(uuid));
      auto resp = client->listen(100ms);
      ASSERT_TRUE(resp);
      ASSERT_EQ(resp->ptype, ailoy::packet_type::respond);
      ASSERT_EQ(*(resp->headers->at(0)->as<ailoy::string_t>()), uuid);
    }
    {
      auto uuid = ailoy::generate_uuid();
      ASSERT_TRUE(client->send<ailoy::packet_type::disconnect>(uuid));
      auto resp = client->listen(100ms);
      ASSERT_TRUE(resp);
      ASSERT_EQ(resp->ptype, ailoy::packet_type::respond);
      ASSERT_EQ(*(resp->headers->at(0)->as<ailoy::string_t>()), uuid);
    }
  });

  tclient.join();
  ailoy::broker_stop(url);
  tserver.join();

  connect_and_disconnect_passed = true;
}

void connect(std::shared_ptr<ailoy::broker_client_t> client) {
  auto uuid = ailoy::generate_uuid();
  client->send<ailoy::packet_type::connect>(uuid);
  client->listen(100ms);
}

void disconnect(std::shared_ptr<ailoy::broker_client_t> client) {
  auto uuid = ailoy::generate_uuid();
  client->send<ailoy::packet_type::disconnect>(uuid);
  client->listen(100ms);
}

TEST(AiloyBrokerTest, LoopbackClient) {
  if (!connect_and_disconnect_passed)
    GTEST_SKIP()
        << "skipping because AiloyBrokerTest.ConnectAndDisconnect did not pass";

  std::string url = "inproc://loopback_client";
  std::thread tserver = std::thread([url] { ailoy::broker_start(url); });
  std::this_thread::sleep_for(10ms);
  auto client = ailoy::create<ailoy::broker_client_t>(url);
  connect(client);
  {
    auto tx_id = ailoy::generate_uuid();
    auto sent =
        client->send<ailoy::packet_type::subscribe,
                     ailoy::instruction_type::call_function>(tx_id, "echo");
    ASSERT_TRUE(sent);
    auto received = client->listen(1s);
    ASSERT_TRUE(received);
  }
  {
    auto tx_id = ailoy::generate_uuid();
    // Send execute
    auto sent1 = client->send<ailoy::packet_type::execute,
                              ailoy::instruction_type::call_function>(
        tx_id, "echo", ailoy::create<ailoy::string_t>("Hello world"));
    ASSERT_TRUE(sent1);

    // Receive execute, since it is loopback
    auto received1 = client->listen(1s);
    ASSERT_TRUE(received1);
    ASSERT_EQ(received1->ptype, ailoy::packet_type::execute);
    ASSERT_TRUE(received1->itype.has_value());
    ASSERT_EQ(received1->itype.value(), ailoy::instruction_type::call_function);
    ASSERT_TRUE(received1->body);
    ASSERT_TRUE(received1->body->find("in") != received1->body->end());
    ASSERT_TRUE(received1->body->at("in")->is_type_of<ailoy::string_t>());
    auto value1 = received1->body->at<ailoy::string_t>("in");
    ASSERT_EQ(*value1, "Hello world");

    // Send response
    auto sent2 = client->send<ailoy::packet_type::respond_execute, true>(
        tx_id, 0, true, value1);
    ASSERT_TRUE(sent2);

    // Receive result
    auto received2 = client->listen(1s);
    ASSERT_TRUE(received2);
    ASSERT_EQ(received2->ptype, ailoy::packet_type::respond_execute);
    ASSERT_TRUE(received2->body);
    ASSERT_TRUE(received2->body->find("out") != received2->body->end());
    ASSERT_TRUE(received2->body->at("out")->is_type_of<ailoy::string_t>());
    auto value2 = received2->body->at<ailoy::string_t>("out");
    ASSERT_EQ(*value2, "Hello world");
  }
  {
    auto tx_id = ailoy::generate_uuid();
    auto sent =
        client->send<ailoy::packet_type::unsubscribe,
                     ailoy::instruction_type::call_function>(tx_id, "echo");
    ASSERT_TRUE(sent);
    auto received = client->listen(1s);
    ASSERT_TRUE(received);
  }
  disconnect(client);
  ailoy::broker_stop(url);
  tserver.join();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}