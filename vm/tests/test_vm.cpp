#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "broker.hpp"
#include "broker_client.hpp"
#include "module.hpp"
#include "uuid.hpp"
#include "vm.hpp"

using namespace std::chrono_literals;

// execute / call_function / echo
static const uint8_t echo_run_bytes[] = {
    0x04, 0x00, 0x2C, 0x00, 0x82, 0x78, 0x24, 0x31, 0x62, 0x32, 0x32, 0x64,
    0x61, 0x36, 0x65, 0x2D, 0x61, 0x30, 0x65, 0x33, 0x2D, 0x34, 0x30, 0x35,
    0x65, 0x2D, 0x39, 0x33, 0x65, 0x64, 0x2D, 0x61, 0x32, 0x64, 0x65, 0x37,
    0x38, 0x65, 0x34, 0x35, 0x62, 0x36, 0x36, 0x64, 0x65, 0x63, 0x68, 0x6F,
    0x10, 0x00, 0x00, 0x00, 0xA1, 0x62, 0x69, 0x6E, 0x6B, 0x68, 0x65, 0x6C,
    0x6C, 0x6F, 0x20, 0x77, 0x6F, 0x72, 0x6C, 0x64};

// execute / call_function / spell
static const uint8_t spell_run_bytes[] = {
    0x04, 0x00, 0x2D, 0x00, 0x82, 0x78, 0x24, 0x31, 0x62, 0x32, 0x32, 0x64,
    0x61, 0x36, 0x65, 0x2D, 0x61, 0x30, 0x65, 0x33, 0x2D, 0x34, 0x30, 0x35,
    0x65, 0x2D, 0x39, 0x33, 0x65, 0x64, 0x2D, 0x61, 0x32, 0x64, 0x65, 0x37,
    0x38, 0x65, 0x34, 0x35, 0x62, 0x36, 0x36, 0x65, 0x73, 0x70, 0x65, 0x6C,
    0x6C, 0x10, 0x00, 0x00, 0x00, 0xA1, 0x62, 0x69, 0x6E, 0x6B, 0x68, 0x65,
    0x6C, 0x6C, 0x6F, 0x20, 0x77, 0x6F, 0x72, 0x6C, 0x64};

TEST(AiloyVMTest, Stoppable) {
  const std::string url = "inproc://stoppable";
  std::thread t_broker = std::thread([url] { ailoy::broker_start(url); });
  std::this_thread::sleep_for(100ms);
  std::shared_ptr<const ailoy::module_t> mods[] = {ailoy::get_default_module()};
  std::thread t_vm =
      std::thread([url, &mods] { ailoy::vm_start(url, mods, "default_vm"); });
  std::this_thread::sleep_for(100ms);

  ailoy::vm_stop(url);
  t_vm.join();
  ailoy::broker_stop(url);
  t_broker.join();
}

TEST(AiloyVMTest, Echo) {
  const std::string url = "inproc://echo";
  std::thread t_broker = std::thread([url] { ailoy::broker_start(url); });
  std::this_thread::sleep_for(100ms);
  std::shared_ptr<const ailoy::module_t> mods[] = {ailoy::get_default_module()};
  std::thread t_vm =
      std::thread([url, &mods] { ailoy::vm_start(url, mods, "default_vm"); });
  std::this_thread::sleep_for(100ms);

  std::shared_ptr<ailoy::bytes_t> pkt = ailoy::create<ailoy::bytes_t>(
      reinterpret_cast<const char *>(echo_run_bytes),
      reinterpret_cast<const char *>(echo_run_bytes + sizeof(echo_run_bytes)));

  auto client = ailoy::create<ailoy::broker_client_t>(url);

  client->send<ailoy::packet_type::connect>(ailoy::generate_uuid());
  ASSERT_TRUE(client->listen(1s));

  client->send_bytes(pkt);
  auto resp = client->listen(1s);
  ASSERT_TRUE(resp);

  // Second header field is sequence
  ASSERT_EQ(*resp->headers->at<ailoy::uint_t>(1), 0);

  // Third header field is finished
  ASSERT_TRUE(*resp->headers->at<ailoy::bool_t>(2));

  auto out = resp->body->as<ailoy::map_t>()->at<ailoy::string_t>("out");
  ASSERT_EQ(*out, "hello world");

  client->send<ailoy::packet_type::disconnect>(ailoy::generate_uuid());
  ASSERT_TRUE(client->listen(1s));

  ailoy::vm_stop(url);
  t_vm.join();
  ailoy::broker_stop(url);
  t_broker.join();
}

TEST(AiloyVMTest, Spell) {
  const std::string url = "inproc://spell";
  std::thread t_broker = std::thread([url] { ailoy::broker_start(url); });
  std::this_thread::sleep_for(100ms);
  std::shared_ptr<const ailoy::module_t> mods[] = {ailoy::get_default_module()};
  std::thread t_vm =
      std::thread([url, &mods] { ailoy::vm_start(url, mods, "default_vm"); });
  std::this_thread::sleep_for(100ms);

  std::shared_ptr<ailoy::bytes_t> pkt = ailoy::create<ailoy::bytes_t>(
      reinterpret_cast<const char *>(spell_run_bytes),
      reinterpret_cast<const char *>(spell_run_bytes +
                                     sizeof(spell_run_bytes)));

  auto client = ailoy::create<ailoy::broker_client_t>(url);

  client->send<ailoy::packet_type::connect>(ailoy::generate_uuid());
  ASSERT_TRUE(client->listen(1s));

  client->send_bytes(pkt);

  for (size_t i = 0; i < 11; i++) {
    auto resp = client->listen(1s);
    ASSERT_TRUE(resp);
    ASSERT_EQ(*resp->headers->at<ailoy::uint_t>(1), i);
    if (i < 10)
      ASSERT_FALSE(*resp->headers->at<ailoy::bool_t>(2));
    else
      ASSERT_TRUE(*resp->headers->at<ailoy::bool_t>(2));
    auto out = resp->body->as<ailoy::map_t>()->at<ailoy::string_t>("out");
    std::cout << *out << std::endl;
  }

  client->send<ailoy::packet_type::disconnect>(ailoy::generate_uuid());
  ASSERT_TRUE(client->listen(1s));

  ailoy::vm_stop(url);
  t_vm.join();
  ailoy::broker_stop(url);
  t_broker.join();
}

TEST(AiloyVMTest, Accumulator) {
  const std::string url = "inproc://accumulator";
  std::thread t_broker = std::thread([url] { ailoy::broker_start(url); });
  std::this_thread::sleep_for(100ms);
  std::shared_ptr<const ailoy::module_t> mods[] = {ailoy::get_default_module()};
  std::thread t_vm =
      std::thread([url, &mods] { ailoy::vm_start(url, mods, "default_vm"); });
  std::this_thread::sleep_for(100ms);

  auto client = ailoy::create<ailoy::broker_client_t>(url);

  client->send<ailoy::packet_type::connect>(ailoy::generate_uuid());
  ASSERT_TRUE(client->listen(1s));

  {
    auto in = ailoy::create<ailoy::string_t>("string0");
    client->send<ailoy::packet_type::execute,
                 ailoy::instruction_type::define_component>(
        ailoy::generate_uuid(), "accumulator", "acc0", in);
    auto resp = client->listen(1s);
    ASSERT_TRUE(resp);
    auto status = resp->body->as<ailoy::map_t>()->at<ailoy::bool_t>("status");
    if (*status) {
    } else {
      auto reason =
          resp->body->as<ailoy::map_t>()->at<ailoy::string_t>("reason");
      std::cout << *reason << std::endl;
      ASSERT_TRUE(false);
    }
  }

  {
    auto in = ailoy::create<ailoy::string_t>("string1");
    client->send<ailoy::packet_type::execute,
                 ailoy::instruction_type::call_method>(ailoy::generate_uuid(),
                                                       "acc0", "put", in);
    auto resp = client->listen(1s);
    ASSERT_TRUE(resp);
    auto status = resp->body->as<ailoy::map_t>()->at<ailoy::bool_t>("status");
    if (*status) {
    } else {
      auto reason =
          resp->body->as<ailoy::map_t>()->at<ailoy::string_t>("reason");
      std::cout << *reason << std::endl;
      ASSERT_TRUE(false);
    }
  }

  {
    client->send<ailoy::packet_type::execute,
                 ailoy::instruction_type::call_method>(ailoy::generate_uuid(),
                                                       "acc0", "get");
    auto resp = client->listen(1s);
    ASSERT_TRUE(resp);
    auto status = resp->body->as<ailoy::map_t>()->at<ailoy::bool_t>("status");
    if (*status) {
      auto out = resp->body->as<ailoy::map_t>()->at<ailoy::string_t>("out");
      ASSERT_EQ(*out, "string0string1");
    } else {
      auto reason =
          resp->body->as<ailoy::map_t>()->at<ailoy::string_t>("reason");
      std::cout << *reason << std::endl;
      ASSERT_TRUE(false);
    }
  }

  {
    client->send<ailoy::packet_type::execute,
                 ailoy::instruction_type::delete_component>(
        ailoy::generate_uuid(), "acc0");
    auto resp = client->listen(1s);
    ASSERT_TRUE(resp);
    auto status = resp->body->as<ailoy::map_t>()->at<ailoy::bool_t>("status");
    if (*status) {
    } else {
      auto reason =
          resp->body->as<ailoy::map_t>()->at<ailoy::string_t>("reason");
      std::cout << *reason << std::endl;
      ASSERT_TRUE(false);
    }
  }
  client->send<ailoy::packet_type::disconnect>(ailoy::generate_uuid());
  ASSERT_TRUE(client->listen(1s));

  ailoy::vm_stop(url);
  t_vm.join();
  ailoy::broker_stop(url);
  t_broker.join();
}

// // message as a simple string
// TEST(AiloyVMTest, HeartbeatEcho) {
//   const std::string url = "inproc://heartbeat-echo";
//   std::string channel = "heartbeat";
//   std::string message = "Hello VM!";

//   std::thread t_broker = std::thread([url] { ailoy::broker_start(url); });
//   std::this_thread::sleep_for(100ms);
//   std::thread t_vm = std::thread([url] { ailoy::vm_start(url); });
//   std::this_thread::sleep_for(100ms);
//   auto t_client = std::thread([&] {
//     auto client = ailoy::broker_client_t::create(url);
//     ASSERT_TRUE(client);
//     ASSERT_TRUE(client->subscribe(channel + "/resp"));
//     ASSERT_TRUE(client->send_to_channel(channel, message));
//     auto result = client->listen();
//     ASSERT_TRUE(result.has_value());
//     ASSERT_EQ(result.value().content, message);
//     ASSERT_TRUE(client->disconnect());
//   });

//   t_client.join();
//   ailoy::vm_stop(url);
//   t_vm.join();
//   ailoy::broker_stop(url);
//   t_broker.join();
// }

// TEST(AiloyVMTest, VM1Echo) {
//   const std::string url = "inproc://run-echo1";
//   std::string channel = "echo";

//   std::thread t_broker = std::thread([url] { ailoy::broker_start(url); });
//   std::this_thread::sleep_for(100ms);
//   std::thread t_vm = std::thread([url] { ailoy::vm_start(url); });
//   std::this_thread::sleep_for(100ms);
//   auto t_client = std::thread([&] {
//     auto client = ailoy::broker_client_t::create(url);
//     ASSERT_TRUE(client);
//     std::string tx_id = "tx/echo";
//     std::string message(echo_run_bytes,
//                         echo_run_bytes + sizeof(echo_run_bytes));
//     ASSERT_TRUE(client->send_to_channel(channel, message, tx_id));
//     auto result = client->listen();
//     ASSERT_TRUE(result.has_value());
//     ASSERT_EQ(
//         result.value().content,
//         std::string(echo_run_output_bytes,
//                     echo_run_output_bytes + sizeof(echo_run_output_bytes)));
//     ASSERT_TRUE(client->disconnect());
//   });

//   t_client.join();
//   ailoy::vm_stop(url);
//   t_vm.join();
//   ailoy::broker_stop(url);
//   t_broker.join();
// }

// TEST(AiloyVMTest, MultiVM) {
//   const std::string url = "inproc://multi-vm";
//   std::string vm1_name = "vm1";
//   std::string vm2_name = "vm2";
//   std::string channel = "heartbeat";
//   std::string message = "Hello VM!";

//   std::thread t_broker = std::thread([url] { ailoy::broker_start(url); });
//   std::this_thread::sleep_for(100ms);
//   std::thread t_vm1 =
//       std::thread([url, vm1_name] { ailoy::vm_start(url, vm1_name); });
//   std::thread t_vm2 =
//       std::thread([url, vm2_name] { ailoy::vm_start(url, vm2_name); });
//   std::this_thread::sleep_for(100ms);
//   auto t_client = std::thread([&] {
//     auto client = ailoy::broker_client_t::create(url);
//     ASSERT_TRUE(client);
//     ASSERT_TRUE(client->subscribe(channel + "/resp"));
//     ASSERT_TRUE(client->send_to_channel(channel, message));
//     auto result = client->listen();
//     ASSERT_TRUE(result.has_value());
//     ASSERT_EQ(result.value().content, message);
//     result = client->listen(10ms);
//     ASSERT_FALSE(result.has_value());
//     ASSERT_TRUE(client->disconnect());
//   });

//   t_client.join();
//   ailoy::vm_stop(url, vm1_name);
//   ailoy::vm_stop(url, vm2_name);
//   t_vm1.join();
//   t_vm2.join();
//   ailoy::broker_stop(url);
//   t_broker.join();
// }

// TEST(AiloyVMTest, RunEcho) {
//   const std::string url = "inproc://run-echo";
//   std::string channel = "echo";

//   std::thread t_broker = std::thread([url] { ailoy::broker_start(url); });
//   std::this_thread::sleep_for(100ms);
//   std::thread t_vm = std::thread([url] {
//     ailoy::vm_start2(url, {{"echo", std::make_shared<ailoy::echo>()},
//                            {"spell", std::make_shared<ailoy::spell>()}});
//   });
//   std::this_thread::sleep_for(100ms);
//   auto t_client = std::thread([&] {
//     auto client = ailoy::broker_client_t::create(url);
//     ASSERT_TRUE(client);
//     std::string tx_id = "tx@run/echo";
//     std::string message(echo_run_bytes,
//                         echo_run_bytes + sizeof(echo_run_bytes));
//     ASSERT_TRUE(client->run(channel, message, tx_id));
//     auto result = client->listen();
//     ASSERT_TRUE(result.has_value());
//     ASSERT_EQ(
//         result.value().content,
//         std::string(echo_run_output_bytes,
//                     echo_run_output_bytes + sizeof(echo_run_output_bytes)));
//     ASSERT_TRUE(client->disconnect());
//   });

//   t_client.join();
//   ailoy::vm_stop(url);
//   t_vm.join();
//   ailoy::broker_stop(url);
//   t_broker.join();
// }

// TEST(AiloyVMTest, RunSpell) {
//   const std::string url = "inproc://run-spell";
//   std::string channel = "spell";

//   std::thread t_broker = std::thread([url] { ailoy::broker_start(url); });
//   std::this_thread::sleep_for(100ms);
//   std::thread t_vm = std::thread([url] {
//     ailoy::vm_start2(url, {{"echo", std::make_shared<ailoy::echo>()},
//                            {"spell", std::make_shared<ailoy::spell>()}});
//   });
//   std::this_thread::sleep_for(100ms);
//   auto t_client = std::thread([&] {
//     auto client = ailoy::broker_client_t::create(url);
//     ASSERT_TRUE(client);
//     std::string tx_id = "tx/spell";
//     std::string message(spell_run_bytes,
//                         spell_run_bytes + sizeof(spell_run_bytes));
//     ASSERT_TRUE(client->run(channel, message, tx_id));

//     for (int i = 0; i < 256; i++) {
//       auto result_opt = client->listen();
//       ASSERT_TRUE(result_opt.has_value());

//       auto result = result_opt.value();
//       bool finish = result.extra != "false";
//       auto c =
//           *ailoy::deserialize(ailoy::create<ailoy::bytes_t>(result.content))
//                ->as<ailoy::map_t>()
//                ->at<ailoy::map_t>("outputs")
//                ->at<ailoy::uint_t>("c");

//       ASSERT_EQ((unsigned char)c,
//                 spell_run_bytes[SPELL_MESSAGE_BEGIN_INDEX + i]);
//       ASSERT_LT(i, SPELL_MESSAGE_SIZE); // check run.finish working

//       if (finish)
//         break;
//     }

//     ASSERT_TRUE(client->disconnect());
//   });

//   t_client.join();
//   ailoy::vm_stop(url);
//   t_vm.join();
//   ailoy::broker_stop(url);
//   t_broker.join();
// }

// TEST(AiloyVMTest, Accumulator) {
//   const std::string url = "inproc://let-accumulator";
//   std::string component_name = "accu0";
//   std::string component_type = "accumulator";
//   std::string base_str = "BASE";

//   std::thread t_broker = std::thread([url] { ailoy::broker_start(url); });
//   std::this_thread::sleep_for(100ms);
//   std::thread t_vm = std::thread([url] {
//     ailoy::vm_start2(
//         url,
//         {
//             {"accumulator_put", std::make_shared<ailoy::accumulator_put>()},
//             {"accumulator_get", std::make_shared<ailoy::accumulator_get>()},
//             {"accumulator_count",
//             std::make_shared<ailoy::accumulator_count>()},
//         },
//         {
//             {"accumulator", std::make_shared<ailoy::accumulator_type_t>()},
//         });
//   });
//   std::this_thread::sleep_for(100ms);
//   auto t_client = std::thread([&] {
//     auto client = ailoy::broker_client_t::create(url);
//     ASSERT_TRUE(client);

//     std::string tx_id = "tx/accumulator";
//     auto let_request = ailoy::create<ailoy::map_t>();
//     auto attrs = ailoy::create<ailoy::map_t>();
//     auto base = ailoy::create<ailoy::string_t>(base_str);
//     attrs->insert_or_assign("base", base);
//     let_request->insert_or_assign(
//         "name", ailoy::create<ailoy::string_t>(component_name));
//     let_request->insert_or_assign(
//         "type", ailoy::create<ailoy::string_t>(component_type));
//     let_request->insert_or_assign("attrs", attrs);

//     ASSERT_TRUE(client->let(component_name, component_type,
//                             *let_request->serialize(), tx_id));
//     auto result = client->listen();
//     ASSERT_TRUE(result.has_value());

//     std::string tx_id0 = "tx/accumulator_get_0";
//     auto run_request0 = ailoy::create<ailoy::map_t>();
//     run_request0->insert_or_assign(
//         "callable", ailoy::create<ailoy::string_t>("accumulator_get"));
//     run_request0->insert_or_assign("inputs", ailoy::create<ailoy::map_t>());
//     run_request0->insert_or_assign("runOptions",
//     ailoy::create<ailoy::map_t>());
//     ASSERT_TRUE(client->run("accumulator_get", *run_request0->serialize(),
//                             tx_id0, component_name));
//     auto result0 = client->listen();
//     ASSERT_TRUE(result0.has_value());
//     auto result_val0 = ailoy::deserialize(
//         ailoy::create<ailoy::bytes_t>(result0.value().content));
//     auto message0 = result_val0->as<ailoy::map_t>()
//                         ->at<ailoy::map_t>("outputs")
//                         ->at<ailoy::string_t>("message");
//     ASSERT_EQ(*message0, base_str);

//     std::string tx_id1 = "tx/accumulator_put_1";
//     auto run_request1 = ailoy::create<ailoy::map_t>();
//     run_request1->insert_or_assign(
//         "callable", ailoy::create<ailoy::string_t>("accumulator_put"));
//     auto inputs1 = ailoy::create<ailoy::map_t>();
//     inputs1->insert_or_assign("s", ailoy::create<ailoy::string_t>("-AAA"));
//     run_request1->insert_or_assign("inputs", inputs1);
//     run_request1->insert_or_assign("runOptions",
//     ailoy::create<ailoy::map_t>());
//     ASSERT_TRUE(client->run("accumulator_put", *run_request1->serialize(),
//                             tx_id1, component_name));
//     auto result1 = client->listen();
//     ASSERT_TRUE(result.has_value());

//     std::string tx_id2 = "tx/accumulator_put_2";
//     auto run_request2 = ailoy::create<ailoy::map_t>();
//     run_request2->insert_or_assign(
//         "callable", ailoy::create<ailoy::string_t>("accumulator_put"));
//     auto inputs2 = ailoy::create<ailoy::map_t>();
//     inputs2->insert_or_assign("s", ailoy::create<ailoy::string_t>("-bbb"));
//     run_request2->insert_or_assign("inputs", inputs2);
//     run_request2->insert_or_assign("runOptions",
//     ailoy::create<ailoy::map_t>());
//     ASSERT_TRUE(client->run("accumulator_put", *run_request2->serialize(),
//                             tx_id2, component_name));
//     auto result2 = client->listen();
//     ASSERT_TRUE(result.has_value());

//     std::string tx_id3 = "tx/accumulator_get_3";
//     auto run_request3 = ailoy::create<ailoy::map_t>();
//     run_request3->insert_or_assign(
//         "callable", ailoy::create<ailoy::string_t>("accumulator_get"));
//     run_request3->insert_or_assign("inputs", ailoy::create<ailoy::map_t>());
//     run_request3->insert_or_assign("runOptions",
//     ailoy::create<ailoy::map_t>());
//     ASSERT_TRUE(client->run("accumulator_get", *run_request3->serialize(),
//                             tx_id3, component_name));
//     auto result3 = client->listen();
//     ASSERT_TRUE(result3.has_value());
//     auto result_val3 = ailoy::deserialize(
//         ailoy::create<ailoy::bytes_t>(result3.value().content));
//     auto message3 = result_val3->as<ailoy::map_t>()
//                         ->at<ailoy::map_t>("outputs")
//                         ->at<ailoy::string_t>("message");
//     ASSERT_EQ(*message3, base_str + "-AAA" + "-bbb");

//     std::string tx_id4 = "tx/accumulator_count_4";
//     auto run_request4 = ailoy::create<ailoy::map_t>();
//     run_request4->insert_or_assign(
//         "callable", ailoy::create<ailoy::string_t>("accumulator_count"));
//     run_request4->insert_or_assign("inputs", ailoy::create<ailoy::map_t>());
//     run_request4->insert_or_assign("runOptions",
//     ailoy::create<ailoy::map_t>());
//     ASSERT_TRUE(client->run("accumulator_count", *run_request4->serialize(),
//                             tx_id4, component_name));
//     auto result4 = client->listen();
//     ASSERT_TRUE(result4.has_value());
//     auto result_val4 = ailoy::deserialize(
//         ailoy::create<ailoy::bytes_t>(result4.value().content));
//     auto count = result_val4->as<ailoy::map_t>()
//                      ->at<ailoy::map_t>("outputs")
//                      ->at<ailoy::uint_t>("count");
//     ASSERT_EQ(static_cast<int>(*count), 2);

//     ASSERT_TRUE(client->disconnect());
//   });

//   t_client.join();
//   ailoy::vm_stop(url);
//   t_vm.join();
//   ailoy::broker_stop(url);
//   t_broker.join();
// }

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}