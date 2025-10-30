#include <thread>

#include <gtest/gtest.h>

#include "inproc_socket.hpp"

using namespace std::chrono_literals;

bool connection_test_passed = false;

std::tuple<std::shared_ptr<ailoy::inproc::socket_t>,
           std::shared_ptr<ailoy::inproc::socket_t>>
connect(const ailoy::url_t &url) {
  std::shared_ptr<ailoy::inproc::socket_t> socket1;
  std::shared_ptr<ailoy::inproc::socket_t> socket2;
  std::thread t1{[url = url, &socket1] {
    auto monitor = ailoy::create<ailoy::monitor_t>();
    auto acceptor = ailoy::create<ailoy::inproc::acceptor_t>(url);
    acceptor->set_monitor(monitor);
    auto signal = monitor->monitor(1s);
    socket1 = acceptor->accept();
  }};
  std::this_thread::sleep_for(10ms);
  socket2 = ailoy::create<ailoy::inproc::socket_t>();
  bool dial_result = socket2->connect(url);
  bool connection_result = socket2->wait_until_attached();
  t1.join();
  return std::make_tuple(socket1, socket2);
}

TEST(AiloyInprocSocketTest, Connection) {
  std::shared_ptr<ailoy::inproc::socket_t> socket1;
  std::shared_ptr<ailoy::inproc::socket_t> socket2;
  const std::string url{"inproc://Connection"};

  std::thread t1{[url, &socket1] {
    auto monitor = ailoy::create<ailoy::monitor_t>();
    auto acceptor = ailoy::create<ailoy::inproc::acceptor_t>(url);
    acceptor->set_monitor(monitor);

    auto signal = monitor->monitor(1s);
    ASSERT_TRUE(signal.has_value());
    ASSERT_EQ(signal.value().who, acceptor->myname);
    ASSERT_EQ(signal.value().what, "accept");

    socket1 = acceptor->accept();
    ASSERT_TRUE(socket1);
  }};

  std::this_thread::sleep_for(10ms);
  socket2 = ailoy::create<ailoy::inproc::socket_t>();
  bool dial_result = socket2->connect(url);
  ASSERT_TRUE(dial_result);
  bool connection_result = socket2->wait_until_attached();
  ASSERT_TRUE(connection_result);

  ASSERT_EQ(socket1->peer_name, socket2->myname);
  ASSERT_EQ(socket2->peer_name, socket1->myname);
  t1.join();
}

TEST(AiloyInprocSocketTest, ConnectAndAccept) {
  std::shared_ptr<ailoy::inproc::socket_t> socket1;
  std::shared_ptr<ailoy::inproc::socket_t> socket2;
  const std::string url{"inproc://ConnectAndAccept"};

  std::thread t1{[url, &socket1] {
    auto monitor = ailoy::create<ailoy::monitor_t>();
    std::this_thread::sleep_for(10ms);
    auto acceptor = ailoy::create<ailoy::inproc::acceptor_t>(url);
    acceptor->set_monitor(monitor);

    auto signal = monitor->monitor(1s);
    ASSERT_TRUE(signal.has_value());
    ASSERT_EQ(signal.value().who, acceptor->myname);
    ASSERT_EQ(signal.value().what, "accept");

    socket1 = acceptor->accept();
    ASSERT_TRUE(socket1);
  }};

  socket2 = ailoy::create<ailoy::inproc::socket_t>();
  bool dial_result = socket2->connect(url);
  ASSERT_TRUE(dial_result);
  bool connection_result = socket2->wait_until_attached();
  ASSERT_TRUE(connection_result);

  ASSERT_EQ(socket1->peer_name, socket2->myname);
  ASSERT_EQ(socket2->peer_name, socket1->myname);
  t1.join();
}

TEST(AiloyInprocSocketTest, SimpleSendReceive) {
  auto [socket1, socket2] = connect("inproc://SimpleSendReceive");
  ASSERT_TRUE(socket1->send(ailoy::create<ailoy::bytes_t>("Hello world")));
  auto r1 = socket2->recv();
  ASSERT_TRUE(r1);
  ASSERT_EQ(r1->operator std::string(), "Hello world");

  ASSERT_TRUE(socket2->send(ailoy::create<ailoy::bytes_t>("World hello")));
  auto r2 = socket1->recv();
  ASSERT_TRUE(r2);
  ASSERT_EQ(r2->operator std::string(), "World hello");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
