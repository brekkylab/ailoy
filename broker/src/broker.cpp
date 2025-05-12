#include <unordered_map>

#include <magic_enum/magic_enum.hpp>

#include "broker.hpp"
#include "exception.hpp"
#include "inproc_socket.hpp"
#include "logging.hpp"
#include "packet.hpp"

using namespace std::chrono_literals;

namespace ailoy {

/**
 * client_id -> socket
 */
using socket_map_t =
    std::unordered_map<std::string, std::shared_ptr<inproc::socket_t>>;

/**
 * channel -> socket
 */
using subscription_map_t =
    std::unordered_map<channel_t, std::shared_ptr<inproc::socket_t>>;

/**
 * tx_id -> socket
 */
using transaction_map_t =
    std::unordered_map<tx_id_t, std::shared_ptr<inproc::socket_t>>;

constexpr size_t NUM_WORKERS = 2;

static std::unordered_map<url_t, std::shared_ptr<stop_t>> stops;

size_t broker_start(const std::string &url) {
  std::shared_ptr<monitor_t> monitor = ailoy::create<monitor_t>();

  socket_map_t sockets;
  subscription_map_t subscriptions;
  transaction_map_t transactions;

  auto acceptor = ailoy::create<inproc::acceptor_t>(url);
  if (!acceptor)
    throw ailoy::exception("URL already occupied");
  acceptor->set_monitor(monitor);

  // Stop signals
  std::shared_ptr<stop_t> stop = std::make_shared<stop_t>();
  stops.insert_or_assign(url, stop);
  stop_t::global_stop.set_monitor(monitor);
  stop->set_monitor(monitor);

  while (true) {
    auto signal_opt = monitor->monitor(100ms);
    if (!signal_opt.has_value())
      continue;
    auto signal = signal_opt.value();

    if (signal.what == "stop")
      break;
    else if (signal.what == "accept") {
      auto socket = acceptor->accept();
      if (!socket)
        continue;
      socket->set_monitor(monitor);
      sockets.insert_or_assign(socket->myname, socket);
    } else if (signal.what == "recv") {
      if (sockets.find(signal.who) == sockets.end())
        continue;
      auto socket = sockets[signal.who];
      auto msg = socket->recv();
      if (!msg)
        continue;
      // Load packet without parsing body, header only
      auto pkt = load_packet(msg, true);
      auto tx_id = pkt->get_tx_id();

      debug("[Broker] packet received: {}", pkt->operator std::string());

      // Handle
      switch (pkt->ptype) {
      case packet_type::connect:
        socket->send(dump_packet<packet_type::respond, true>(tx_id));
        break;
      case packet_type::disconnect: {
        std::vector<channel_t> channels_to_be_deleted;
        for (const auto [ch, sock] : subscriptions)
          if (sock->myname == socket->myname)
            channels_to_be_deleted.push_back(ch);
        for (const auto ch : channels_to_be_deleted)
          subscriptions.erase(ch);
        sockets.erase(socket->myname);
        socket->send(dump_packet<packet_type::respond, true>(tx_id));
        break;
      }
      case packet_type::subscribe: {
        auto channel = pkt->get_channel();
        if (subscriptions.find(channel) != subscriptions.end()) {
          socket->send(dump_packet<packet_type::respond, false>(
              tx_id, "Subscription already occupied by {}"));
          break;
        }
        subscriptions.insert_or_assign(channel, socket);
        socket->send(dump_packet<packet_type::respond, true>(tx_id));
        break;
      }
      case packet_type::unsubscribe: {
        auto channel = pkt->get_channel();
        auto sub_it = subscriptions.find(channel);
        if (sub_it == subscriptions.end()) {
          socket->send(dump_packet<packet_type::respond, false>(
              tx_id, "Subscription not exists {}"));
          break;
        }
        if (sub_it->second->myname != socket->myname) {
          socket->send(dump_packet<packet_type::respond, false>(
              tx_id, "Trying to remove subscription made by other node"));
          break;
        }
        subscriptions.erase(sub_it);
        socket->send(dump_packet<packet_type::respond, true>(tx_id));
        break;
      }
      case packet_type::execute: {
        auto channel = pkt->get_channel();
        auto target_it = subscriptions.find(channel);
        if (target_it == subscriptions.end()) {
          socket->send(dump_packet<packet_type::respond_execute, false>(
              tx_id, 0, "There is no channel can handle this request"));
          break;
        }
        transactions.insert_or_assign(tx_id, socket);
        target_it->second->send(msg);
        break;
      }
      case packet_type::respond_execute: {
        auto target_it = transactions.find(tx_id);
        if (target_it == transactions.end()) {
          warn("[Broker] Transaction id vanished, ignored: {}", tx_id);
          break;
        }
        target_it->second->send(msg);

        if (*(pkt->headers->at<bool_t>(2)) == true) {
          transactions.erase(target_it);
        }
        break;
      }
      default:
        error("[Broker] There is no handler for packet");
        break;
      }
    } else {
      error("[Broker] Unknown signal type: {} (by {})", signal.what,
            signal.who);
    }
  }

  if (sockets.size() > 0) {
    warn("[Broker] Remaining connection exists: ", sockets.size());
  }
  return sockets.size();
}

void broker_stop(const std::string &url) {
  auto it = stops.find(url);
  if (it == stops.end())
    return;
  it->second->stop();
}

} // namespace ailoy

extern "C" {

size_t ailoy_broker_start(const char *url) { return ailoy::broker_start(url); }

void ailoy_broker_stop(const char *url) { ailoy::broker_stop(url); }
}
