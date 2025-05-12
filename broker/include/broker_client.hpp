#pragma once

#include "inproc_socket.hpp"
#include "packet.hpp"

namespace ailoy {

/**
 * @brief A synchronous broker client used to send/receive packets over an
 * in-process socket
 * @details
 * This class abstracts packet-based communication for AI component
 * orchestration. It supports sending and listening packets
 */
class broker_client_t {
public:
  /**
   * @brief Constructor with an internal monitor
   * @param url Connection target
   * @throws ailoy::exception if connection fails
   */
  broker_client_t(const url_t &url)
      : socket_(ailoy::create<inproc::socket_t>()),
        monitor_(ailoy::create<monitor_t>()), external_monitor_(false) {
    if (!socket_->connect(url))
      throw ailoy::exception("Connection failed");
    socket_->set_monitor(monitor_);
  }

  /**
   * @brief Constructor with externally provided monitor
   * @param url Connection target
   * @param monitor Monitor instance to use
   * @throws ailoy::exception if connection fails
   */
  broker_client_t(const url_t &url, std::shared_ptr<monitor_t> monitor)
      : socket_(ailoy::create<inproc::socket_t>()), monitor_(monitor),
        external_monitor_(true) {
    if (!socket_->connect(url))
      throw ailoy::exception("Connection failed");
    socket_->set_monitor(monitor_);
  }

  /**
   * @brief Sends a raw byte packet
   * @param packet A shared pointer to the byte buffer
   * @return true if the send succeeds
   */
  bool send_bytes(std::shared_ptr<bytes_t> packet) const;

  /**
   * @brief Receives a raw byte packet
   * @return The received packet as bytes
   */
  std::shared_ptr<bytes_t> recv_bytes();

  /**
   * @brief Sends a packet of types: connect or disconnect
   * @tparam ptype Packet type (must be connect or disconnect)
   * @param args Arguments for the packet constructor
   * @return true if the send succeeds
   */
  template <packet_type ptype, typename... args_t>
    requires(ptype == packet_type::connect || ptype == packet_type::disconnect)
  bool send(args_t... args) const {
    return send_bytes(dump_packet<ptype>(args...));
  }

  /**
   * @brief Sends a packet of types: subscribe, unsubscribe, or execute
   * @tparam ptype Packet type
   * @tparam itype Instruction type
   * @param args Arguments for the packet constructor
   * @return true if the send succeeds
   */
  template <packet_type ptype, instruction_type itype, typename... args_t>
    requires(ptype == packet_type::subscribe ||
             ptype == packet_type::unsubscribe || ptype == packet_type::execute)
  bool send(args_t... args) const {
    return send_bytes(dump_packet<ptype, itype>(args...));
  }

  /**
   * @brief Sends a response packet
   * @tparam ptype Packet type (must be respond or respond_execute)
   * @tparam status Status flag (true for success, false for failure)
   * @param args Arguments for the packet constructor
   * @return true if the send succeeds
   */
  template <packet_type ptype, bool status, typename... args_t>
    requires(ptype == packet_type::respond ||
             ptype == packet_type::respond_execute)
  bool send(args_t... args) const {
    return send_bytes(dump_packet<ptype, status>(args...));
  }

  /**
   * @brief Receives and parses the next packet
   * @param skip_body Whether to skip deserializing the packet body
   * @return Parsed packet object
   */
  std::shared_ptr<packet_t> recv(bool skip_body = false);

  /**
   * @brief Listens until a packet is received or timeout is reached
   * @param timeout Absolute timeout time point
   * @param skip_body Whether to skip parsing the body
   * @return Received packet or nullptr on timeout
   */
  std::shared_ptr<packet_t> listen(time_point_t timeout,
                                   bool skip_body = false);

  /**
   * @brief Listens with a relative timeout duration
   * @param due Timeout duration from now
   * @param skip_body Whether to skip parsing the body
   * @return Received packet or nullptr on timeout
   */
  std::shared_ptr<packet_t> listen(duration_t due, bool skip_body = false);

private:
  std::shared_ptr<inproc::socket_t> socket_;

  std::shared_ptr<monitor_t> monitor_;

  const bool external_monitor_;
};

} // namespace ailoy
