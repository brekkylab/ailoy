#include "broker_client.hpp"

#include "exception.hpp"

using namespace std::chrono_literals;

namespace ailoy {

bool broker_client_t::send_bytes(std::shared_ptr<bytes_t> packet) const {
  if (!packet)
    return false;
  return socket_->send(packet);
}

std::shared_ptr<bytes_t> broker_client_t::recv_bytes() {
  return socket_->recv();
}

std::shared_ptr<packet_t> broker_client_t::recv(bool skip_body) {
  auto payload = socket_->recv();
  if (!payload)
    return nullptr;
  return load_packet(payload, skip_body);
}

std::shared_ptr<packet_t> broker_client_t::listen(time_point_t timeout,
                                                  bool skip_body) {
  if (external_monitor_)
    throw ailoy::exception("You cannot call listen in this client");
  std::optional<signal_t> sig_opt = monitor_->monitor(timeout);
  if (sig_opt.has_value()) {
    return recv(skip_body);
  }
  return nullptr;
}

std::shared_ptr<packet_t> broker_client_t::listen(duration_t due,
                                                  bool skip_body) {
  return listen(now() + due, skip_body);
}

} // namespace ailoy
