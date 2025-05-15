#pragma once

#include <deque>
#include <memory>
#include <string>

#include "object.hpp"
#include "thread.hpp"
#include "value.hpp"

namespace ailoy {
namespace inproc {

template <typename mail_t>
  requires is_object<mail_t>
struct mailbox_t : public object_t {
  mailbox_t() = default;

  struct setter_t : private std::enable_shared_from_this<setter_t> {
    setter_t(std::weak_ptr<mailbox_t<mail_t>> ack) : inner_(ack) {}

    bool set(std::shared_ptr<mail_t> mail);

  private:
    std::weak_ptr<mailbox_t<mail_t>> inner_;
  };

  std::shared_ptr<mail_t> get();

  /**
   * @brief maximum size of the mailbox
   */
  static constexpr size_t limit = 128;

  mutex_t m;

  std::deque<std::shared_ptr<mail_t>> q;
};

struct socket_t : public notify_t {
  socket_t()
      : my_mailbox(ailoy::create<mailbox_t<bytes_t>>()), peer_mailbox() {}

  bool connect(const url_t &url);

  bool wait_until_attached(time_point_t tp);

  bool wait_until_attached(duration_t due = std::chrono::seconds(1)) {
    return wait_until_attached(now() + due);
  }

  void attach(std::shared_ptr<socket_t> peer);

  bool send(const std::shared_ptr<bytes_t> msg);

  std::shared_ptr<bytes_t> recv();

  std::shared_ptr<mailbox_t<bytes_t>> my_mailbox;

  std::unique_ptr<mailbox_t<bytes_t>::setter_t> peer_mailbox;

  std::optional<std::string> peer_name;

  std::optional<std::function<bool()>> peer_notify;

  mutex_t m;

  condition_variable_t cv;
};

struct acceptor_t : public notify_t {
public:
  acceptor_t(const std::string &url);

  ~acceptor_t();

  void on_monitor_set() override;

  std::shared_ptr<socket_t> accept();

  bool request_accepting(std::shared_ptr<socket_t> client_socket);

private:
  std::string url_;

  std::shared_ptr<mailbox_t<socket_t>> mailbox_;
};

} // namespace inproc
} // namespace ailoy
