#include "inproc_socket.hpp"

#include <unordered_map>

using namespace std::chrono_literals;

namespace ailoy {
namespace inproc {

struct dialer_t {
  std::unordered_map<url_t, acceptor_t *> acceptors;
  std::unordered_multimap<url_t, std::shared_ptr<socket_t>> connections;

  mutex_t m;
  condition_variable_t cv;
};

static dialer_t dialer;

template <typename mail_t>
  requires is_object<mail_t>
bool mailbox_t<mail_t>::setter_t::set(std::shared_ptr<mail_t> mail) {
  if (inner_.expired())
    return false;
  std::shared_ptr<mailbox_t<mail_t>> inner = inner_.lock();
  wlock_t lk(inner->m, std::defer_lock);
  lk.lock();
  if (inner->q.size() >= limit)
    return false;
  inner->q.push_back(mail);
  return true;
}

template <typename mail_t>
  requires is_object<mail_t>
std::shared_ptr<mail_t> mailbox_t<mail_t>::get() {
  wlock_t lk(m, std::defer_lock);
  lk.lock();
  if (q.size() == 0)
    return nullptr;
  std::shared_ptr<mail_t> rv = q.front();
  q.pop_front();
  return rv;
}

bool socket_t::connect(const url_t &url) {
  wlock_t lk(dialer.m, std::defer_lock);
  lk.lock();
  auto acceptor_it = dialer.acceptors.find(url);

  if (acceptor_it != dialer.acceptors.end())
    return acceptor_it->second->request_accepting(as<socket_t>());
  else {
    // Add this into `dialer.connections`(waitlist)
    dialer.connections.insert(std::make_pair(url, as<socket_t>()));
    bool res = dialer.cv.wait_for(
        lk, 1s, [&] { return !dialer.connections.contains(url); });
    if (res)
      return true;
    else {
      // Clear waitlist registration
      auto range = dialer.connections.equal_range(url);
      for (auto conn_it = range.first; conn_it != range.second; ++conn_it) {
        if (conn_it->second == as<socket_t>()) {
          dialer.connections.erase(conn_it);
          break;
        }
      }
      return false;
    }
  }
}

bool socket_t::wait_until_attached(time_point_t tp) {
  wlock_t lk(dialer.m, std::defer_lock);
  lk.lock();
  return cv.wait_until(lk, tp, [&] { return peer_name.has_value(); });
}

void socket_t::attach(std::shared_ptr<socket_t> peer) {
  peer_mailbox = create_setter(peer->my_mailbox);
  peer_name = peer->myname;
  peer_notify = [peer = std::weak_ptr(peer)]() {
    auto peer_ = peer.lock();
    if (!peer_)
      return false;
    peer_->notify("recv");
    return true;
  };
}

bool socket_t::send(const std::shared_ptr<bytes_t> msg) {
  if (!peer_name.has_value() || !peer_mailbox || !peer_notify.has_value())
    return false;
  if (peer_mailbox->set(msg))
    return peer_notify.value()();
  else
    return false;
}

std::shared_ptr<bytes_t> socket_t::recv() { return my_mailbox->get(); }

acceptor_t::acceptor_t(const std::string &url)
    : notify_t(), url_(url), mailbox_(ailoy::create<mailbox_t<socket_t>>()) {
  wlock_t lk(dialer.m, std::defer_lock);
  lk.lock();

  // Register itself into dialer
  if (dialer.acceptors.find(url) != dialer.acceptors.end())
    throw ailoy::exception("Acceptor for this url already exists");
  dialer.acceptors.insert_or_assign(url, this);
}

acceptor_t::~acceptor_t() {
  wlock_t lk(dialer.m, std::defer_lock);
  lk.lock();

  // Find url of this
  std::optional<url_t> this_url;
  for (auto [k, v] : dialer.acceptors) {
    if (v == this) {
      this_url = k;
      break;
    }
  }

  // Unregister itself from dialer
  if (this_url.has_value())
    dialer.acceptors.erase(this_url.value());
}

void acceptor_t::on_monitor_set() {
  // Handle all trailing connections
  auto range = dialer.connections.equal_range(url_);
  for (auto it = range.first; it != range.second; ++it) {
    request_accepting(it->second);
  }
  dialer.connections.erase(range.first, range.second);
}

std::shared_ptr<socket_t> acceptor_t::accept() {
  auto peer_socket = mailbox_->get();
  if (peer_socket) {
    auto my_socket = create<socket_t>();
    my_socket->attach(peer_socket);
    peer_socket->attach(my_socket);
    peer_socket->cv.notify_all();
    dialer.cv.notify_all();
    return my_socket;
  } else
    return nullptr;
}

bool acceptor_t::request_accepting(std::shared_ptr<socket_t> client_socket) {
  auto rv = create_setter(mailbox_)->set(client_socket);
  notify("accept");
  return rv;
}

} // namespace inproc
} // namespace ailoy
