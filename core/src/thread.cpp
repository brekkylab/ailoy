#include "thread.hpp"

namespace ailoy {

std::optional<signal_t> monitor_t::monitor(const time_point_t &due) {
  wlock_t lk{*m_, std::defer_lock};
  lk.lock();
  if (cv_->wait_until(lk, due, [&] { return q_.size() > 0; })) {
    auto rv = q_.front();
    q_.pop_front();
    lk.unlock();
    return rv;
  } else
    return std::nullopt;
}

size_t notify_t::next_id = 0;

void notify_t::notify(const std::string &what) {
  std::shared_ptr<monitor_t> monitor = monitor_.lock();
  if (!monitor)
    return;

  wlock_t lk{*monitor->m_, std::defer_lock};
  lk.lock();
  monitor->q_.push_back(signal_t(myname, what));
  lk.unlock();
  monitor->cv_->notify_all();
}

void notify_t::set_monitor(std::shared_ptr<monitor_t> monitor) {
  monitor_ = monitor;
  on_monitor_set();
}

stop_t stop_t::global_stop(true);

} // namespace ailoy
