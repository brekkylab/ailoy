#include "thread.hpp"

#ifdef EMSCRIPTEN
#include <emscripten.h>
#endif

namespace ailoy {

#ifdef EMSCRIPTEN

std::optional<signal_t> monitor_t::monitor(const time_point_t &due) {
  const auto sleep_interval = std::chrono::milliseconds(1);
  while (true) {
    // Check current time
    auto now_time = std::chrono::system_clock::now();
    if (now_time >= due) {
      // Timeout reached, check one final time for signals
      wlock_t lk{*m_, std::defer_lock};
      lk.lock();
      if (q_.size() > 0) {
        auto rv = q_.front();
        q_.pop_front();
        lk.unlock();
        return rv;
      }
      lk.unlock();
      return std::nullopt;
    }

    // Check for signals without blocking
    {
      wlock_t lk{*m_, std::defer_lock};
      if (lk.try_lock()) {
        if (q_.size() > 0) {
          auto rv = q_.front();
          q_.pop_front();
          lk.unlock();
          return rv;
        }
        lk.unlock();
      }
    }

    // Calculate how long to sleep (don't sleep past the due time)
    auto time_remaining = due - now_time;
    auto sleep_duration = std::min(
        sleep_interval,
        std::chrono::duration_cast<std::chrono::milliseconds>(time_remaining));
    if (sleep_duration.count() > 0) {
      emscripten_sleep(static_cast<double>(sleep_duration.count()));
    }
  }
}

#else

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

#endif

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

} // namespace ailoy
