/**
 * @file thread.hpp
 * @brief Basic building blocks for multithreading
 * @details
 *
 * ## `signal_t`, `monitor_t`, `notify_t`
 *
 * A class of type `notify_t` can generate signals. For example:
 *
 * - Creating a new connection
 * - Receiving data from a socket
 * - Program termination
 *
 * When such signals occur, `notify_t` can notify this event(signal) to its
 * subscribers.
 *
 * The `monitor_t` class observes changes in `notify_t` instances. The
 * connection between `monitor_t` and `notify_t` is established using the
 * `notify_t::set_monitor` function, as shown below:
 *
 * ```cpp
 * {
 *   std::shared_ptr<monitor_t> monitor = monitor_t::create();
 *   std::shared_ptr<notify_t> notify = my_notifier_t::create();
 *   notify->set_monitor(monitor);
 * }
 * ```
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <shared_mutex>

#ifdef _WIN32
#include <Windows.h>
#endif

#if defined(UNIX) || defined(APPLE)
#include <signal.h>
#endif

#include "exception.hpp"
#include "object.hpp"

namespace ailoy {

using mutex_t = std::shared_mutex;

using condition_variable_t = std::condition_variable_any;

using wlock_t = std::unique_lock<mutex_t>;

using rlock_t = std::shared_lock<mutex_t>;

using time_point_t = std::chrono::system_clock::time_point;

using duration_t = std::chrono::system_clock::duration;

constexpr duration_t timeout_default = std::chrono::milliseconds(1000);

inline time_point_t now() { return std::chrono::system_clock::now(); }

/**
 * @brief A simple signal structure representing an event occurrence
 * @details
 * This structure captures the source (`who`) and the description (`what`)
 * of an event. It is passed from `notify_t` to `monitor_t` when an event
 * occurs.
 *
 * Example:
 * ```cpp
 * signal_t sig("socket", "recv");
 * ```
 */
struct signal_t {
  signal_t(const std::string &who, const std::string &what)
      : who(who), what(what) {}
  std::string who;
  std::string what;
};

struct notify_t;

/**
 * @brief A monitor that listens for `signal_t` events from attached `notify_t`
 * @details
 * `monitor_t` acts as a central receiver of events. `notify_t`-derived classes
 * call `notify()` to push a signal into the monitor's internal queue. The
 * `monitor()` method blocks until either a signal is received or the timeout
 * expires.
 *
 * ### Example usage:
 * ```cpp
 * auto monitor = std::make_shared<monitor_t>();
 * auto notifier = std::make_shared<my_notify_impl_t>();
 * notifier->set_monitor(monitor);
 *
 * if (auto sig = monitor->monitor(std::chrono::seconds(5))) {
 *     std::cout << sig->who << " sent " << sig->what << std::endl;
 * }
 * ```
 *
 * It is thread-safe for multiple `notify_t` to emit signals concurrently.
 */
class monitor_t : public object_t {
public:
  monitor_t()
      : m_(std::make_unique<mutex_t>()),
        cv_(std::make_unique<condition_variable_t>()), q_() {}

  monitor_t(const monitor_t &) = delete;

  monitor_t(monitor_t &&) = default;

  /**
   * @brief Waits for a signal until the specified deadline
   * @param due The absolute time point by which a signal should be received
   * @return A signal if one is received before the deadline, `std::nullopt`
   * otherwise
   */
  std::optional<signal_t> monitor(const time_point_t &due);

  /**
   * @brief Waits for a signal until the specified deadline
   * @param due The time duration by which a signal should be received
   * @return A signal if one is received before the deadline, `std::nullopt`
   * otherwise
   */
  std::optional<signal_t> monitor(const duration_t &due) {
    return monitor(now() + due);
  }

private:
  friend class notify_t;

  std::unique_ptr<mutex_t> m_;

  std::unique_ptr<condition_variable_t> cv_;

  std::deque<signal_t> q_;
};

/**
 * @brief Base class that emits `signal_t` to an attached `monitor_t`
 * @details
 * This class is meant to be inherited by components that detect or represent
 * significant system events, such as socket reads or task completions.
 *
 * It uses a unique name identifier (`myname`) per instance, which is
 * automatically generated. This name is used in the `signal_t::who` field when
 * notifying.
 *
 * To use it:
 * - Derive a class from `notify_t`
 * - Attach a monitor using `set_monitor()`
 * - Call `notify("your_event")` when an event occurs
 */
class notify_t : public object_t {
public:
  notify_t() : myname(new_name()) {}

  notify_t(std::shared_ptr<monitor_t> monitor)
      : myname(new_name()), monitor_(monitor) {
    on_monitor_set();
  }

  notify_t(const notify_t &) = default;

  notify_t(notify_t &&) = default;

  /**
   * @brief Emits a `signal_t` to the associated monitor
   * @param what The description of the event (e.g., "recv", "accept",
   * "disconnect")
   * @note Thread-safe if monitor remains valid
   */
  void notify(const std::string &what);

  /**
   * @brief Assigns a monitor to this notifier
   * @param monitor Shared pointer to a `monitor_t` instance
   * @note This will overwrite any previously set monitor
   * @note Calls `on_monitor_set()` internally, which can be overridden by
   * subclasses
   */
  void set_monitor(std::shared_ptr<monitor_t> monitor);

  const std::string myname;

private:
  friend class monitor_t;

  static size_t next_id;

  static std::string new_name() { return std::to_string(next_id++); }

  /**
   * @brief Hook can be called when a monitor is set to this
   */
  virtual void on_monitor_set() {}

  std::weak_ptr<monitor_t> monitor_;
};

/**
 * @brief Simple notifier implementation for stopping thread
 */
class stop_t : public notify_t {
public:
  stop_t(bool handle_signal = false) {
    exit_.store(false);
#if defined(_WIN32)
    if (handle_signal)
      if (!SetConsoleCtrlHandler(stop_t::console_handler, TRUE))
        throw exception("Failed to set sigint");
#elif defined(EMSCRIPTEN)
#else
    if (handle_signal)
      signal(SIGINT, stop_t::signal_handler);
#endif
  }

  operator bool() const { return exit_.load(); }

  void stop() {
    exit_.store(true);
    notify("stop");
  }

#if defined(_WIN32)
  static BOOL WINAPI console_handler(DWORD signal) {
    if (signal == CTRL_C_EVENT) {
      global_stop.stop();
      return TRUE;
    }
    return TRUE;
  }
#elif defined(EMSCRIPTEN)
#else
  static void signal_handler(int signum) {
    if (signum == SIGINT)
      global_stop.stop();
  }
#endif

  static stop_t global_stop;

private:
  std::atomic_bool exit_;
};

} // namespace ailoy
