#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "broker.hpp"
#include "language.hpp"
#include "uuid.hpp"
#include "vm.hpp"

using namespace std::chrono_literals;
using namespace emscripten;

static std::unordered_map<std::string, std::thread> broker_threads;
static std::unordered_map<std::string, std::thread> vm_threads;

void start_threads() {
  std::string url = "inproc://";
  if (!broker_threads.contains(url)) {
    std::thread broker_thread =
        std::thread([&]() { ailoy::broker_start(url); });
    broker_threads.insert_or_assign(url, std::move(broker_thread));
  }
  if (!vm_threads.contains(url)) {
    std::shared_ptr<const ailoy::module_t> mods[] = {
        ailoy::get_default_module(), ailoy::get_language_module(),
        ailoy::get_debug_module()};
    std::thread vm_thread = std::thread{[&]() { ailoy::vm_start(url, mods); }};
    vm_threads.insert_or_assign(url, std::move(vm_thread));
    std::this_thread::sleep_for(100ms);
  }
}

void stop_threads() {
  std::string url = "inproc://";

  std::unordered_map<std::string, std::thread>::iterator vm_thread =
      vm_threads.find(url);
  if (vm_thread != vm_threads.end()) {
    ailoy::vm_stop(url);
    vm_thread->second.join();
    vm_threads.erase(vm_thread);
  }
  std::unordered_map<std::string, std::thread>::iterator broker_thread =
      broker_threads.find(url);
  if (broker_thread != broker_threads.end()) {
    ailoy::broker_stop(url);
    broker_thread->second.join();
    broker_threads.erase(broker_thread);
  }
}

val generate_uuid() {
  auto uuid = ailoy::generate_uuid();
  return val(uuid);
}

EMSCRIPTEN_BINDINGS(ailoy_web) {
  function("start_threads", &start_threads);
  function("stop_threads", &stop_threads);
  function("generate_uuid", &generate_uuid);
}
