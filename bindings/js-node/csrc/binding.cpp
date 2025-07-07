#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include <napi.h>

#include "broker.hpp"
#include "js_broker_client.hpp"
#include "js_ndarray.hpp"
#include "language.hpp"
#include "vm.hpp"

static std::unordered_map<std::string, std::thread> broker_threads;
static std::unordered_map<std::string, std::thread> vm_threads;

static void log(Napi::Env env, Napi::Value val) {
  Napi::Object console = env.Global().Get("console").As<Napi::Object>();
  Napi::Function logFunc = console.Get("log").As<Napi::Function>();
  logFunc.Call(console, {val});
}

void start_threads(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1)
    Napi::Error::New(info.Env(), "You must provide at least one argument")
        .ThrowAsJavaScriptException();
  std::string url = info[0].As<Napi::String>();

  if (!broker_threads.contains(url)) {
    std::thread broker_thread =
        std::thread{[&]() { ailoy::broker_start(url); }};
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

void stop_threads(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1)
    Napi::Error::New(info.Env(), "You must provide at least one argument")
        .ThrowAsJavaScriptException();
  std::string url = info[0].As<Napi::String>();

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

Napi::Value generate_uuid(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  auto uuid = ailoy::generate_uuid();
  return Napi::String::New(env, uuid);
}

Napi::Object init(Napi::Env env, Napi::Object exports) {
  exports.Set("NDArray", js_ndarray_t::initialize(env));
  exports.Set("BrokerClient", js_broker_client_t::initialize(env));
  exports.Set("startThreads", Napi::Function::New(env, start_threads));
  exports.Set("stopThreads", Napi::Function::New(env, stop_threads));
  exports.Set("generateUUID", Napi::Function::New(env, generate_uuid));
  return exports;
}

NODE_API_MODULE(_ailoy, init)
