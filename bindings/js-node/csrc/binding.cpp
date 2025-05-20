#include <optional>
#include <thread>

#include <napi.h>

#include "broker.hpp"
#include "js_broker_client.hpp"
#include "js_ndarray.hpp"
#include "language.hpp"
#include "vm.hpp"

static std::optional<std::thread> broker_thread;
static std::optional<std::thread> vm_thread;

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

  broker_thread = std::thread{[&]() { ailoy::broker_start(url); }};
  std::shared_ptr<const ailoy::module_t> mods[] = {ailoy::get_default_module(),
                                                   ailoy::get_language_module(),
                                                   ailoy::get_debug_module()};
  vm_thread = std::thread{[&]() { ailoy::vm_start(url, mods); }};
  std::this_thread::sleep_for(100ms);
}

void stop_threads(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1)
    Napi::Error::New(info.Env(), "You must provide at least one argument")
        .ThrowAsJavaScriptException();
  std::string url = info[0].As<Napi::String>();

  if (vm_thread.has_value()) {
    ailoy::vm_stop(url);
    vm_thread.value().join();
  }
  if (broker_thread.has_value()) {
    ailoy::broker_stop(url);
    broker_thread.value().join();
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
