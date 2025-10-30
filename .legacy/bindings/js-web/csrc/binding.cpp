#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "broker.hpp"
#include "broker_client.hpp"
#include "language.hpp"
#include "module.hpp"
#include "uuid.hpp"
#include "vm.hpp"

#include "./ndarray.hpp"
#include "./value_converters.hpp"

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
  }

  // Wait until VM is ready
  while (!ailoy::vm_ready.load()) {
    emscripten_sleep(1);
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

std::string packet_type_to_string(const ailoy::packet_type &ptype) {
  if (ptype == ailoy::packet_type::connect)
    return "connect";
  else if (ptype == ailoy::packet_type::disconnect)
    return "disconnect";
  else if (ptype == ailoy::packet_type::subscribe)
    return "subscribe";
  else if (ptype == ailoy::packet_type::unsubscribe)
    return "unsubscribe";
  else if (ptype == ailoy::packet_type::execute)
    return "execute";
  else if (ptype == ailoy::packet_type::respond)
    return "respond";
  else if (ptype == ailoy::packet_type::respond_execute)
    return "respond_execute";
  else
    throw ailoy::exception("invalid packet type");
}

std::string instruction_type_to_string(const ailoy::instruction_type &itype) {
  if (itype == ailoy::instruction_type::call_function)
    return "call_function";
  else if (itype == ailoy::instruction_type::define_component)
    return "define_component";
  else if (itype == ailoy::instruction_type::delete_component)
    return "delete_component";
  else if (itype == ailoy::instruction_type::call_method)
    return "call_method";
  else
    throw ailoy::exception("invalid instruction type");
}

class broker_client_wrapper_t {
public:
  broker_client_wrapper_t(const std::string &url)
      : client_(std::make_shared<ailoy::broker_client_t>(url)) {}

  bool send_type1(const std::string &txid, const std::string &ptype) {
    if (ptype == "connect")
      return client_->send<ailoy::packet_type::connect>(txid);
    else if (ptype == "disconnect")
      return client_->send<ailoy::packet_type::disconnect>(txid);
    else
      return false;
  }

  bool send_type2(const std::string &txid, const std::string &ptype,
                  const std::string &itype, const val &args) {
    auto args_ = *(from_em_val(args)->as<ailoy::array_t>());
    if (itype == "call_function") {
      std::string fname = *args_[0]->as<ailoy::string_t>();
      if (ptype == "subscribe")
        return client_->send<ailoy::packet_type::subscribe,
                             ailoy::instruction_type::call_function>(txid,
                                                                     fname);
      else if (ptype == "unsubscribe")
        return client_->send<ailoy::packet_type::unsubscribe,
                             ailoy::instruction_type::call_function>(txid,
                                                                     fname);
      else {
        auto in = args_[1]->as<ailoy::map_t>();
        return client_->send<ailoy::packet_type::execute,
                             ailoy::instruction_type::call_function>(txid,
                                                                     fname, in);
      }
    } else if (itype == "define_component") {
      std::string ctname = *args_[0]->as<ailoy::string_t>();
      if (ptype == "subscribe")
        return client_->send<ailoy::packet_type::subscribe,
                             ailoy::instruction_type::define_component>(txid,
                                                                        ctname);
      else if (ptype == "unsubscribe")
        return client_->send<ailoy::packet_type::unsubscribe,
                             ailoy::instruction_type::define_component>(txid,
                                                                        ctname);
      else {
        std::string cname = *args_[1]->as<ailoy::string_t>();
        auto in = args_[2]->as<ailoy::map_t>();
        return client_->send<ailoy::packet_type::execute,
                             ailoy::instruction_type::define_component>(
            txid, ctname, cname, in);
      }
    } else if (itype == "delete_component") {
      std::string cname = *args_[0]->as<ailoy::string_t>();
      if (ptype == "subscribe")
        return client_->send<ailoy::packet_type::subscribe,
                             ailoy::instruction_type::delete_component>(txid,
                                                                        cname);
      else if (ptype == "unsubscribe")
        return client_->send<ailoy::packet_type::unsubscribe,
                             ailoy::instruction_type::delete_component>(txid,
                                                                        cname);
      else {
        return client_->send<ailoy::packet_type::execute,
                             ailoy::instruction_type::delete_component>(txid,
                                                                        cname);
      }
    } else if (itype == "call_method") {
      std::string cname = *args_[0]->as<ailoy::string_t>();
      std::string fname = *args_[1]->as<ailoy::string_t>();
      if (ptype == "subscribe")
        return client_->send<ailoy::packet_type::subscribe,
                             ailoy::instruction_type::call_method>(txid, cname,
                                                                   fname);
      else if (ptype == "unsubscribe")
        return client_->send<ailoy::packet_type::unsubscribe,
                             ailoy::instruction_type::call_method>(txid, cname,
                                                                   fname);
      else {
        auto in = args_[2];
        return client_->send<ailoy::packet_type::execute,
                             ailoy::instruction_type::call_method>(txid, cname,
                                                                   fname, in);
      }
    } else {
      return false;
    }
  }

  bool send_type3(const std::string &txid, const std::string &ptype,
                  bool status, uint32_t sequence, const val &args) {
    auto args_ = *(from_em_val(args)->as<ailoy::array_t>());
    if (status) {
      bool done = *args_[0]->as<ailoy::bool_t>();
      auto out = args_[1]->as<ailoy::map_t>();
      return client_->send<ailoy::packet_type::respond_execute, true>(
          txid, sequence, done, out);
    } else {
      std::string reason = *args_[0]->as<ailoy::string_t>();
      return client_->send<ailoy::packet_type::respond_execute, false>(
          txid, sequence, reason);
    }
  }

  val listen() {
    auto resp = client_->listen(ailoy::timeout_default);
    if (resp == nullptr)
      return val::null();

    auto ret = val::object();
    std::string ptype = packet_type_to_string(resp->ptype);
    ret.set("packet_type", ptype);

    if (resp->itype.has_value()) {
      std::string itype = instruction_type_to_string(resp->itype.value());
      ret.set("instruction_type", itype);
    } else {
      ret.set("instruction_type", val::null());
    }

    ret.set("headers", to_em_val(resp->headers));
    ret.set("body", to_em_val(resp->body));

    return ret;
  }

private:
  std::shared_ptr<ailoy::broker_client_t> client_;
};

EMSCRIPTEN_BINDINGS(ailoy_web) {
  register_vector<std::string>("VectorString");

  function("start_threads", &start_threads);
  function("stop_threads", &stop_threads);
  function("generate_uuid", &ailoy::generate_uuid);

  class_<js_ndarray_t>("NDArray")
      .constructor<const val &>()
      .function("toString", &js_ndarray_t::to_string)
      .function("valueOf", &js_ndarray_t::to_string)
      .function("getShape", &js_ndarray_t::get_shape)
      .function("getDtype", &js_ndarray_t::get_dtype)
      .function("getData", &js_ndarray_t::get_data);

  class_<broker_client_wrapper_t>("BrokerClient")
      .constructor<std::string>()
      .function("send_type1", &broker_client_wrapper_t::send_type1)
      .function("send_type2", &broker_client_wrapper_t::send_type2)
      .function("send_type3", &broker_client_wrapper_t::send_type3)
      .function("listen", &broker_client_wrapper_t::listen);
}
