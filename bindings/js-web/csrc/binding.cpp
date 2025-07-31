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
                  const std::string &itype,
                  const std::vector<std::string> args) {
    if (itype == "call_function") {
      std::string fname = args[0];
      if (ptype == "subscribe")
        return client_->send<ailoy::packet_type::subscribe,
                             ailoy::instruction_type::call_function>(txid,
                                                                     fname);
      else if (ptype == "unsubscribe")
        return client_->send<ailoy::packet_type::unsubscribe,
                             ailoy::instruction_type::call_function>(txid,
                                                                     fname);
      else {
        // TODO: convert second arg to value_t
        // auto in = py::cast<std::shared_ptr<ailoy::value_t>>(args[4]);
        auto in = ailoy::create<ailoy::map_t>();
        return client_->send<ailoy::packet_type::execute,
                             ailoy::instruction_type::call_function>(txid,
                                                                     fname, in);
      }
    } else if (itype == "define_component") {
      auto ctname = args[0];
      if (ptype == "subscribe")
        return client_->send<ailoy::packet_type::subscribe,
                             ailoy::instruction_type::define_component>(txid,
                                                                        ctname);
      else if (ptype == "unsubscribe")
        return client_->send<ailoy::packet_type::unsubscribe,
                             ailoy::instruction_type::define_component>(txid,
                                                                        ctname);
      else {
        // auto cname = py::cast<std::string>(args[4]);
        // auto in = py::cast<std::shared_ptr<ailoy::value_t>>(args[5]);
        auto cname = args[1];
        auto in = ailoy::create<ailoy::map_t>();
        return client_->send<ailoy::packet_type::execute,
                             ailoy::instruction_type::define_component>(
            txid, ctname, cname, in);
      }
    } else if (itype == "delete_component") {
      auto cname = args[0];
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
      auto cname = args[0];
      auto fname = args[1];
      if (ptype == "subscribe")
        return client_->send<ailoy::packet_type::subscribe,
                             ailoy::instruction_type::call_method>(txid, cname,
                                                                   fname);
      else if (ptype == "unsubscribe")
        return client_->send<ailoy::packet_type::unsubscribe,
                             ailoy::instruction_type::call_method>(txid, cname,
                                                                   fname);
      else {
        // auto in = py::cast<std::shared_ptr<ailoy::value_t>>(args[5]);
        auto in = ailoy::create<ailoy::map_t>();
        return client_->send<ailoy::packet_type::execute,
                             ailoy::instruction_type::call_method>(txid, cname,
                                                                   fname, in);
      }
    } else {
      return false;
    }
  }

  bool send_type3(const std::string &txid, const std::string &ptype,
                  uint32_t sequence, bool done) {
    //   auto out = py::cast<std::shared_ptr<ailoy::value_t>>(args[5]);
    auto out = ailoy::create<ailoy::map_t>();
    return client_->send<ailoy::packet_type::respond_execute, true>(
        txid, sequence, done, out);
  }

  bool send_type4(const std::string &txid, const std::string &ptype,
                  uint32_t sequence, const std::string &reason) {
    return client_->send<ailoy::packet_type::respond_execute, false>(
        txid, sequence, reason);
  }

  val listen() {
    auto resp = client_->listen(ailoy::timeout_default);
    if (resp == nullptr)
      return val::null();

    auto ret = val::object();
    ret.set("packet_type", packet_type_to_string(resp->ptype));

    if (resp->itype.has_value()) {
      ret.set("instruction_type",
              instruction_type_to_string(resp->itype.value()));
    } else {
      ret.set("instruction_type", val::null());
    }
    // TODO: parse headers and body
    // ret.set("headers", resp->headers);
    // ret.set("body", resp->body);
    ret.set("headers", val::array());
    ret.set("body", val::object());
    return ret;
  }

private:
  std::shared_ptr<ailoy::broker_client_t> client_;
};

EMSCRIPTEN_BINDINGS(ailoy_web) {
  register_vector<std::string>("VectorString");

  function("start_threads", &start_threads);
  function("stop_threads", &stop_threads);
  function("generate_uuid", &generate_uuid);

  class_<broker_client_wrapper_t>("BrokerClient")
      .constructor<std::string>()
      .function("send_type1", &broker_client_wrapper_t::send_type1)
      .function("send_type2", &broker_client_wrapper_t::send_type2)
      .function("send_type3", &broker_client_wrapper_t::send_type3)
      .function("send_type4", &broker_client_wrapper_t::send_type4)
      .function("listen", &broker_client_wrapper_t::listen);
}
