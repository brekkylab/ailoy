#include <memory>
#include <string>
#include <unordered_map>

#include <emscripten/bind.h>
#include <emscripten/threading.h>
#include <emscripten/val.h>

#include "broker.hpp"
#include "broker_client.hpp"
#include "js_value_converters.hpp"
#include "language.hpp"
#include "module.hpp"
#include "uuid.hpp"
#include "vm.hpp"

// static std::unordered_map<std::string, std::thread> broker_threads;
// static std::unordered_map<std::string, std::thread> vm_threads;

using namespace emscripten;
using namespace std::chrono_literals;

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

// void start_threads(const std::string &url) {
//   if (!broker_threads.contains(url)) {
//     std::thread broker_thread =
//         std::thread{[&]() { ailoy::broker_start(url); }};
//     broker_threads.insert_or_assign(url, std::move(broker_thread));
//   }
//   if (!vm_threads.contains(url)) {
//     std::shared_ptr<const ailoy::module_t> mods[] = {
//         ailoy::get_default_module(), ailoy::get_language_module(),
//         ailoy::get_debug_module()};
//     std::thread vm_thread = std::thread{[&]() { ailoy::vm_start(url, mods);
//     }}; vm_threads.insert_or_assign(url, std::move(vm_thread));
//     std::this_thread::sleep_for(100ms);
//   }
// }

// void stop_threads(const std::string &url) {
//   std::unordered_map<std::string, std::thread>::iterator vm_thread;
//   if ((vm_thread = vm_threads.find(url)) != vm_threads.end()) {
//     ailoy::vm_stop(url);
//     vm_thread->second.join();
//     vm_threads.erase(vm_thread);
//   }
//   std::unordered_map<std::string, std::thread>::iterator broker_thread;
//   if ((broker_thread = broker_threads.find(url)) != broker_threads.end()) {
//     ailoy::broker_stop(url);
//     broker_thread->second.join();
//     broker_threads.erase(broker_thread);
//   }
// }

std::string generate_uuid() { return ailoy::generate_uuid(); }

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

class js_broker_client_t {
public:
  js_broker_client_t(const std::string &url = "inproc://") {
    inner_ = ailoy::create<ailoy::broker_client_t>(url);
  }

  bool send_type1(const std::string &txid, const std::string &ptype) {
    if (ptype == "connect")
      return inner_->send<ailoy::packet_type::connect>(txid);
    else if (ptype == "disconnect")
      return inner_->send<ailoy::packet_type::disconnect>(txid);
    else
      throw std::runtime_error("Unknown packet type: " + ptype);
  }

  bool send_type2(const std::string &txid, const std::string &ptype,
                  const std::string &itype, const std::string &arg1 = "",
                  const std::string &arg2 = "", val input = val::null()) {

    if (itype == "call_function") {
      std::string fname = arg1;
      if (ptype == "subscribe")
        return inner_->send<ailoy::packet_type::subscribe,
                            ailoy::instruction_type::call_function>(txid,
                                                                    fname);
      else if (ptype == "unsubscribe")
        return inner_->send<ailoy::packet_type::unsubscribe,
                            ailoy::instruction_type::call_function>(txid,
                                                                    fname);
      else { // if (ptype == "execute")
        auto in = from_js_val(input);
        return inner_->send<ailoy::packet_type::execute,
                            ailoy::instruction_type::call_function>(txid, fname,
                                                                    in);
      }
    } else if (itype == "define_component") {
      std::string ctname = arg1;
      if (ptype == "subscribe")
        return inner_->send<ailoy::packet_type::subscribe,
                            ailoy::instruction_type::define_component>(txid,
                                                                       ctname);
      else if (ptype == "unsubscribe")
        return inner_->send<ailoy::packet_type::unsubscribe,
                            ailoy::instruction_type::define_component>(txid,
                                                                       ctname);
      else { // if (ptype == "execute")
        std::string cname = arg2;
        auto in = from_js_val(input);
        return inner_->send<ailoy::packet_type::execute,
                            ailoy::instruction_type::define_component>(
            txid, ctname, cname, in);
      }
    } else if (itype == "delete_component") {
      std::string cname = arg1;
      if (ptype == "subscribe")
        return inner_->send<ailoy::packet_type::subscribe,
                            ailoy::instruction_type::delete_component>(txid,
                                                                       cname);
      else if (ptype == "unsubscribe")
        return inner_->send<ailoy::packet_type::unsubscribe,
                            ailoy::instruction_type::delete_component>(txid,
                                                                       cname);
      else { // execute
        return inner_->send<ailoy::packet_type::execute,
                            ailoy::instruction_type::delete_component>(txid,
                                                                       cname);
      }
    } else if (itype == "call_method") {
      std::string cname = arg1;
      std::string fname = arg2;
      if (ptype == "subscribe")
        return inner_->send<ailoy::packet_type::subscribe,
                            ailoy::instruction_type::call_method>(txid, cname,
                                                                  fname);
      else if (ptype == "unsubscribe")
        return inner_->send<ailoy::packet_type::unsubscribe,
                            ailoy::instruction_type::call_method>(txid, cname,
                                                                  fname);
      else { // execute
        auto in = from_js_val(input);
        return inner_->send<ailoy::packet_type::execute,
                            ailoy::instruction_type::call_method>(txid, cname,
                                                                  fname, in);
      }
    } else {
      return false;
    }
  }

  bool send_type3(const std::string &txid, const std::string &ptype,
                  bool status, uint32_t sequence, val arg1 = val::null(),
                  val arg2 = val::null()) {

    if (status) {
      // 성공 응답: arg1 = done (boolean), arg2 = output (value)
      bool done = arg1.as<bool>();
      auto out = from_js_val(arg2);
      return inner_->send<ailoy::packet_type::respond_execute, true>(
          txid, sequence, done, out);
    } else {
      // 실패 응답: arg1 = reason (string)
      std::string reason = arg1.as<std::string>();
      return inner_->send<ailoy::packet_type::respond_execute, false>(
          txid, sequence, reason);
    }
  }

private:
  std::shared_ptr<ailoy::broker_client_t> inner_;

  emscripten::val listen() { return emscripten::val::global("listen_async")(); }
};

EM_ASYNC_JS(EM_VAL, listen_async, (), {
  let resp = await Module.listen_async();
  return Emval.toHandle(resp);
});

EMSCRIPTEN_BINDINGS(ailoy_web) {
  // function("startThreads", &start_threads);
  // function("stopThreads", &stop_threads);
  function("generateUUID", &generate_uuid);

  class_<js_broker_client_t>("BrokerClient")
      .constructor<>()
      .constructor<std::string>()
      .function("sendType1", &js_broker_client_t::send_type1)
      .function("sendType2", &js_broker_client_t::send_type2)
      .function("sendType3", &js_broker_client_t::send_type3)
      .function("listen", &js_broker_client_t::listen)
}
