#include <memory>
#include <optional>
#include <string>
#include <thread>

#include <pybind11/pybind11.h>

#include "broker.hpp"
#include "broker_client.hpp"
#include "language.hpp"
#include "module.hpp"
#include "uuid.hpp"
#include "vm.hpp"

#include "py_value_converters.hpp"

static std::optional<std::thread> broker_thread;
static std::optional<std::thread> vm_thread;

using namespace std::chrono_literals;
namespace py = pybind11;

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

void start_threads(const std::string &url) {
  broker_thread = std::thread{[&]() { ailoy::broker_start(url); }};
  std::shared_ptr<const ailoy::module_t> mods[] = {
      ailoy::get_default_module(), ailoy::get_language_module()};
  vm_thread = std::thread{[&]() { ailoy::vm_start(url, mods); }};
  std::this_thread::sleep_for(100ms);
}

void stop_threads(const std::string &url) {
  if (vm_thread.has_value()) {
    ailoy::vm_stop(url);
    vm_thread.value().join();
  }
  if (broker_thread.has_value()) {
    ailoy::broker_stop(url);
    broker_thread.value().join();
  }
}

std::string generate_uuid() { return ailoy::generate_uuid(); }

PYBIND11_MODULE(ailoy_py, m) {
  m.def("start_threads", &start_threads);
  m.def("stop_threads", &stop_threads);
  m.def("generate_uuid", &generate_uuid);

  py::class_<ailoy::broker_client_t, std::shared_ptr<ailoy::broker_client_t>>(
      m, "BrokerClient")
      .def(py::init<const std::string &>())
      .def("send_type1",
           [](std::shared_ptr<ailoy::broker_client_t> self,
              py::args args) -> bool {
             if (args.size() != 2) {
               throw std::runtime_error("Expected 2 arguments");
             }
             std::string txid = py::cast<std::string>(args[0]);
             std::string ptype = py::cast<std::string>(args[1]);
             if (ptype == "connect")
               return self->send<ailoy::packet_type::connect>(txid);
             else if (ptype == "disconnect")
               return self->send<ailoy::packet_type::disconnect>(txid);
           })
      .def("send_type2",
           [](std::shared_ptr<ailoy::broker_client_t> self,
              py::args args) -> bool {
             std::string txid = py::cast<std::string>(args[0]);
             std::string ptype = py::cast<std::string>(args[1]);
             std::string itype = py::cast<std::string>(args[2]);
             if (itype == "call_function") {
               std::string fname = py::cast<std::string>(args[3]);
               if (ptype == "subscribe")
                 return self->send<ailoy::packet_type::subscribe,
                                   ailoy::instruction_type::call_function>(
                     txid, fname);
               else if (ptype == "unsubscribe")
                 return self->send<ailoy::packet_type::unsubscribe,
                                   ailoy::instruction_type::call_function>(
                     txid, fname);
               else {
                 auto in = py::cast<std::shared_ptr<ailoy::value_t>>(args[4]);
                 return self->send<ailoy::packet_type::execute,
                                   ailoy::instruction_type::call_function>(
                     txid, fname, in);
               }
             } else if (itype == "define_component") {
               auto ctname = py::cast<std::string>(args[3]);
               if (ptype == "subscribe")
                 return self->send<ailoy::packet_type::subscribe,
                                   ailoy::instruction_type::define_component>(
                     txid, ctname);
               else if (ptype == "unsubscribe")
                 return self->send<ailoy::packet_type::unsubscribe,
                                   ailoy::instruction_type::define_component>(
                     txid, ctname);
               else {
                 auto cname = py::cast<std::string>(args[4]);
                 auto in = py::cast<std::shared_ptr<ailoy::value_t>>(args[5]);
                 return self->send<ailoy::packet_type::execute,
                                   ailoy::instruction_type::define_component>(
                     txid, ctname, cname, in);
               }
             } else if (itype == "delete_component") {
               auto cname = py::cast<std::string>(args[3]);
               if (ptype == "subscribe")
                 return self->send<ailoy::packet_type::subscribe,
                                   ailoy::instruction_type::delete_component>(
                     txid, cname);
               else if (ptype == "unsubscribe")
                 return self->send<ailoy::packet_type::unsubscribe,
                                   ailoy::instruction_type::delete_component>(
                     txid, cname);
               else {
                 return self->send<ailoy::packet_type::execute,
                                   ailoy::instruction_type::delete_component>(
                     txid, cname);
               }
             } else if (itype == "call_method") {
               auto cname = py::cast<std::string>(args[3]);
               auto fname = py::cast<std::string>(args[4]);
               if (ptype == "subscribe")
                 return self->send<ailoy::packet_type::subscribe,
                                   ailoy::instruction_type::call_method>(
                     txid, cname, fname);
               else if (ptype == "unsubscribe")
                 return self->send<ailoy::packet_type::unsubscribe,
                                   ailoy::instruction_type::call_method>(
                     txid, cname, fname);
               else {
                 auto in = py::cast<std::shared_ptr<ailoy::value_t>>(args[5]);
                 return self->send<ailoy::packet_type::execute,
                                   ailoy::instruction_type::call_method>(
                     txid, cname, fname, in);
               }
             } else {
               return false;
             }
           })
      .def("send_type3",
           [](std::shared_ptr<ailoy::broker_client_t> self,
              py::args args) -> bool {
             auto txid = py::cast<std::string>(args[0]);
             auto ptype = py::cast<std::string>(args[1]);
             auto status = py::cast<bool>(args[2]);
             auto sequence = py::cast<uint32_t>(args[3]);
             if (status) {
               auto done = py::cast<bool>(args[4]);
               auto out = py::cast<std::shared_ptr<ailoy::value_t>>(args[5]);
               return self->send<ailoy::packet_type::respond_execute, true>(
                   txid, sequence, done, out);
             } else {
               auto reason = py::cast<std::string>(args[4]);
               return self->send<ailoy::packet_type::respond_execute, false>(
                   txid, sequence, reason);
             }
             return false;
           })
      .def("listen",
           [](std::shared_ptr<ailoy::broker_client_t> self)
               -> std::shared_ptr<ailoy::value_t> {
             auto resp = self->listen(ailoy::timeout_default);
             if (resp == nullptr)
               return ailoy::create<ailoy::null_t>();

             auto ret = ailoy::create<ailoy::map_t>();
             ret->insert_or_assign("packet_type",
                                   ailoy::create<ailoy::string_t>(
                                       packet_type_to_string(resp->ptype)));

             if (resp->itype.has_value()) {
               ret->insert_or_assign(
                   "instruction_type",
                   ailoy::create<ailoy::string_t>(
                       instruction_type_to_string(resp->itype.value())));
             } else {
               ret->insert_or_assign("instruction_type",
                                     ailoy::create<ailoy::null_t>());
             }
             ret->insert_or_assign("headers", resp->headers);
             ret->insert_or_assign("body", resp->body);
             return ret;
           });
}
