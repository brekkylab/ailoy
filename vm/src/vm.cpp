#include <iostream>
#include <set>

#include <magic_enum/magic_enum.hpp>

#include "broker_client.hpp"
#include "exception.hpp"
#include "logging.hpp"
#include "uuid.hpp"
#include "vm.hpp"

using namespace std::chrono_literals;

namespace ailoy {

struct vm_state_t : public notify_t {
  vm_state_t(std::shared_ptr<monitor_t> monitor) : stop(create<stop_t>()) {
    stop->set_monitor(monitor);
    set_monitor(monitor);
  }

  /**
   * @return list of subscription packets have to be sent to broker
   */
  std::vector<std::pair<tx_id_t, std::shared_ptr<bytes_t>>>
  on_import_module(std::shared_ptr<const module_t> mod) {
    std::vector<std::pair<tx_id_t, std::shared_ptr<bytes_t>>> rv;
    for (auto kv : mod->ops) {
      operators.insert(kv);
      tx_id_t tx_id = generate_uuid();
      auto pkt =
          dump_packet<packet_type::subscribe, instruction_type::call_function>(
              tx_id, kv.first);
      rv.push_back(std::make_pair(tx_id, pkt));
    }
    for (auto kv : mod->factories) {
      factories.insert(kv);
      auto tx_id = generate_uuid();
      auto pkt =
          dump_packet<packet_type::subscribe,
                      instruction_type::define_component>(tx_id, kv.first);
      rv.push_back(std::make_pair(tx_id, pkt));
    }
    return rv;
  }

  std::vector<std::pair<tx_id_t, std::shared_ptr<bytes_t>>> on_stop() {
    std::vector<std::pair<tx_id_t, std::shared_ptr<bytes_t>>> rv;
    for (auto [name, _] : operators) {
      auto tx_id = generate_uuid();
      auto pkt = dump_packet<packet_type::unsubscribe,
                             instruction_type::call_function>(tx_id, name);
      rv.push_back(std::make_pair(tx_id, pkt));
    }
    for (auto [name, _] : factories) {
      auto tx_id = generate_uuid();
      auto pkt = dump_packet<packet_type::unsubscribe,
                             instruction_type::define_component>(tx_id, name);
      rv.push_back(std::make_pair(tx_id, pkt));
    }
    for (auto [cname, comp] : components) {
      auto tx_id = generate_uuid();
      auto pkt = dump_packet<packet_type::unsubscribe,
                             instruction_type::delete_component>(tx_id, cname);
      rv.push_back(std::make_pair(tx_id, pkt));
      for (auto [opname, _] : comp->get_operators()) {
        auto tx_id = generate_uuid();
        auto pkt =
            dump_packet<packet_type::unsubscribe,
                        instruction_type::call_method>(tx_id, cname, opname);
        rv.push_back(std::make_pair(tx_id, pkt));
      }
    }
    return rv;
  }

  std::unordered_map<std::string, component_factory_t> factories;

  std::unordered_map<std::string, std::shared_ptr<component_t>> components;

  std::unordered_map<std::string, std::shared_ptr<operator_t>> operators;

  std::shared_ptr<stop_t> stop;
};

static std::unordered_map<std::string, std::shared_ptr<vm_state_t>> vm_states;

void handle_call_function(std::shared_ptr<packet_t> pkt,
                          std::shared_ptr<broker_client_t> client,
                          std::shared_ptr<vm_state_t> vm_state);

void handle_define_component(std::shared_ptr<packet_t> pkt,
                             std::shared_ptr<broker_client_t> client,
                             std::shared_ptr<vm_state_t> vm_state,
                             std::set<tx_id_t> &expected_responses);

void handle_delete_component(std::shared_ptr<packet_t> pkt,
                             std::shared_ptr<broker_client_t> client,
                             std::shared_ptr<vm_state_t> vm_state,
                             std::set<tx_id_t> &expected_responses);

void handle_call_method(std::shared_ptr<packet_t> pkt,
                        std::shared_ptr<broker_client_t> client,
                        std::shared_ptr<vm_state_t> vm_state);

void vm_start(const std::string &url,
              std::span<std::shared_ptr<const module_t>> mods,
              const std::string &name) {
  // Unique ID of the vm
  std::string vm_id = url + ":" + name;
  if (vm_states.find(vm_id) != vm_states.end()) {
    throw ailoy::exception(std::format(
        "The VM name '{}' is already occupied for the url '{}'.", name, url));
  }

  // Helper function
  auto raise_exception = [&](const std::string &reason) {
    vm_states.erase(vm_id);
    throw ailoy::exception(reason);
  };

  // tx_id of packets that have been transmitted and are waiting for a response
  std::set<tx_id_t> expected_responses;

  // Monitor used by this vm
  auto monitor = ailoy::create<monitor_t>();

  // Create broker client
  auto client = ailoy::create<broker_client_t>(url, monitor);
  if (!client)
    raise_exception("Connection failed");

  // Initialize vm state
  auto vm_state = ailoy::create<vm_state_t>(monitor);
  vm_states.insert_or_assign(vm_id, vm_state);
  vm_state->set_monitor(monitor);

  // Send connection & Receive response
  {
    auto tx_id = generate_uuid();
    if (!client->send<packet_type::connect>(tx_id))
      raise_exception("Connection packet send failed");
    if (!monitor->monitor(1s))
      raise_exception("Connection response packet not arrived");
    auto packet = client->recv();
    if (!packet)
      raise_exception("Connection response packet not arrived");
    if (packet->ptype != packet_type::respond || packet->get_tx_id() != tx_id)
      raise_exception("Invalid connection response packet");
    if (!packet->body->at<bool_t>("status")) {
      if (packet->body->find("reason") == packet->body->end())
        raise_exception("Invalid connection response packet");
      else
        raise_exception(*(packet->body->at<string_t>("reason")));
    }
  }

  // Import modules & subscribe channels
  for (auto m : mods) {
    for (auto [tx_id, pkt] : vm_state->on_import_module(m)) {
      expected_responses.insert(tx_id);
      if (!client->send_bytes(pkt))
        raise_exception("Initialization packet send failed");
    }
  }

  // Main event loop
  while (true) {
    auto signal_opt = monitor->monitor(100ms);
    if (!signal_opt.has_value())
      continue;
    auto signal = signal_opt.value();

    if (signal.what == "stop") {
      for (auto [tx_id, pkt] : vm_state->on_stop()) {
        expected_responses.insert(tx_id);
        client->send_bytes(pkt);
      }
      break;
    } else if (signal.what == "recv") {
      auto pkt = client->recv();
      if (!pkt) {
        error("[VM] message arrived but no contents");
        continue;
      }

      if (pkt->itype.has_value())
        debug("[VM] packet received: {} {} {}", pkt->get_tx_id(),
              magic_enum::enum_name(pkt->ptype),
              magic_enum::enum_name(pkt->itype.value()));
      else
        debug("[VM] packet received: {} {}", pkt->get_tx_id(),
              magic_enum::enum_name(pkt->ptype));

      if (pkt->ptype == packet_type::respond) {
        if (expected_responses.find(pkt->get_tx_id()) !=
            expected_responses.end())
          expected_responses.erase(pkt->get_tx_id());
        if (!(*pkt->body->at<bool_t>("status"))) {
          error("[VM] {}", std::string(*pkt->body->at<string_t>("reason")));
        }
      } else if (pkt->ptype == packet_type::execute) {
        if (pkt->itype.value() == instruction_type::call_function) {
          handle_call_function(pkt, client, vm_state);
        } else if (pkt->itype.value() == instruction_type::define_component) {
          handle_define_component(pkt, client, vm_state, expected_responses);
        } else if (pkt->itype.value() == instruction_type::delete_component) {
          handle_delete_component(pkt, client, vm_state, expected_responses);
        } else if (pkt->itype.value() == instruction_type::call_method) {
          handle_call_method(pkt, client, vm_state);
        }
      }
    }
  }

  // Squeeze all remaining responses
  while (true) {
    static size_t retry = 0;
    if (expected_responses.empty())
      break;
    if (retry >= 3) {
      std::string warnstr =
          "[VM] Failed to get some responses for the transaction: ";
      for (auto tx_id : expected_responses)
        warnstr += tx_id + " ";
      warnstr += " -> Force exit";
      warn(warnstr);
      break;
    }
    auto signal_opt = monitor->monitor(100ms);
    if (!signal_opt.has_value()) {
      retry++;
      continue;
    }
    retry = 0;
    auto signal = signal_opt.value();
    if (signal.what == "recv") {
      auto packet = client->recv();

      if (packet->itype.has_value())
        debug("[VM] packet received: {} {} {}", packet->get_tx_id(),
              magic_enum::enum_name(packet->ptype),
              magic_enum::enum_name(packet->itype.value()));
      else
        debug("[VM] packet received: {} {}", packet->get_tx_id(),
              magic_enum::enum_name(packet->ptype));

      if (packet->ptype != packet_type::respond) {
        std::cerr << "[VM] ignoring packet " << packet->get_tx_id()
                  << std::endl;
        continue;
      }
      if (expected_responses.find(packet->get_tx_id()) !=
          expected_responses.end())
        expected_responses.erase(packet->get_tx_id());
    }
  }

  // Send disconnect
  {
    auto tx_id = generate_uuid();
    if (!client->send<packet_type::disconnect>(tx_id))
      raise_exception("Disconnection packet send failed");
    if (!monitor->monitor(1s))
      raise_exception("Disconnection response packet not arrived");
    client->recv();
  }
}

void vm_stop(const std::string &url, const std::string &name) {
  std::string vm_id = url + ":" + name;
  auto it = vm_states.find(vm_id);
  if (it != vm_states.end()) {
    it->second->stop->stop();
  }
}

void handle_call_function(std::shared_ptr<packet_t> pkt,
                          std::shared_ptr<broker_client_t> client,
                          std::shared_ptr<vm_state_t> vm_state) {
  if (pkt->headers->size() != 2) {
    error("[VM] Invalid header");
    return;
  }
  tx_id_t &tx_id = *pkt->headers->at<string_t>(0);
  string_t &opname = *pkt->headers->at<string_t>(1);
  if (!pkt->body) {
    client->send<packet_type::respond_execute, false>(tx_id, 0, "Invalid body");
    return;
  }
  if (vm_state->operators.find(opname) == vm_state->operators.end()) {
    client->send<packet_type::respond_execute, false>(
        tx_id, 0, "Unknown operator: " + opname);
    return;
  }

  // Run
  auto op = vm_state->operators.at(opname);
  auto init_result = op->initialize(pkt->body->at("in"));
  if (init_result.has_value()) {
    client->send<packet_type::respond_execute, false>(
        tx_id, 0, init_result.value().reason);
    return;
  }
  for (size_t i = 0;; i++) {
    auto out = op->step();
    if (out.index() == 0) {
      auto ok_out = std::get<0>(out);
      client->send<packet_type::respond_execute, true>(tx_id, i, ok_out.finish,
                                                       ok_out.val);
      if (ok_out.finish)
        break;
    } else {
      auto err_out = std::get<1>(out);
      client->send<packet_type::respond_execute, false>(tx_id, i,
                                                        err_out.reason);
      break;
    }
  }
}

void handle_define_component(std::shared_ptr<packet_t> pkt,
                             std::shared_ptr<broker_client_t> client,
                             std::shared_ptr<vm_state_t> vm_state,
                             std::set<tx_id_t> &expected_responses) {
  if (pkt->headers->size() != 2) {
    error("[VM] Invalid header");
    return;
  }
  tx_id_t &tx_id = *pkt->headers->at<string_t>(0);
  string_t &comptype = *pkt->headers->at<string_t>(1);

  if (vm_state->factories.find(comptype) == vm_state->factories.end()) {
    client->send<packet_type::respond_execute, false>(
        tx_id, 0, "Unknown operator: " + comptype);
    return;
  }
  auto &factory = vm_state->factories.find(comptype)->second;

  if (!pkt->body) {
    client->send<packet_type::respond_execute, false>(tx_id, 0, "Invalid body");
    return;
  }
  string_t &compname = *pkt->body->at<string_t>("name");
  if (vm_state->factories.find(compname) != vm_state->factories.end()) {
    client->send<packet_type::respond_execute, false>(
        tx_id, 0, "Component already exists: " + compname);
    return;
  }

  // Create
  auto comp_opt = factory(pkt->body->at("in"));
  if (comp_opt.index() == 1) {
    client->send<packet_type::respond_execute, false>(
        tx_id, 0, std::get<1>(comp_opt).reason);
    return;
  }
  auto comp = std::get<0>(comp_opt);

  // Send subscriptions (unsubscribe)
  {
    auto tx_id = generate_uuid();
    client->send<packet_type::subscribe, instruction_type::delete_component>(
        tx_id, compname);
    expected_responses.insert(tx_id);
  }
  // Send subscriptions (operators)
  for (auto [opname, _] : comp->get_operators()) {
    auto tx_id = generate_uuid();
    client->send<packet_type::subscribe, instruction_type::call_method>(
        tx_id, compname, opname);
    expected_responses.insert(tx_id);
  }

  // register component
  vm_state->components.insert_or_assign(compname, comp);
  client->send<packet_type::respond_execute, true>(tx_id, 0, true,
                                                   ailoy::create<map_t>());
}

void handle_delete_component(std::shared_ptr<packet_t> pkt,
                             std::shared_ptr<broker_client_t> client,
                             std::shared_ptr<vm_state_t> vm_state,
                             std::set<tx_id_t> &expected_responses) {
  if (pkt->headers->size() != 2) {
    error("[VM] Invalid header");
    return;
  }
  tx_id_t &tx_id = *pkt->headers->at<string_t>(0);
  string_t &compname = *pkt->headers->at<string_t>(1);

  if (vm_state->components.find(compname) == vm_state->components.end()) {
    client->send<packet_type::respond_execute, false>(
        tx_id, 0, "Component not exists: " + compname);
    return;
  }

  // Erase component
  auto comp = vm_state->components.at(compname);
  vm_state->components.erase(compname);

  // Send unsubscriptions (operators)
  for (auto [opname, _] : comp->get_operators()) {
    auto tx_id = generate_uuid();
    client->send<packet_type::unsubscribe, instruction_type::call_method>(
        tx_id, compname, opname);
    expected_responses.insert(tx_id);
  }
  // Send unsubscriptions (unsubscribe)
  {
    auto tx_id = generate_uuid();
    client->send<packet_type::unsubscribe, instruction_type::delete_component>(
        tx_id, compname);
    expected_responses.insert(tx_id);
  }

  // Response
  client->send<packet_type::respond_execute, true>(tx_id, 0, true,
                                                   ailoy::create<map_t>());
}

void handle_call_method(std::shared_ptr<packet_t> pkt,
                        std::shared_ptr<broker_client_t> client,
                        std::shared_ptr<vm_state_t> vm_state) {
  if (pkt->headers->size() != 3) {
    error("[VM] Invalid header");
    return;
  }
  tx_id_t &tx_id = *pkt->headers->at<string_t>(0);
  string_t &compname = *pkt->headers->at<string_t>(1);
  string_t &opname = *pkt->headers->at<string_t>(2);
  if (!pkt->body) {
    client->send<packet_type::respond_execute, false>(tx_id, 0, "Invalid body");
    return;
  }

  if (vm_state->components.find(compname) == vm_state->components.end()) {
    client->send<packet_type::respond_execute, false>(
        tx_id, 0, "Component not exists: " + compname);
    return;
  }
  auto comp = vm_state->components.at(compname);
  auto op = comp->get_operator(opname);
  if (!op) {
    client->send<packet_type::respond_execute, false>(
        tx_id, 0, "Method not exists: " + compname + "." + opname);
    return;
  }

  // Run
  auto init_result = op->initialize(pkt->body->at("in"));
  if (init_result.has_value()) {
    client->send<packet_type::respond_execute, false>(
        tx_id, 0, init_result.value().reason);
    return;
  }
  for (size_t i = 0;; i++) {
    auto out = op->step();
    if (out.index() == 0) {
      auto ok_out = std::get<0>(out);
      client->send<packet_type::respond_execute, true>(tx_id, i, ok_out.finish,
                                                       ok_out.val);
      if (ok_out.finish)
        break;
    } else {
      auto err_out = std::get<1>(out);
      client->send<packet_type::respond_execute, false>(tx_id, i,
                                                        err_out.reason);
      break;
    }
  }
}

} // namespace ailoy
