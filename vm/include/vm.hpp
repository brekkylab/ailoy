/**
 * @file vm.hpp
 * @brief Task executor on request from broker
 * @details
 *
 * VM performs tasks defined as `operator_t`s and `component_t`s
 * on the local machine(that the runtime runs on) or on a separate machine.
 *
 * ## `module_t`
 *
 * a `module_t` contains several `operator_t`s and `component_t`s,
 * and can make a VM can use those by being loaded into the VM.
 * (details are in module.hpp)
 *
 * ```cpp
 * {
 *   std::shared_ptr<const pakky::module_t> mods[] =
 *       {pakky::get_default_module()};
 *   std::thread t_vm =
 *       std::thread([url, &mods] { pakky::vm_start(url, mods, "default_vm");
 * });
 * }
 * ```
 */
#pragma once

#include <initializer_list>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "module.hpp"
#include "thread.hpp"

namespace ailoy {

/**
 * @brief Start VM
 * @param url  URL
 * @param mods Modules to import
 * @param name Name of VM
 */
void vm_start(const std::string &url,
              std::span<std::shared_ptr<const module_t>> mods,
              const std::string &name = "default_vm");

/**
 * @brief Stop VM
 * @param url  URL
 * @param name name of VM
 */
void vm_stop(const std::string &url, const std::string &name = "default_vm");

} // namespace ailoy
