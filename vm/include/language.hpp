#pragma once

#include "module.hpp"

namespace ailoy {

/**
 * @brief Get language module with functionality of LLM and its applications
 */
std::shared_ptr<const module_t> get_language_module();

} // namespace ailoy
