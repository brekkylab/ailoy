#pragma once

#include "module.hpp"

namespace ailoy {

component_or_error_t
create_tvm_language_model_component(std::shared_ptr<const value_t> inputs);

} // namespace ailoy
