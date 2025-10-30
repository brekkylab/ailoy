#pragma once

#include "module.hpp"
#include "value.hpp"

namespace ailoy {

value_or_error_t http_request_op(std::shared_ptr<const value_t> inputs);

} // namespace ailoy
