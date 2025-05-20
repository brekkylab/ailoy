#include "module.hpp"

#include "calculator.hpp"
#include "http_request.hpp"

namespace ailoy {

static std::shared_ptr<module_t> default_module = ailoy::create<module_t>();

std::shared_ptr<const module_t> get_default_module() {
  auto &default_ops = default_module->ops;
  auto &default_factories = default_module->factories;

  // Add Operator HTTP Request
  if (default_ops.find("http_request") == default_ops.end()) {
    default_ops.insert_or_assign("http_request",
                                 create<instant_operator_t>(http_request_op));
  }

  // Add Operator Calculator
  if (!default_ops.contains("calculator")) {
    default_ops.insert_or_assign("calculator",
                                 create<instant_operator_t>(calculator_op));
  }

  return default_module;
}

} // namespace ailoy
