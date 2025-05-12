#include <optional>

#include <tinyexpr.h>

#include "module.hpp"

namespace ailoy {

std::variant<double, int> evaluate_expression(std::string expression) {
  int err;
  te_expr *expr = te_compile(expression.c_str(), {}, 0, &err);
  if (expr) {
    const double value = te_eval(expr);
    te_free(expr);
    return value;
  }
  return err;
}

value_or_error_t calculator_op(std::shared_ptr<const value_t> inputs) {
  // Get input parameters
  if (!inputs->is_type_of<map_t>())
    return error_output_t("[Calculator] inputs should be a type of map");
  auto input_map = inputs->as<map_t>();

  // Parse expression
  if (!input_map->contains("expression"))
    return error_output_t("[Calculator] Field expression not exists");
  if (!input_map->at("expression")->is_type_of<string_t>())
    return error_output_t("[Calculator] Field expression should be string");
  const std::string &expression = *input_map->at<string_t>("expression");

  auto value = evaluate_expression(expression);

  if (const double *pval = std::get_if<double>(&value)) {
    auto outputs = create<map_t>();
    outputs->insert_or_assign("value", create<double_t>(*pval));
    return outputs;
  }
  return error_output_t(
      std::format("Error near here in the expression:\n\t{}\n\t{}^", expression,
                  std::string(std::get<int>(value) - 1, ' ')));
}

} // namespace ailoy
