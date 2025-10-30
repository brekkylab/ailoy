#pragma once

#include <string>

namespace ailoy {

/**
 * @brief Start broker
 * @param url URL
 * @return Number of remaining connections
 */
size_t broker_start(const std::string &url);

/**
 * @brief Stop broker
 * @param url URL
 */
void broker_stop(const std::string &url);

} // namespace ailoy
