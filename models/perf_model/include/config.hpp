#pragma once
#include <string>

namespace perf_model {
namespace config {

extern int num_pes;
extern int num_matmuls;

void loadConfig(const std::string &filename);

void saveConfig(const std::string &filename);

} // namespace config
} // namespace perf_model