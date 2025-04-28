#pragma once
#include <string>

namespace panda {
namespace config {

extern int num_pes;
extern int num_matmuls;

void loadConfig(const std::string &filename);

void saveConfig(const std::string &filename);

} // namespace config
} // namespace panda