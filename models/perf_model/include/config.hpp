#pragma once
#include <string>

namespace perf_model {
namespace config {

extern int num_pes;
extern int num_matmuls;
extern double expected_weight_sparsity;
extern double expected_activation_sparsity;
extern int adder_tree_width;
extern int accumulation_mode;         
extern int synchronisation_mode;      
extern int batch_size;                

void loadConfig(const std::string &filename);

void saveConfig(const std::string &filename);


} // namespace config
} // namespace perf_model