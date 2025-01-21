#pragma once
#include "weight_memory.hpp"
#include "pe_array.hpp"
#include <memory>

namespace spmpGEMM {

class SIMDEngine {
private:
    std::unique_ptr<WeightMemory> weight_mem;
    std::unique_ptr<PEArray> pe_array;
    size_t matrix_rows;
    size_t matrix_cols;
    size_t tile_size;
    size_t num_pes;
    SystemStats system_stats;
    
public:
    explicit SIMDEngine(const std::string& weight_file, size_t num_processing_elements = 4);
    Tile<int32_t> compute(const std::vector<int16_t>& activations, int16_t activation_threshold = 0);
    const SystemStats& getStats() const { return system_stats; }
};

} // namespace spmpGEMM