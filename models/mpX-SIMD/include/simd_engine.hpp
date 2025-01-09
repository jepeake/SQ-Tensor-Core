#pragma once
#include "weight_memory.hpp"
#include "processing_element.hpp"
#include <memory>

namespace mpX {

class SIMDEngine {
private:
    std::unique_ptr<WeightMemory> weight_mem;
    std::unique_ptr<ProcessingElement> pe;
    size_t matrix_rows;
    size_t matrix_cols;
    size_t tile_size;
    
public:
    explicit SIMDEngine(const std::string& weight_file);
    Tile<int32_t> compute(const std::vector<int16_t>& activations);
};

} // namespace mpX