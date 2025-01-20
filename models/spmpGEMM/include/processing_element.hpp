#pragma once
#include "tile.hpp"
#include "weight_memory.hpp"
#include <vector>

namespace spmpGEMM {

class ProcessingElement {
private:
    size_t tile_size;
    
public:
    explicit ProcessingElement(size_t ts);
    Tile<int32_t> mpGEMM(const std::vector<Tile<uint8_t>>& weight_tiles, 
                         const Tile<int16_t>& activation_tile,
                         size_t num_bits);
};

} // namespace spmpGEMM