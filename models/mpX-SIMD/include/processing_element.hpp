#pragma once
#include "tile.hpp"

namespace mpX {

class ProcessingElement {
private:
    size_t tile_size;
    
public:
    explicit ProcessingElement(size_t ts);
    Tile<int32_t> mpGEMM(const Tile<uint8_t>& weight_tile, 
                         const Tile<int16_t>& activation_tile);
};

} // namespace mpX