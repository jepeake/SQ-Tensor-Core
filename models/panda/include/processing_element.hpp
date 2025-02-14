#pragma once
#include "tile.hpp"
#include "weight_memory.hpp"
#include "stats.hpp"
#include <vector>

namespace panda {

class ProcessingElement {
private:
    size_t tile_size;
    PEStats stats;
    
public:
    explicit ProcessingElement(size_t ts);
    Tile<int32_t> mpGEMM(const std::vector<Tile<uint8_t>>& weight_tiles, 
                         const Tile<int16_t>& activation_tile,
                         size_t num_bits,
                         int16_t activation_threshold = 0);

    const PEStats& get_stats() const { return stats; }
    void reset_stats() { stats.clear(); }
};

} // namespace panda