#pragma once
#include "processing_element.hpp"
#include "tile.hpp"
#include <vector>
#include <memory>

namespace spmpGEMM {

class PEArray {
private:
    std::vector<std::unique_ptr<ProcessingElement>> pes;
    size_t num_pes;
    size_t tile_size;

public:
    PEArray(size_t num_processing_elements, size_t ts) 
        : num_pes(num_processing_elements), tile_size(ts) {
        for (size_t i = 0; i < num_pes; i++) {
            pes.push_back(std::make_unique<ProcessingElement>(tile_size));
        }
    }

    // Process Tiles Sequentially. Organise Results as if Parallel.
    std::vector<Tile<int32_t>> processTiles(
        const std::vector<std::pair<std::vector<Tile<uint8_t>>, Tile<int16_t>>>& tiles,
        size_t num_bits,
        int16_t activation_threshold = 0) {
        
        std::vector<Tile<int32_t>> results;
        size_t total_tiles = tiles.size();

        if (total_tiles > num_pes) {
            throw std::runtime_error("Not Enough PEs for All Tiles");
        }
        
        // Process All Tiles in Parallel Using Separate PEs
        for (size_t i = 0; i < total_tiles; i++) {
            const auto& work_item = tiles[i];
            auto result = pes[i]->mpGEMM(
                work_item.first,
                work_item.second, 
                num_bits,
                activation_threshold
            );
            results.push_back(result);
        }

        return results;
    }

    std::vector<PEStats> getStats() const {
        std::vector<PEStats> all_stats;
        for (const auto& pe : pes) {
            all_stats.push_back(pe->get_stats());
        }
        return all_stats;
    }
};

} // namespace spmpGEMM 