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
        
        for (size_t i = 0; i < total_tiles; i += num_pes) {
            size_t chunk_size = std::min(num_pes, total_tiles - i);
            std::vector<Tile<int32_t>> chunk_results;
            
            // Process Each Tile Sequentially
            for (size_t j = 0; j < chunk_size; j++) {
                const auto& work_item = tiles[i + j];
                auto result = pes[j]->mpGEMM(
                    work_item.first,
                    work_item.second,
                    num_bits,
                    activation_threshold
                );
                chunk_results.push_back(result);
            }
            
            // Collect Results as if Computed in Parallel
            results.insert(results.end(), chunk_results.begin(), chunk_results.end());
        }
        
        return results;
    }
};

} // namespace spmpGEMM 