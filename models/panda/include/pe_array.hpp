#pragma once
#include "processing_element.hpp"
#include "tile.hpp"
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath> 

namespace panda {

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

    std::vector<Tile<int32_t>> processTiles(
        const std::vector<std::pair<std::vector<Tile<uint8_t>>, Tile<int16_t>>>& tiles,
        size_t num_bits,
        int16_t activation_threshold = 0) {
        
        size_t total_tiles = tiles.size();
        std::vector<Tile<int32_t>> results(total_tiles);
        
        // Timing Parameters
        const size_t L_mask = 1;  // Masking Stage Latency
        size_t L_add = static_cast<size_t>(std::ceil(std::log2(tile_size)));
        const size_t L_total = L_mask + L_add;  // Total Latency per Tile

        // Track Finish Time (Cycle Count) for Each PE
        std::vector<size_t> pe_last_finish(num_pes, 0);
        
        // Schedule Each Tile Round-Robin on the Fixed Set of PEs
        for (size_t i = 0; i < total_tiles; i++) {
            size_t pe_index = i % num_pes;
            size_t start_cycle = i / num_pes;  // New Job Starts Each Cycle on a Given PE
            size_t finish_time = start_cycle + L_total;
            
            const auto& work_item = tiles[i];
            
            // Synchronously Perform the Matrix Multiply on the Assigned PE.
            results[i] = pes[pe_index]->mpGEMM(
                work_item.first,
                work_item.second, 
                num_bits,
                activation_threshold
            );
            
            // Update the Finish Time for the Assigned PE if Later than the Previous Finish Time.
            if (finish_time > pe_last_finish[pe_index]) {
                pe_last_finish[pe_index] = finish_time;
            }
        }
        
        // Update Each PE's Stats to Reflect the Simulated Pipelined Total Cycles.
        for (size_t i = 0; i < num_pes; i++) {
            if (pe_last_finish[i] > 0) {
                pes[i]->set_simulated_total_cycles(pe_last_finish[i]);
            } else {
                // Ensure that Idle PEs are Recorded with Zero Cycles.
                pes[i]->set_simulated_total_cycles(0);
            }
        }
        
        return results;
    }

    // Retrieve the Vector of Stats for Each Processing Element.
    std::vector<PEStats> getAllStats() const {
        std::vector<PEStats> all_stats;
        all_stats.reserve(num_pes);
        for (const auto &pe : pes) {
            all_stats.push_back(pe->get_stats());
        }
        return all_stats;
    }
};

} // namespace panda 