#include "../include/processing_element.hpp"
#include <iostream>
#include <iomanip>

namespace panda {

ProcessingElement::ProcessingElement(size_t ts) : tile_size(ts) {
    stats.clear();
}

Tile<int32_t> ProcessingElement::mpGEMM(
    const std::vector<Tile<uint8_t>>& weight_tiles,
    const Tile<int16_t>& activation_tile,
    size_t num_bits,
    int16_t activation_threshold) {  
    
    Tile<int32_t> result(tile_size, tile_size);
    
    for (size_t i = 0; i < tile_size; i++) {
        for (size_t j = 0; j < tile_size; j++) {
            int32_t final_acc = 0;
            for (size_t k = 0; k < tile_size; k++) {

                // Check Activation Sparsity
                if (std::abs(activation_tile.at(i, k)) <= activation_threshold) {
                    continue;  // Skip Computation for Sparse Activations
                }
                
                // Mask
                // In Hardware, all masks occur in parallel across all weight-activation pairs, across all bit-planes
                stats.masking_operations = 1;
                std::vector<int32_t> selected_activations(num_bits, 0);
                for (size_t bit = 0; bit < num_bits; bit++) {
                    stats.total_mask_ops++;
                    if (weight_tiles[bit].at(k, j) == 1) {
                        selected_activations[bit] = activation_tile.at(i, k);
                    }
                }
                
                // Shift
                // In Hardware, all shifts occur in parallel (rewiring)
                stats.shifting_operations = 0;
                int32_t shifted_activation = 0;
                for (size_t bit = 0; bit < num_bits; bit++) {
                    stats.total_shifts++;
                    shifted_activation += (selected_activations[bit] << bit);
                }

                // Adder Tree Stats
                size_t max_values_to_add = tile_size * num_bits; 
                size_t adders_needed = max_values_to_add - 1; // N-1 adders needed for N inputs
                size_t adder_stages = static_cast<size_t>(std::ceil(std::log2(max_values_to_add))); // log2(N) stages

                stats.total_additions = adders_needed * (tile_size * tile_size);
                stats.addition_operations = adder_stages;
                
                // Add
                final_acc += shifted_activation;
                
            }
            result.at(i, j) = final_acc;
        }
    }

    stats.total_cycles = stats.masking_operations + 
                         stats.shifting_operations + 
                         stats.addition_operations;

    return result;
}

} // namespace panda