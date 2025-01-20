#include "../include/processing_element.hpp"
#include <iostream>
#include <iomanip>

namespace spmpGEMM {

ProcessingElement::ProcessingElement(size_t ts) : tile_size(ts) {}

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
                std::vector<int32_t> masked_values(num_bits, 0);
                for (size_t bit = 0; bit < num_bits; bit++) {
                    if (weight_tiles[bit].at(k, j) == 1) {
                        masked_values[bit] = activation_tile.at(i, k);
                    }
                }
                
                // Shift
                int32_t shifted_activation = 0;
                for (size_t bit = 0; bit < num_bits; bit++) {
                    shifted_activation += (masked_values[bit] << bit);
                }
                
                // Add
                final_acc += shifted_activation;
            }
            result.at(i, j) = final_acc;
        }
    }
    return result;
}

} // namespace spmpGEMM