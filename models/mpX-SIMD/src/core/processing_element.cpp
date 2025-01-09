#include "../include/processing_element.hpp"
#include <iostream>
#include <iomanip>

namespace mpX {

ProcessingElement::ProcessingElement(size_t ts) : tile_size(ts) {}

Tile<int32_t> ProcessingElement::mpGEMM(
    const std::vector<Tile<uint8_t>>& weight_tiles,
    const Tile<int16_t>& activation_tile,
    size_t num_bits) {
    
    Tile<int32_t> result(tile_size, tile_size);
    
    for (size_t i = 0; i < tile_size; i++) {
        for (size_t j = 0; j < tile_size; j++) {
            int32_t acc = 0;
            
            for (size_t bit = 0; bit < num_bits; bit++) { // For Each Bit Plane
                int32_t bit_plane_sum = 0;
                for (size_t k = 0; k < tile_size; k++) {
                    int16_t mask = -(weight_tiles[bit].at(k, j) & 1); // Create Bitmask
                    bit_plane_sum += activation_tile.at(i, k) & mask; // Apply Mask to Activation Row
                }   
                acc += bit_plane_sum << bit; // Accumulate Bit Plane Results with Appropriate Shifts
            }
            result.at(i, j) = acc;
        }
    }
    
    std::cout << "Result Matrix:" << std::endl;
    for (size_t i = 0; i < tile_size; i++) {
        for (size_t j = 0; j < tile_size; j++) {
            std::cout << std::setw(8) << result.at(i, j) << " ";
        }
        std::cout << std::endl;
    }
    return result;
}

} // namespace mpX