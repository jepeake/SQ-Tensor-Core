#include "../include/processing_element.hpp"
#include <iostream>
#include <iomanip>

namespace spmpGEMM {

ProcessingElement::ProcessingElement(size_t ts) : tile_size(ts) {}

Tile<int32_t> ProcessingElement::mpGEMM(
    const std::vector<Tile<uint8_t>>& weight_tiles,
    const Tile<int16_t>& activation_tile,
    size_t num_bits) {
    
    Tile<int32_t> result(tile_size, tile_size);
    
    for (size_t bit = 0; bit < num_bits; bit++) { // For Each Bit Plane
        for (size_t i = 0; i < tile_size; i++) { // For Each Row in the Activation
            for (size_t j = 0; j < tile_size; j++) { // For Each Column in the Weight
                int32_t acc = 0;
                for (size_t k = 0; k < tile_size; k++) { // Loop Over Weight Elements
                    if (weight_tiles[bit].at(k, j) == 1) {
                        acc += activation_tile.at(i, k);
                    }
                }
                result.at(i, j) += acc << bit;
            }
        }
    }
    return result;
}

} // namespace spmpGEMM