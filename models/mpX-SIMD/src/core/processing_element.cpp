#include "../include/processing_element.hpp"

namespace mpX {

ProcessingElement::ProcessingElement(size_t ts) : tile_size(ts) {}

Tile<int32_t> ProcessingElement::mpGEMM(
    const Tile<uint8_t>& weight_tile,
    const Tile<int16_t>& activation_tile) {
    Tile<int32_t> result(tile_size, tile_size);

    // std::cout << "Activation Tile:" << std::endl;
    // for (size_t a_i = 0; a_i < tile_size; a_i++) {
    //     for (size_t a_j = 0; a_j < tile_size; a_j++) {
    //         std::cout << activation_tile.at(a_i, a_j) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    // std::cout << "Weight Tile:" << std::endl;
    // for (size_t w_i = 0; w_i < tile_size; w_i++) {
    //     for (size_t w_j = 0; w_j < tile_size; w_j++) {
    //         std::cout << (int)weight_tile.at(w_i, w_j) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    // For Each Row of Result
    for (size_t i = 0; i < tile_size; i++) {
        // For Each Column of Result
        for (size_t j = 0; j < tile_size; j++) {
            int32_t sum = 0;
            for (size_t k = 0; k < tile_size; k++) { // Loop over Weight Column
                if (weight_tile.at(k, j) == 1) {
                    sum += activation_tile.at(i, k);
                }
            }
            // Scale
            result.at(i, j) = sum;
        }
    }

    // std::cout << "\nResult Matrix:" << std::endl;
    // for (size_t i = 0; i < tile_size; i++) {
    //     for (size_t j = 0; j < tile_size; j++) {
    //         std::cout << result.at(i, j) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    return result;
}

} // namespace mpX