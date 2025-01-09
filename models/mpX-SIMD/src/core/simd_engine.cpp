#include "../include/simd_engine.hpp"
#include <iostream>
#include <iomanip>

namespace mpX {

SIMDEngine::SIMDEngine(const std::string& weight_file) {
    weight_mem = std::make_unique<WeightMemory>(weight_file);
    tile_size = weight_mem->getTileSize();
    pe = std::make_unique<ProcessingElement>(tile_size);
    matrix_rows = weight_mem->getNumRows();
    matrix_cols = weight_mem->getNumCols();
}

Tile<int32_t> SIMDEngine::compute(const std::vector<int16_t>& activations) {
    Tile<int32_t> result(matrix_rows, matrix_cols);
    
    // Create Activation Tiles
    std::vector<Tile<int16_t>> activation_tiles;
    size_t num_act_tiles = (matrix_cols + tile_size - 1) / tile_size;
    
    for (size_t i = 0; i < num_act_tiles; i++) {
        Tile<int16_t> tile(tile_size, tile_size);
        for (size_t row = 0; row < tile_size; row++) {
            for (size_t col = 0; col < tile_size; col++) {
                size_t idx = i * tile_size * tile_size + row * tile_size + col;
                if (idx < activations.size()) {
                    tile.at(row, col) = activations[idx];
                }
            }
        }
        activation_tiles.push_back(tile);
    }
    
    // Process Tiles
    for (size_t tile_row = 0; tile_row < (matrix_rows + tile_size - 1) / tile_size; tile_row++) {
        for (size_t tile_col = 0; tile_col < num_act_tiles; tile_col++) {
            size_t tile_idx = tile_row * num_act_tiles + tile_col;
                        
            // Collect All Weight Tiles
            std::vector<Tile<uint8_t>> weight_tiles;
            for (size_t bit = 0; bit < weight_mem->getNumBits(); bit++) {
                weight_tiles.push_back(weight_mem->getTile(bit, tile_idx));
            }
            
            // Process mpGEMM
            result = pe->mpGEMM(weight_tiles, activation_tiles[tile_col], 
                               weight_mem->getNumBits());
        }
    }
    return result;
}

} // namespace mpX