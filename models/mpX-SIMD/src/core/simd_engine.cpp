#include "../include/simd_engine.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <stdexcept>
#include <cstring>

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
    
    // Tile Activations into 4x4 tiles
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
    
    for (size_t tile_row = 0; tile_row < (matrix_rows + tile_size - 1) / tile_size; tile_row++) {
        for (size_t tile_col = 0; tile_col < num_act_tiles; tile_col++) {
            size_t tile_idx = tile_row * num_act_tiles + tile_col;
            
            std::cout << "\n\n=== Processing Tile [" << tile_row << "," << tile_col << "] ===" << std::endl;
            
            std::cout << "\nInput Activation Tile:" << std::endl;
            for (size_t i = 0; i < tile_size; i++) {
                std::cout << "    ";
                for (size_t j = 0; j < tile_size; j++) {
                    std::cout << std::setw(4) << activation_tiles[tile_col].at(i, j) << " ";
                }
                std::cout << std::endl;
            }
            
            // Process All Bit Planes in Parallel Using Separate PEs
            std::vector<Tile<int32_t>> pe_outputs(weight_mem->getNumBits());
            for (auto& tile : pe_outputs) {
                tile = Tile<int32_t>(tile_size, tile_size);
            }
        
            // Perform mpGEMM in Each PE
            for (size_t bit = 0; bit < weight_mem->getNumBits(); bit++) {
                pe_outputs[bit] = pe->mpGEMM(
                    weight_mem->getTile(bit, tile_idx),
                    activation_tiles[tile_col]
                );
            }   

            // --- Printing for Terminal ---

            std::cout << "\nParallel PE Processing:\n" << std::endl;
            std::cout << std::string(weight_mem->getNumBits() * (tile_size * 6 + 8), '-') << std::endl;

            for (size_t bit = 0; bit < weight_mem->getNumBits(); bit++) {
                std::cout << " PE" << bit << " (Bit " << bit << ")              ";
            }
            std::cout << std::endl;
            for (size_t row = 0; row < tile_size; row++) {
                for (size_t bit = 0; bit < weight_mem->getNumBits(); bit++) {
                    std::cout << " | ";
                    for (size_t col = 0; col < tile_size; col++) {
                        std::cout << std::setw(4) << pe_outputs[bit].at(row, col) << " ";
                    }
                }
                std::cout << std::endl;
            }
            std::cout << std::string(weight_mem->getNumBits() * (tile_size * 6 + 8), '-') << std::endl;
            std::cout << "\nAccumulator Output (with shifts):" << std::endl;
            Tile<int32_t> tile_result(tile_size, tile_size);
            for (size_t i = 0; i < tile_size; i++) {
                std::cout << "    ";
                for (size_t j = 0; j < tile_size; j++) {
                    int32_t acc = 0;
                    for (size_t bit = 0; bit < weight_mem->getNumBits(); bit++) {
                        acc += pe_outputs[bit].at(i, j) << bit;
                    }
                    tile_result.at(i, j) = acc;
                    std::cout << std::setw(6) << acc << " ";
                }
                std::cout << std::endl;
            }
            // --- End of Printing for Terminal ---
            
            result = tile_result;
        }
    }
    return result;
}

} // namespace mpX