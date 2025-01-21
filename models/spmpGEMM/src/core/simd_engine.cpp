#include "../include/simd_engine.hpp"
#include <iostream>
#include <iomanip>

namespace spmpGEMM {

SIMDEngine::SIMDEngine(const std::string& weight_file, size_t num_processing_elements) 
    : num_pes(num_processing_elements) {
    weight_mem = std::make_unique<WeightMemory>(weight_file);
    tile_size = weight_mem->getTileSize();
    pe_array = std::make_unique<PEArray>(num_pes, tile_size);
    matrix_rows = weight_mem->getNumRows();
    matrix_cols = weight_mem->getNumCols();
}

Tile<int32_t> SIMDEngine::compute(const std::vector<int16_t>& activations, int16_t activation_threshold) {
    // Clear previous stats
    system_stats.clear();
    
    Tile<int32_t> result(matrix_rows, matrix_cols);

    std::vector<std::vector<Tile<int16_t>>> activation_tiles; 
    size_t num_row_tiles = (matrix_rows + tile_size - 1) / tile_size;
    size_t num_col_tiles = (matrix_cols + tile_size - 1) / tile_size;
    
    // Initialise 2D Vector of Tiles
    activation_tiles.resize(num_row_tiles);
    for (auto& row : activation_tiles) {
        row.resize(num_col_tiles);
    }
    
    // Create Activation Tiles
    for (size_t tile_row = 0; tile_row < num_row_tiles; tile_row++) {
        for (size_t tile_col = 0; tile_col < num_col_tiles; tile_col++) {
            Tile<int16_t> tile(tile_size, tile_size);
            
            // Fill Tile with Activation Values
            for (size_t row = 0; row < tile_size; row++) {
                for (size_t col = 0; col < tile_size; col++) {
                    size_t global_row = tile_row * tile_size + row;
                    size_t global_col = tile_col * tile_size + col;
                    size_t idx = global_row * matrix_cols + global_col;
                    
                    if (global_row < matrix_rows && global_col < matrix_cols && idx < activations.size()) {
                        tile.at(row, col) = activations[idx];
                    } else {
                        tile.at(row, col) = 0; // Zero Padding for Edge Tiles
                    }
                }
            }
            activation_tiles[tile_row][tile_col] = tile;

            // --- Printing ---
            std::cout << "\nActivation Tile [" << tile_row << "," << tile_col << "]:" << std::endl;
            std::cout << "--------------------" << std::endl;
            for (size_t i = 0; i < tile_size; i++) {
                for (size_t j = 0; j < tile_size; j++) {
                    std::cout << std::setw(4) << tile.at(i, j) << " ";
                }
                std::cout << std::endl;
            }

            size_t tile_idx = tile_row * num_col_tiles + tile_col;
            for (size_t bit = 0; bit < weight_mem->getNumBits(); bit++) {
                Tile<uint8_t> weight_tile = weight_mem->getTile(bit, tile_idx);
                std::cout << "\nWeight Tile [" << tile_row << "," << tile_col << "] Bit " << bit << ":" << std::endl;
                std::cout << "--------------------" << std::endl;
                for (size_t i = 0; i < tile_size; i++) {
                    for (size_t j = 0; j < tile_size; j++) {
                        std::cout << std::setw(2) << (int)weight_tile.at(i, j) << " ";
                    }
                    std::cout << std::endl;
                }
            }
            // --- End Printing ---
        }
    }

    // Process Tiles with Simulated Parallelism
    for (size_t tile_row = 0; tile_row < num_row_tiles; tile_row++) {
        for (size_t tile_col = 0; tile_col < num_col_tiles; tile_col++) {
            Tile<int32_t> acc_result(tile_size, tile_size);
            
            // Prepare Tiles
            std::vector<std::pair<std::vector<Tile<uint8_t>>, Tile<int16_t>>> tiles;
            
            for (size_t k = 0; k < num_col_tiles; k++) {
                const auto& act_tile = activation_tiles[tile_row][k];
                size_t weight_tile_idx = k * num_col_tiles + tile_col;
                
                std::vector<Tile<uint8_t>> weight_tiles;
                for (size_t bit = 0; bit < weight_mem->getNumBits(); bit++) {
                    weight_tiles.push_back(weight_mem->getTile(bit, weight_tile_idx));
                }
                
                tiles.emplace_back(weight_tiles, act_tile);
            }
            
            // Process Tiles as if in Parallel
            auto partial_results = pe_array->processTiles(
                tiles, 
                weight_mem->getNumBits(),
                activation_threshold
            );
            
            // Accumulate Results
            for (const auto& partial : partial_results) {
                for (size_t i = 0; i < tile_size; i++) {
                    for (size_t j = 0; j < tile_size; j++) {
                        acc_result.at(i, j) += partial.at(i, j);
                    }
                }
            }
            
            // Copy to Final Result
            for (size_t i = 0; i < tile_size; i++) {
                for (size_t j = 0; j < tile_size; j++) {
                    size_t global_row = tile_row * tile_size + i;
                    size_t global_col = tile_col * tile_size + j;
                    
                    if (global_row < matrix_rows && global_col < matrix_cols) {
                        result.at(global_row, global_col) = acc_result.at(i, j);
                    }
                }
            }
        }
    }
    
    // After processing tiles, collect stats
    system_stats.pe_stats = pe_array->getStats();
    
    return result;
}

} // namespace spmpGEMM