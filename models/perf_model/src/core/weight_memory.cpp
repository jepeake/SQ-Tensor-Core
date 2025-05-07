#include "weight_memory.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace perf_model {

WeightMemory::WeightMemory(const std::string& weight_file) {
    loadWeights(weight_file);
}

const Tile<uint8_t>& WeightMemory::getTile(size_t bit, size_t tile_idx) const {
    if (bit >= num_bits || tile_idx >= bit_matrices[bit].size()) {
        throw std::out_of_range("Invalid bit or tile index");
    }
    return bit_matrices[bit][tile_idx];
}

size_t WeightMemory::getNumBits() const { return num_bits; }
size_t WeightMemory::getTileSize() const { return tile_size; }
size_t WeightMemory::getNumRows() const { return matrix_rows; }
size_t WeightMemory::getNumCols() const { return matrix_cols; }

void WeightMemory::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open weight file: " + filename);
    }

    uint32_t rows, cols, bits, tsize;
    file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&bits), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&tsize), sizeof(uint32_t));

    try {
        num_bits = bits;
        tile_size = tsize;
        matrix_rows = rows;
        matrix_cols = cols;
        
        // Calculate No. of Tiles
        size_t num_row_tiles = (rows + tile_size - 1) / tile_size;
        size_t num_col_tiles = (cols + tile_size - 1) / tile_size;
        size_t num_tiles = num_row_tiles * num_col_tiles;
        
        bit_matrices.resize(num_bits);
        
        // Calculate Bytes per Tile
        size_t bits_per_tile = tile_size * tile_size;
        size_t bytes_per_tile = (bits_per_tile + 7) / 8;  // Round up to nearest byte
        
        std::vector<uint8_t> packed_tile(bytes_per_tile);
        
        // Create Tiles
        for (size_t i = 0; i < num_bits; i++) {
            bit_matrices[i].resize(num_tiles, Tile<uint8_t>(tile_size, tile_size));
            
            // Read Data for this Bit Layer
            for (size_t tile = 0; tile < num_tiles; tile++) {
                auto& current_tile = bit_matrices[i][tile];
                
                // Read Packed Bytes
                if (!file.read(reinterpret_cast<char*>(packed_tile.data()), bytes_per_tile)) {
                    throw std::runtime_error("Failed to read weight data");
                }
                
                // Unpack Bits into Tile
                size_t bit_position = 0;
                for (size_t row = 0; row < tile_size; row++) {
                    for (size_t col = 0; col < tile_size; col++) {
                        size_t byte_idx = bit_position / 8;
                        size_t bit_offset = 7 - (bit_position % 8);
                        current_tile.at(row, col) = (packed_tile[byte_idx] >> bit_offset) & 1;
                        bit_position++;
                    }
                }
            }
        }
        
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }
}

} // namespace perf_model