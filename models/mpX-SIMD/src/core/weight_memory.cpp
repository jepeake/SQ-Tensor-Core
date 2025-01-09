#include "weight_memory.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace mpX {

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
        size_t bytes_per_tile = (bits_per_tile + 7) / 8; 
        
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
                for (size_t byte_idx = 0; byte_idx < bytes_per_tile; byte_idx++) {
                    uint8_t packed_byte = packed_tile[byte_idx];
                    for (size_t bit = 0; bit < 8; bit++) {
                        size_t linear_idx = byte_idx * 8 + bit;
                        if (linear_idx < bits_per_tile) {
                            size_t row = linear_idx / tile_size;
                            size_t col = linear_idx % tile_size;
                            current_tile.at(row, col) = (packed_byte >> (7 - bit)) & 1;
                        }
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

} // namespace mpX