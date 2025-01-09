#include "hardware.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <stdexcept>
#include <cstring>

namespace mpX {

WeightMemory::WeightMemory(const std::string& weight_file) {
    loadWeights(weight_file);
}

const Tile<uint8_t>& WeightMemory::getTile(size_t bit, size_t tile_idx) const {
    return bit_matrices[bit][tile_idx];
}

size_t WeightMemory::getNumBits() const { return num_bits; }
size_t WeightMemory::getTileSize() const { return tile_size; }
size_t WeightMemory::getNumRows() const { return matrix_rows; }
size_t WeightMemory::getNumCols() const { return matrix_cols; }

void WeightMemory::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open weight file");
    }
    
    // Read Header
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
                            current_tile.at(row, col) = (packed_byte >> bit) & 1;
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

// Main function stays in cpp file
int main(int argc, char* argv[]) {
    try {
        std::string weight_file = "weight_bits.bin";  
        
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
                weight_file = argv[i + 1];
                i++;  
            }
        }

        if (!std::filesystem::exists(weight_file)) {
            std::string parent_path = "../" + weight_file;
            if (std::filesystem::exists(parent_path)) {
                weight_file = parent_path;
            } else {
                throw std::runtime_error("Weight File Not Found: " + weight_file);
            }
        }

        std::cout << "Loading weights from: " << weight_file << std::endl;
        mpX::SIMDEngine engine(weight_file);
        
        std::vector<int16_t> activations = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        };
        
        mpX::Tile<int32_t> hw_result = engine.compute(activations);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}