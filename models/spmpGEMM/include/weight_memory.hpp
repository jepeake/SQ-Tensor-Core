#pragma once
#include "tile.hpp"
#include <string>
#include <vector>

namespace spmpGEMM {

class WeightMemory {
private:
    std::vector<std::vector<Tile<uint8_t>>> bit_matrices;
    size_t num_bits;
    size_t tile_size;
    size_t matrix_rows;
    size_t matrix_cols;
    
    void loadWeights(const std::string& filename);

public:
    explicit WeightMemory(const std::string& weight_file);
    const Tile<uint8_t>& getTile(size_t bit, size_t tile_idx) const;
    size_t getNumBits() const;
    size_t getTileSize() const;
    size_t getNumRows() const;
    size_t getNumCols() const;
};

} // namespace spmpGEMM