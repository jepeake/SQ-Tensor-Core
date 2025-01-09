#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

namespace mpX {

class SIMDEngine;
class WeightMemory;
class ProcessingElement;

template<typename T>
struct Tile {
    size_t rows;
    size_t cols;
    std::vector<T> data;
    
    Tile() : rows(0), cols(0) {}
    Tile(size_t r, size_t c) : rows(r), cols(c), data(r * c) {}
    T& at(size_t row, size_t col) {
        return data[row * cols + col];
    }
    
    const T& at(size_t row, size_t col) const {
        return data[row * cols + col];
    }
};

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

class ProcessingElement {
private:
    size_t tile_size;
    
public:
    explicit ProcessingElement(size_t ts);
    Tile<int32_t> mpGEMM(const Tile<uint8_t>& weight_tile, const Tile<int16_t>& activation_tile);
};

class SIMDEngine {
private:
    std::unique_ptr<WeightMemory> weight_mem;
    std::unique_ptr<ProcessingElement> pe;
    size_t matrix_rows;
    size_t matrix_cols;
    size_t tile_size;
    
public:
    explicit SIMDEngine(const std::string& weight_file);
    Tile<int32_t> compute(const std::vector<int16_t>& activations);
};

} // namespace mpX
