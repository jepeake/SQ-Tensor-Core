#pragma once
#include <vector>
#include <cstdint>

namespace perf_model {

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

} // namespace perf_model