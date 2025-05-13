#pragma once
#include "weight_memory.hpp"
#include "pe_array.hpp"
#include "performance_metrics.hpp"
#include <memory>


//   █████████  █████ ██████   ██████ ██████████      ██████████                      ███                     
//  ███░░░░░███░░███ ░░██████ ██████ ░░███░░░░███    ░░███░░░░░█                     ░░░                      
// ░███    ░░░  ░███  ░███░█████░███  ░███   ░░███    ░███  █ ░  ████████    ███████ ████  ████████    ██████ 
// ░░█████████  ░███  ░███░░███ ░███  ░███    ░███    ░██████   ░░███░░███  ███░░███░░███ ░░███░░███  ███░░███
//  ░░░░░░░░███ ░███  ░███ ░░░  ░███  ░███    ░███    ░███░░█    ░███ ░███ ░███ ░███ ░███  ░███ ░███ ░███████ 
//  ███    ░███ ░███  ░███      ░███  ░███    ███     ░███ ░   █ ░███ ░███ ░███ ░███ ░███  ░███ ░███ ░███░░░  
// ░░█████████  █████ █████     █████ ██████████      ██████████ ████ █████░░███████ █████ ████ █████░░██████ 
//  ░░░░░░░░░  ░░░░░ ░░░░░     ░░░░░ ░░░░░░░░░░      ░░░░░░░░░░ ░░░░ ░░░░░  ░░░░░███░░░░░ ░░░░ ░░░░░  ░░░░░░  
//                                                                          ███ ░███                          
//                                                                         ░░██████                           
//                                                                          ░░░░░░                            


namespace perf_model {


class SIMDEngine {
private:
    std::unique_ptr<WeightMemory> weight_mem;
    std::unique_ptr<PEArray> pe_array;
    size_t matrix_rows;
    size_t matrix_cols;
    size_t tile_size;
    size_t num_pes;
    size_t num_matmuls;
    SystemStats system_stats;
    
    // Helper Method to Perform a Single Matrix Multiplication
    Tile<int32_t> performSingleMatrixMultiply(const std::vector<int16_t>& activations, int16_t activation_threshold);
    
    // Different Accumulation Mode Implementations
    Tile<int32_t> performRowWiseAccumulation(
        const std::vector<std::vector<Tile<int16_t>>>& activation_tiles,
        size_t num_row_tiles,
        size_t num_col_tiles,
        int16_t activation_threshold);
        
    Tile<int32_t> performLocalPEAccumulation(
        const std::vector<std::vector<Tile<int16_t>>>& activation_tiles,
        size_t num_row_tiles,
        size_t num_col_tiles,
        int16_t activation_threshold);
    
public:
    explicit SIMDEngine(const std::string& weight_file);
    Tile<int32_t> compute(const std::vector<int16_t>& activations, int16_t activation_threshold = 0);
    const SystemStats& getStats() const { return system_stats; }
    
    size_t getMatrixRows() const { return matrix_rows; }
    size_t getMatrixCols() const { return matrix_cols; }
    size_t getTileSize() const { return tile_size; }
    size_t getNumPEs() const { return num_pes; }
    size_t getNumMatMuls() const { return num_matmuls; }
    
    PerformanceMetrics getPerformanceMetrics(double clock_frequency_hz) const;
};


} // namespace perf_model