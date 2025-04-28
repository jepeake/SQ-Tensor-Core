#include "../include/simd_engine.hpp"
#include "../include/config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <filesystem>

namespace panda {

SIMDEngine::SIMDEngine(const std::string& weight_file) {
    std::filesystem::path config_path = std::filesystem::current_path() / "panda_config.json";
    if (!std::filesystem::exists(config_path)) {
        config_path = std::filesystem::current_path() / "src/core/panda_config.json";
    }
    config::loadConfig(config_path.string());

    weight_mem = std::make_unique<WeightMemory>(weight_file);
    tile_size = weight_mem->getTileSize();
    matrix_rows = weight_mem->getNumRows();
    matrix_cols = weight_mem->getNumCols();

    num_pes = config::num_pes;
    num_matmuls = config::num_matmuls;
    pe_array = std::make_unique<PEArray>(num_pes, tile_size);
}

Tile<int32_t> SIMDEngine::compute(const std::vector<int16_t>& activations, int16_t activation_threshold) {
    std::cout << "Running " << num_matmuls << " matrix multiplications..." << std::endl;

    system_stats.clear();
    
    Tile<int32_t> result(matrix_rows, matrix_cols);

    // Perform the first matrix multiplication to get the actual stats for a single operation
    result = performSingleMatrixMultiply(activations, activation_threshold);
    
    // Store stats from a single matrix multiply
    SystemStats single_matmul_stats = system_stats;
    
    // If we have more than one matrix multiply, scale only the cycle and operation counts
    // but we don't multiply the PE stats directly as that would incorrectly scale area
    if (num_matmuls > 1) {
        std::cout << "Scaling cycle and operation counts for " << num_matmuls << " matrix multiplies..." << std::endl;
        
        // Scale the total cycle and operation counts
        system_stats.total_parallel_cycles *= num_matmuls;
        system_stats.total_parallel_mask_ops *= num_matmuls;
        system_stats.total_parallel_shifts *= num_matmuls;
        system_stats.total_parallel_additions *= num_matmuls;
        
        // For each PE's stats, we scale only the operation counts but not the hardware itself
        for (auto& pe_stat : system_stats.pe_stats) {
            pe_stat.total_cycles *= num_matmuls;
            pe_stat.total_mask_ops *= num_matmuls;
            pe_stat.total_shifts *= num_matmuls;
            pe_stat.total_additions *= num_matmuls;
            
            // We don't scale these as they represent instantaneous hardware properties
            // pe_stat.masking_operations stays the same
            // pe_stat.shifting_operations stays the same
            // pe_stat.addition_operations stays the same
        }
    }
    
    return result;
}

// Helper method to perform a single matrix multiplication
// This contains the original compute method logic
Tile<int32_t> SIMDEngine::performSingleMatrixMultiply(const std::vector<int16_t>& activations, int16_t activation_threshold) {
    Tile<int32_t> result(matrix_rows, matrix_cols);

    std::vector<std::vector<Tile<int16_t>>> activation_tiles; 
    size_t num_row_tiles = (matrix_rows + tile_size - 1) / tile_size;
    size_t num_col_tiles = (matrix_cols + tile_size - 1) / tile_size;

    size_t total_tiles = num_row_tiles * num_col_tiles;

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


            // ----- Printing -----
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
            // ----- End Printing -----
        }
    }


    // ----- Get Tile Pairs for Each PE -----

    std::vector<std::pair<std::vector<Tile<uint8_t>>, Tile<int16_t>>> tiles;
    for (size_t tile_row = 0; tile_row < num_row_tiles; tile_row++) {
        for (size_t tile_col = 0; tile_col < num_col_tiles; tile_col++) {
            
            for (size_t k = 0; k < num_col_tiles; k++) {
                const auto& act_tile = activation_tiles[tile_row][k];
                size_t weight_tile_idx = k * num_col_tiles + tile_col;
                
                std::vector<Tile<uint8_t>> weight_tiles;
                for (size_t bit = 0; bit < weight_mem->getNumBits(); bit++) {
                    weight_tiles.push_back(weight_mem->getTile(bit, weight_tile_idx));
                }
                
                tiles.emplace_back(weight_tiles, act_tile);
            }
        }
    }


    // ----- Parallel PE Processing -----
             
    // Process Tiles as if in Parallel
    auto partial_results = pe_array->processTiles(
        tiles, 
        weight_mem->getNumBits(),
        activation_threshold
    );
    

    // ----- Row-Wise Adder Tree -----
    
    // Each Row of Tiles Sums in Parallel

    // For Each Position in Result Matrix
    for (size_t tile_row = 0; tile_row < num_row_tiles; tile_row++) { 
        for (size_t local_row = 0; local_row < tile_size; local_row++) { 
            for (size_t global_col = 0; global_col < num_col_tiles * tile_size; global_col++) { 

                // Find Position
                size_t global_row = tile_row * tile_size + local_row; 
                if (global_row >= matrix_rows) continue;                                         
                size_t tile_col = global_col / tile_size;
                size_t local_col = global_col % tile_size; 
                
                // Accumulate Partial Products
                int32_t sum = 0;
                size_t base_idx = (tile_row * num_col_tiles + tile_col) * num_col_tiles;
                for (size_t k = 0; k < num_col_tiles; k++) {
                    sum += partial_results[base_idx + k].at(local_row, local_col); 
                }

                // Store Result 
                result.at(global_row, global_col) = sum; 
            }
        }
    }

    // After processing tiles, aggregate the stats from all PEs
    // (Even if some PEs never received work - they will be included with zeroed stats)
    system_stats.pe_stats = pe_array->getAllStats();

    system_stats.calculateSystemStats();
    
    return result;
}

PerformanceMetrics SIMDEngine::getPerformanceMetrics(double clock_frequency_hz) const {
    PerformanceMetrics metrics;

    double clock_period_ns = 1e9 / clock_frequency_hz;

    // Overall Latency in ns - scales with number of matrix multiplications
    double total_cycles = static_cast<double>(system_stats.total_parallel_cycles);
    metrics.system_latency_ns = total_cycles * clock_period_ns;

    // Estimate Total Number of Tiles
    size_t num_tiles_dim = (matrix_rows + tile_size - 1) / tile_size;
    size_t total_tiles = num_tiles_dim * num_tiles_dim * num_tiles_dim;
    
    // Total MAC Operations per Tile Multiplication
    size_t macs_per_tile = tile_size * tile_size * tile_size;
    size_t total_MACs = total_tiles * macs_per_tile * num_matmuls;
    
    size_t total_FLOPs = 2 * total_MACs;

    // Calculate Throughput: Total FLOPs divided by Total Run Time in Seconds
    double system_time_sec = metrics.system_latency_ns * 1e-9;
    metrics.throughput_ops = (system_time_sec > 0) ? (static_cast<double>(total_FLOPs) / system_time_sec) : 0.0;
    
    // Calculate Operations per Cycle: Total FLOPs divided by Total Cycles
    metrics.ops_per_cycle = (total_cycles > 0) ? (static_cast<double>(total_FLOPs) / total_cycles) : 0.0;

    // Effective Memory Traffic - scales with number of matrix multiplications
    // -- Activations are loaded once per (tile_row, k) pair.
    // -- Weight tiles are assumed to be part of a constant weight matrix and loaded once for the (k, tile_col) pair.
    // -- Result tiles are written once.
    size_t num_row_tiles = (matrix_rows + tile_size - 1) / tile_size;
    size_t num_col_tiles = (matrix_cols + tile_size - 1) / tile_size;
    size_t activation_bytes = num_row_tiles * num_col_tiles * tile_size * tile_size * sizeof(int16_t) * num_matmuls;
    size_t weights_bytes = num_col_tiles * num_col_tiles * tile_size * tile_size * weight_mem->getNumBits() * sizeof(uint8_t) * num_matmuls;
    size_t result_bytes = num_row_tiles * num_col_tiles * tile_size * tile_size * sizeof(int32_t) * num_matmuls;
    size_t effective_total_bytes = activation_bytes + weights_bytes + result_bytes;

    metrics.memory_bandwidth_bytes_per_sec = (system_time_sec > 0) ? (static_cast<double>(effective_total_bytes) / system_time_sec) : 0.0;
    metrics.arithmetic_intensity = (effective_total_bytes > 0) ? (static_cast<double>(total_FLOPs) / effective_total_bytes) : 0.0;

    // Calculate Hardware Costs
    const double ADDER_ENERGY_PJ = 0.03;        // Energy per 8-bit adder in pJ
    const double ADDER_AREA_UM2 = 36.0;         // Area per 8-bit adder in μm²
    const double MASK_ENERGY_PJ = 0.0012;       // Energy per transmission gate + AND in pJ
    const double MASK_AREA_UM2 = 1.4;           // Area per transmission gate + AND in μm²

    // Get total operations across all matrix multiplications for energy calculation
    size_t total_additions = 0;
    size_t total_mask_ops = 0;
    
    for (const auto& pe_stat : system_stats.pe_stats) {
        total_additions += pe_stat.total_additions;
        total_mask_ops += pe_stat.total_mask_ops;
    }
    
    // Energy scales with the total number of operations (all matrix multiplications)
    metrics.adder_energy_pj = total_additions * ADDER_ENERGY_PJ;
    metrics.mask_energy_pj = total_mask_ops * MASK_ENERGY_PJ;
    metrics.total_energy_pj = metrics.adder_energy_pj + metrics.mask_energy_pj;
    
    // Area calculation - does NOT scale with num_matmuls
    // We need to divide the total operations by num_matmuls to get the hardware area
    metrics.adder_area_um2 = (num_matmuls > 0) ? (total_additions / num_matmuls) * ADDER_AREA_UM2 : 0;
    metrics.mask_area_um2 = (num_matmuls > 0) ? (total_mask_ops / num_matmuls) * MASK_AREA_UM2 : 0;
    metrics.total_area_um2 = metrics.adder_area_um2 + metrics.mask_area_um2;
    
    return metrics;
}

} // namespace panda