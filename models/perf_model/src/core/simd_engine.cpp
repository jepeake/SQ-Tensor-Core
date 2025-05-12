#include "../include/simd_engine.hpp"
#include "../include/config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <filesystem>


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


SIMDEngine::SIMDEngine(const std::string& weight_file) {
    std::filesystem::path config_path = std::filesystem::current_path() / "perf_model_config.json";
    if (!std::filesystem::exists(config_path)) {
        config_path = std::filesystem::current_path() / "src/core/perf_model_config.json";
    }
    config::loadConfig(config_path.string());

    weight_mem = std::make_unique<WeightMemory>(weight_file);
    tile_size = weight_mem->getTileSize();
    matrix_rows = weight_mem->getNumRows();
    matrix_cols = weight_mem->getNumCols();

    num_pes = config::num_pes;
    num_matmuls = config::num_matmuls;
    pe_array = std::make_unique<PEArray>(num_pes, tile_size);
    
    // Set Synchronisation Mode
    system_stats.sync_mode = static_cast<SynchronisationMode>(config::synchronization_mode);
    system_stats.fifo_depth = pe_array->getFIFODepth();
    system_stats.output_buffer_size = pe_array->getOutputBufferSize();
}

Tile<int32_t> SIMDEngine::compute(const std::vector<int16_t>& activations, int16_t activation_threshold) {
    std::cout << "Running " << num_matmuls << " matrix multiplications..." << std::endl;
    std::cout << "Using Accumulation Mode: " << config::accumulation_mode << std::endl;
    std::cout << "Using Synchronisation Mode: " << system_stats.getSyncModeDescription() << std::endl;
    
    if (system_stats.sync_mode == SynchronisationMode::GLOBAL_BARRIER_PER_BATCH) {
        std::cout << "Batch size: " << pe_array->getBatchSize() << std::endl;
    } else if (system_stats.sync_mode == SynchronisationMode::ASYNC_LOCAL_FIFO) {
        std::cout << "FIFO depth: " << system_stats.fifo_depth << std::endl;
    } else if (system_stats.sync_mode == SynchronisationMode::ASYNC_SHARED_BUFFER) {
        std::cout << "Output buffer size: " << system_stats.output_buffer_size << " elements" << std::endl;
    }

    system_stats.clear();
    
    // Set Synchronisation Mode
    system_stats.sync_mode = pe_array->getSyncMode();
    system_stats.fifo_depth = pe_array->getFIFODepth();
    system_stats.output_buffer_size = pe_array->getOutputBufferSize();
    
    Tile<int32_t> result(matrix_rows, matrix_cols);

    // Perform Single Matrix Multiplication to Get Actual Stats for a Single Operation
    result = performSingleMatrixMultiply(activations, activation_threshold);
    
    // Store Stats for Single Matrix Multiplication
    SystemStats single_matmul_stats = system_stats;
    
    // If We Have More Than One Matrix Multiplication, Scale Only the Cycle and Operation Counts
    // But We Don't Multiply the PE Stats Directly as That Would Incorrectly Scale Area
    if (num_matmuls > 1) {
        std::cout << "Scaling Cycle and Operation Counts for " << num_matmuls << " Matrix Multiplications..." << std::endl;
        
        // Scale the Total Cycle and Operation Counts
        system_stats.total_parallel_cycles *= num_matmuls;
        system_stats.total_parallel_mask_ops *= num_matmuls;
        system_stats.total_parallel_shifts *= num_matmuls;
        system_stats.total_parallel_additions *= num_matmuls;
        system_stats.global_barriers *= num_matmuls;
        system_stats.global_stalls *= num_matmuls;
        system_stats.batch_barriers *= num_matmuls;
        system_stats.fifo_overflows *= num_matmuls;
        
        // For Each PE's Stats, We Scale Only the Operation Counts but Not the Hardware Itself
        for (auto& pe_stat : system_stats.pe_stats) {
            pe_stat.total_cycles *= num_matmuls;
            pe_stat.total_mask_ops *= num_matmuls;
            pe_stat.total_shifts *= num_matmuls;
            pe_stat.total_additions *= num_matmuls;
            pe_stat.extra_adder_cycles *= num_matmuls;
            pe_stat.stall_cycles *= num_matmuls;
            pe_stat.barrier_waits *= num_matmuls;
            pe_stat.barrier_wait_cycles *= num_matmuls;
            pe_stat.fifo_wait_cycles *= num_matmuls;
            
            // We Don't Scale These as They Represent Instantaneous Hardware Properties
            // pe_stat.masking_operations stays the same
            // pe_stat.shifting_operations stays the same
            // pe_stat.addition_operations stays the same
            // pe_stat.adder_tree_width stays the same
            // pe_stat.max_adder_tree_inputs stays the same
            // pe_stat.max_fifo_occupancy stays the same
        }
    }
    
    return result;
}


// Helper Method to Perform Single Matrix Multiplication
// This Contains the Original Compute Method Logic
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
    
    if (config::accumulation_mode == 0) {
        // Original Row-Wise Accumulation (Mode 0)
        result = performRowWiseAccumulation(activation_tiles, num_row_tiles, num_col_tiles, activation_threshold);
    } else if (config::accumulation_mode == 1) {
        // New Local PE Accumulation (Mode 1)
        result = performLocalPEAccumulation(activation_tiles, num_row_tiles, num_col_tiles, activation_threshold);
    } else {
        std::cerr << "Invalid Accumulation Mode: " << config::accumulation_mode << ". Using Row-Wise." << std::endl;
        result = performRowWiseAccumulation(activation_tiles, num_row_tiles, num_col_tiles, activation_threshold);
    }

    // Print Sparsity and Barrier Statistics
    std::cout << "\n----- Performance Model Statistics -----" << std::endl;
    std::cout << "Synchronisation Mode: " << system_stats.getSyncModeDescription() << std::endl;
    std::cout << "Accumulation Mode: " << (config::accumulation_mode == 0 ? "Row-wise (0)" : "Local PE (1)") << std::endl;
    std::cout << "Expected Weight Sparsity: " << config::expected_weight_sparsity << std::endl;
    std::cout << "Expected Activation Sparsity: " << config::expected_activation_sparsity << std::endl;
    
    // Calculate and Report Actual Sparsity Observed
    size_t total_weight_elements = 0;
    size_t total_nonzero_weights = 0;
    size_t total_stall_cycles = 0;
    size_t total_barrier_wait_cycles = 0;
    size_t total_fifo_wait_cycles = 0;
    size_t max_fifo_occupancy = 0;
    
    for (auto& pe_stat : system_stats.pe_stats) {
        if (pe_stat.total_cycles > 0) {  // Only Count Active PEs
            // For Each PE, Estimate Total Elements and Non-Zero Elements
            size_t pe_max_elements = pe_stat.max_adder_tree_inputs;
            total_weight_elements += (tile_size * tile_size * weight_mem->getNumBits());
            total_nonzero_weights += pe_max_elements;
            
            // Collect Synchronisation Stats
            total_stall_cycles += pe_stat.stall_cycles;
            total_barrier_wait_cycles += pe_stat.barrier_wait_cycles;
            total_fifo_wait_cycles += pe_stat.fifo_wait_cycles;
            max_fifo_occupancy = std::max(max_fifo_occupancy, pe_stat.max_fifo_occupancy);
        }
    }
    
    double actual_sparsity = 1.0;
    if (total_weight_elements > 0) {
        actual_sparsity = 1.0 - (static_cast<double>(total_nonzero_weights) / total_weight_elements);
    }
    
    std::cout << "Observed Effective Sparsity: " << actual_sparsity << std::endl;
    std::cout << "Maximum Skew Between PEs: " << system_stats.max_skew_cycles << " cycles" << std::endl;
    
    // Print Synchronisation-Specific Stats
    switch (system_stats.sync_mode) {
        case SynchronisationMode::GLOBAL_STALLING:
            std::cout << "Total Stall Cycles: " << total_stall_cycles << std::endl;
            std::cout << "Average Stall Cycles per PE: " 
                      << (system_stats.pe_stats.size() > 0 ? 
                          static_cast<double>(total_stall_cycles) / system_stats.pe_stats.size() : 0) 
                      << std::endl;
            break;
            
        case SynchronisationMode::GLOBAL_BARRIER_PER_GEMM:
            std::cout << "Global Barriers Applied: " << system_stats.global_barriers << std::endl;
            std::cout << "Total Barrier Wait Cycles: " << total_barrier_wait_cycles << std::endl;
            std::cout << "Slowest PE Index: " << system_stats.slowest_pe_indices << std::endl;
            break;
            
        case SynchronisationMode::GLOBAL_BARRIER_PER_BATCH:
            std::cout << "Batch Size: " << pe_array->getBatchSize() << std::endl;
            std::cout << "Total Batch Barriers: " << system_stats.global_barriers << std::endl;
            std::cout << "Total Barrier Wait Cycles: " << total_barrier_wait_cycles << std::endl;
            break;
            
        case SynchronisationMode::ASYNC_LOCAL_FIFO:
            std::cout << "FIFO Depth: " << system_stats.fifo_depth << std::endl;
            std::cout << "Maximum FIFO Occupancy: " << max_fifo_occupancy << std::endl;
            std::cout << "Total FIFO Wait Cycles: " << total_fifo_wait_cycles << std::endl;
            break;
            
        case SynchronisationMode::ASYNC_SHARED_BUFFER:
            std::cout << "Output Buffer Size: " << system_stats.output_buffer_size << " elements" << std::endl;
            break;
    }
    
    std::cout << "--------------------------------------------" << std::endl;
    
    return result;
}


// Row-Wise Accumulation Method
Tile<int32_t> SIMDEngine::performRowWiseAccumulation(
    const std::vector<std::vector<Tile<int16_t>>>& activation_tiles,
    size_t num_row_tiles,
    size_t num_col_tiles,
    int16_t activation_threshold) {
        
    Tile<int32_t> result(matrix_rows, matrix_cols);

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

    // After Processing Tiles, Aggregate the Stats from All PEs
    system_stats.pe_stats = pe_array->getAllStats();

    // Record Global Barrier Information
    system_stats.global_barriers = pe_array->getGlobalBarriers();

    system_stats.calculateSystemStats();
    
    return result;
}


// Local PE Accumulation Method
Tile<int32_t> SIMDEngine::performLocalPEAccumulation(
    const std::vector<std::vector<Tile<int16_t>>>& activation_tiles,
    size_t num_row_tiles,
    size_t num_col_tiles,
    int16_t activation_threshold) {
    
    Tile<int32_t> result(matrix_rows, matrix_cols);
    
    // Create a Structure to Hold the Work Items for Each Output Tile
    // Each Output Tile Position (row, col) Maps to a List of Work Items (Weight and Activation Tiles)
    std::vector<std::vector<std::vector<std::pair<std::vector<Tile<uint8_t>>, Tile<int16_t>>>>> output_tile_work;
    output_tile_work.resize(num_row_tiles);
    for (auto& row : output_tile_work) {
        row.resize(num_col_tiles);
    }
    
    // Organize Work by Output Tile Position
    for (size_t tile_row = 0; tile_row < num_row_tiles; tile_row++) {
        for (size_t tile_col = 0; tile_col < num_col_tiles; tile_col++) {
            // For Each Output Tile, Collect All Input Tile Pairs Needed
            for (size_t k = 0; k < num_col_tiles; k++) {
                const auto& act_tile = activation_tiles[tile_row][k];
                size_t weight_tile_idx = k * num_col_tiles + tile_col;
                
                std::vector<Tile<uint8_t>> weight_tiles;
                for (size_t bit = 0; bit < weight_mem->getNumBits(); bit++) {
                    weight_tiles.push_back(weight_mem->getTile(bit, weight_tile_idx));
                }
                
                output_tile_work[tile_row][tile_col].emplace_back(weight_tiles, act_tile);
            }
        }
    }
    
    // Flatten the Output Tile Work Items into a 1D Vector for PE Assignment
    std::vector<std::vector<std::pair<std::vector<Tile<uint8_t>>, Tile<int16_t>>>> pe_work_items;
    for (size_t tile_row = 0; tile_row < num_row_tiles; tile_row++) {
        for (size_t tile_col = 0; tile_col < num_col_tiles; tile_col++) {
            pe_work_items.push_back(output_tile_work[tile_row][tile_col]);
        }
    }
    
    // Assign Output Tiles to PEs (Round-Robin if We Have More Output Tiles than PEs)
    size_t total_output_tiles = num_row_tiles * num_col_tiles;
    
    // Process Each Output Tile with a PE
    std::vector<Tile<int32_t>> output_tiles;
    for (size_t i = 0; i < total_output_tiles; i++) {
        size_t pe_index = i % num_pes;
        size_t tile_row = i / num_col_tiles;
        size_t tile_col = i % num_col_tiles;
        
        // Have the PE Process All the Work Items for This Output Tile
        Tile<int32_t> output_tile = pe_array->processOutputTile(
            pe_index,
            pe_work_items[i],
            weight_mem->getNumBits(),
            activation_threshold
        );
        
        output_tiles.push_back(output_tile);
        
        // Copy the Results to the Appropriate Position in the Final Result Matrix
        for (size_t local_row = 0; local_row < tile_size; local_row++) {
            for (size_t local_col = 0; local_col < tile_size; local_col++) {
                size_t global_row = tile_row * tile_size + local_row;
                size_t global_col = tile_col * tile_size + local_col;
                
                if (global_row < matrix_rows && global_col < matrix_cols) {
                    result.at(global_row, global_col) = output_tile.at(local_row, local_col);
                }
            }
        }
    }
    
    // After Processing Tiles, Aggregate the Stats from All PEs
    system_stats.pe_stats = pe_array->getAllStats();

    // Record Global Barrier Information
    system_stats.global_barriers = pe_array->getGlobalBarriers();

    system_stats.calculateSystemStats();
    
    return result;
}


// Performance Metrics Calculation Method
PerformanceMetrics SIMDEngine::getPerformanceMetrics(double clock_frequency_hz) const {
    PerformanceMetrics metrics;

    double clock_period_ns = 1e9 / clock_frequency_hz;

    // Overall Latency in ns - Scales with Number of Matrix Multiplications
    double total_cycles = static_cast<double>(system_stats.total_parallel_cycles);
    metrics.system_latency_ns = total_cycles * clock_period_ns;

    // Estimate Total Number of Tiles
    size_t num_tiles_dim = (matrix_rows + tile_size - 1) / tile_size;
    size_t total_tiles = num_tiles_dim * num_tiles_dim * num_tiles_dim;
    
    // Total MAC Operations per Tile Multiplication
    size_t macs_per_tile = tile_size * tile_size * tile_size;
    size_t total_MACs = total_tiles * macs_per_tile * num_matmuls;
    
    size_t total_FLOPs = 2 * total_MACs;

    // Calculate Throughput: Total FLOPs Divided by Total Run Time in Seconds
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

} // namespace perf_model