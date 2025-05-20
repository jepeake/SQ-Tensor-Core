#include <stdlib.h>
#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vtop_processing_element_array.h"
#include <random>
#include <ctime>
#include <vector>
#include <iomanip>


//  ███████████                        ██████                                                                           
// ░░███░░░░░███                      ███░░███                                                                          
//  ░███    ░███  ██████  ████████   ░███ ░░░   ██████  ████████  █████████████    ██████   ████████    ██████   ██████ 
//  ░██████████  ███░░███░░███░░███ ███████    ███░░███░░███░░███░░███░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███
//  ░███░░░░░░  ░███████  ░███ ░░░ ░░░███░    ░███ ░███ ░███ ░░░  ░███ ░███ ░███   ███████  ░███ ░███ ░███ ░░░ ░███████ 
//  ░███        ░███░░░   ░███       ░███     ░███ ░███ ░███      ░███ ░███ ░███  ███░░███  ░███ ░███ ░███  ███░███░░░  
//  █████       ░░██████  █████      █████    ░░██████  █████     █████░███ █████░░████████ ████ █████░░██████ ░░██████ 
// ░░░░░         ░░░░░░  ░░░░░      ░░░░░      ░░░░░░  ░░░░░     ░░░░░ ░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  
//
//                                                                                                                                                                                                                                                                                                                                             
//  ███████████                   █████    █████                                  █████                                 
// ░█░░░███░░░█                  ░░███    ░░███                                  ░░███                                  
// ░   ░███  ░   ██████   █████  ███████   ░███████   ██████  ████████    ██████  ░███████                              
//     ░███     ███░░███ ███░░  ░░░███░    ░███░░███ ███░░███░░███░░███  ███░░███ ░███░░███                             
//     ░███    ░███████ ░░█████   ░███     ░███ ░███░███████  ░███ ░███ ░███ ░░░  ░███ ░███                             
//     ░███    ░███░░░   ░░░░███  ░███ ███ ░███ ░███░███░░░   ░███ ░███ ░███  ███ ░███ ░███                             
//     █████   ░░██████  ██████   ░░█████  ████████ ░░██████  ████ █████░░██████  ████ █████                            
//    ░░░░░     ░░░░░░  ░░░░░░     ░░░░░  ░░░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░  ░░░░ ░░░░░                             


struct SharedTestConfig {
    // Per-row weights (each row shares the same weights)
    uint8_t row_weights[4][4][4];        // [row][i][j]
    uint8_t row_bit_planes[4][4][4][4];  // [row][bp][i][j]
    
    // Per-column activations (each column shares the same activations)
    int8_t col_activations[4][4][4];     // [col][i][j]
    
    int32_t expected_results[4][4][4][4]; // [row][col][i][j]
};


void convert_weights_to_bit_planes(
    const uint8_t full_weights[4][4],
    uint8_t bit_planes[4][4][4]) {
    
    for (int bp = 0; bp < 4; bp++) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                bit_planes[bp][i][j] = (full_weights[i][j] >> bp) & 0x1;
            }
        }
    }
}


void calculate_expected_results(
    const uint8_t full_weights[4][4],
    const int8_t activations[4][4],
    int32_t expected_results[4][4]) {
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            expected_results[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                expected_results[i][j] += activations[i][k] * full_weights[k][j];
            }
        }
    }
}

// Display row-specific weight data
void display_row_weight_matrices(int row, const SharedTestConfig& data) {
    std::cout << "\n==== WEIGHT DATA FOR ROW " << row << " (SHARED BY ALL PEs IN THIS ROW) ====\n" << std::endl;
    
    std::cout << "Full Weight Matrix:" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "    ";
        for (int j = 0; j < 4; j++) {
            std::cout << std::setw(3) << (int)data.row_weights[row][i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nWeight Bit Planes:" << std::endl;
    for (int bp = 0; bp < 4; bp++) {
        std::cout << "Bit Plane " << bp << ":" << std::endl;
        for (int i = 0; i < 4; i++) {
            std::cout << "    ";
            for (int j = 0; j < 4; j++) {
                std::cout << (int)data.row_bit_planes[row][bp][i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "\n==== END ROW " << row << " WEIGHT DATA ====\n" << std::endl;
}


void display_col_activation_matrices(int col, const SharedTestConfig& data) {
    std::cout << "\n==== ACTIVATION DATA FOR COLUMN " << col << " (SHARED BY ALL PEs IN THIS COLUMN) ====\n" << std::endl;
    
    std::cout << "Activation Matrix:" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "    ";
        for (int j = 0; j < 4; j++) {
            std::cout << std::setw(4) << (int)data.col_activations[col][i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n==== END COLUMN " << col << " ACTIVATION DATA ====\n" << std::endl;
}


void display_expected_results(int row, int col, const SharedTestConfig& data) {
    std::cout << "\n==== EXPECTED RESULTS FOR PE [" << row << "][" << col << "] ====\n" << std::endl;
    
    std::cout << "Expected Results:" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "    ";
        for (int j = 0; j < 4; j++) {
            std::cout << std::setw(6) << data.expected_results[row][col][i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n==== END PE [" << row << "][" << col << "] EXPECTED RESULTS ====\n" << std::endl;
}

    
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    Vtop_processing_element_array* pe_array = new Vtop_processing_element_array;
    
    // Minimal Waveform Tracing - Only Record Critical Cycles
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    pe_array->trace(tfp, 99); 
    tfp->open("waveform_array_perf.vcd");
    
    // Constants
    const int GRID_HEIGHT = 4;
    const int GRID_WIDTH = 4;
    const int TILE_SIZE = 4;
    const int NUM_BIT_PLANES = 4;

    // Create Random Test Data for Design Exercise
    std::mt19937 rng(42); // Fixed Seed for Reproducibility
    std::uniform_int_distribution<int> weight_dist(0, 1);   // Binary Weights
    std::uniform_int_distribution<int> act_dist(-128, 127); // 8-bit Signed Range
    
    vluint64_t sim_time = 0;
    
    // Reset Sequence
    pe_array->clk = 0;
    pe_array->rst_n = 0;
    pe_array->pe_start = 0;
    pe_array->global_start = 0;
    pe_array->pe_activation_threshold = 0;
    pe_array->broadcast_enable = 0;
    pe_array->broadcast_weights = 0;
    pe_array->broadcast_activations = 0;
    pe_array->broadcast_all = 0;
    
    // Single Reset Cycle
    pe_array->clk = 0;
    pe_array->eval();
    pe_array->clk = 1;
    pe_array->eval();
    tfp->dump(sim_time++);
    
    pe_array->rst_n = 1;
    
    std::cout << "===== PERFORMANCE TESTBENCH: =====\n" << std::endl;
    std::cout << "Configuration: " << GRID_HEIGHT << "x" << GRID_WIDTH << " PE array, " 
              << TILE_SIZE << "x" << TILE_SIZE << " tiles, " 
              << NUM_BIT_PLANES << " bit planes\n" << std::endl;
    
    // Fill the Weight and Activation Buffers with Random Data
    for (int row = 0; row < GRID_HEIGHT; row++) {
        for (int bp = 0; bp < NUM_BIT_PLANES; bp++) {
            for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
                    pe_array->row_weight_buffer[row][bp][i][j] = weight_dist(rng);
                }
            }
        }
    }
    
    for (int col = 0; col < GRID_WIDTH; col++) {
        for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                pe_array->col_activation_buffer[col][i][j] = act_dist(rng);
            }
        }
    }
    
    // Track Essential Timestamps for Performance Measurement
    vluint64_t start_time = sim_time;
    
    // PHASE 1: Configure All PEs in One Cycle
    std::cout << "Phase 1: One-shot configuration" << std::endl;
    
    // Enable Broadcast Mode and All-at-Once Broadcast
    pe_array->broadcast_enable = 1;
    pe_array->broadcast_all = 1;
    
    // Single Cycle for Configuration
    pe_array->clk = 0;
    pe_array->eval();
    pe_array->clk = 1;
    pe_array->eval();
    tfp->dump(sim_time++);
    
    // Disable Broadcast
    pe_array->broadcast_enable = 0;
    pe_array->broadcast_all = 0;
    
    // PHASE 2: Computation
    std::cout << "Phase 2: Parallel Execution" << std::endl;
    
    // Start All PEs
    pe_array->global_start = 1;
    pe_array->clk = 0;
    pe_array->eval();
    pe_array->clk = 1;
    pe_array->eval();
    tfp->dump(sim_time++);
    
    // De-assert Global Start
    pe_array->global_start = 0;
    
    // Wait for Computation to Complete
    int computation_cycles = 0;
    int max_cycles = 100;
    
    while (!pe_array->all_pes_done && computation_cycles < max_cycles) {
        pe_array->clk = 0;
        pe_array->eval();
        pe_array->clk = 1;
        pe_array->eval();
        tfp->dump(sim_time++);
        computation_cycles++;
    }
    
    // Record End Time
    vluint64_t end_time = sim_time;
    
    // Performance Report
    std::cout << "\n===== PERFORMANCE RESULTS =====\n" << std::endl;
    std::cout << "Configuration Cycles: 1" << std::endl;
    std::cout << "Computation Cycles: " << computation_cycles << std::endl;
    std::cout << "Total Cycles: " << (end_time - start_time) << std::endl;
    std::cout << "Effective Throughput: " << 
        (GRID_HEIGHT * GRID_WIDTH * TILE_SIZE * TILE_SIZE * 2) / (end_time - start_time) << 
        " operations per cycle" << std::endl;
    
    // Clean Up
    tfp->close();
    delete tfp;
    delete pe_array;
    
    return 0;
} 