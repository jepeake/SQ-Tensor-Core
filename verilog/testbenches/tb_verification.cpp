#include <stdlib.h>
#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vtop_processing_element_array.h"
#include <random>
#include <ctime>
#include <vector>
#include <iomanip>


//  █████   █████                     ███     ██████   ███                      █████     ███                     
// ░░███   ░░███                     ░░░     ███░░███ ░░░                      ░░███     ░░░                      
//  ░███    ░███   ██████  ████████  ████   ░███ ░░░  ████   ██████   ██████   ███████   ████   ██████  ████████  
//  ░███    ░███  ███░░███░░███░░███░░███  ███████   ░░███  ███░░███ ░░░░░███ ░░░███░   ░░███  ███░░███░░███░░███ 
//  ░░███   ███  ░███████  ░███ ░░░  ░███ ░░░███░     ░███ ░███ ░░░   ███████   ░███     ░███ ░███ ░███ ░███ ░███ 
//   ░░░█████░   ░███░░░   ░███      ░███   ░███      ░███ ░███  ███ ███░░███   ░███ ███ ░███ ░███ ░███ ░███ ░███ 
//     ░░███     ░░██████  █████     █████  █████     █████░░██████ ░░████████  ░░█████  █████░░██████  ████ █████
//      ░░░       ░░░░░░  ░░░░░     ░░░░░  ░░░░░     ░░░░░  ░░░░░░   ░░░░░░░░    ░░░░░  ░░░░░  ░░░░░░  ░░░░ ░░░░░ 
//
//                                                                                                                                                                                                                                                                                                                                        
//  ███████████                   █████    █████                                  █████                           
// ░█░░░███░░░█                  ░░███    ░░███                                  ░░███                            
// ░   ░███  ░   ██████   █████  ███████   ░███████   ██████  ████████    ██████  ░███████                        
//     ░███     ███░░███ ███░░  ░░░███░    ░███░░███ ███░░███░░███░░███  ███░░███ ░███░░███                       
//     ░███    ░███████ ░░█████   ░███     ░███ ░███░███████  ░███ ░███ ░███ ░░░  ░███ ░███                       
//     ░███    ░███░░░   ░░░░███  ░███ ███ ░███ ░███░███░░░   ░███ ░███ ░███  ███ ░███ ░███                       
//     █████   ░░██████  ██████   ░░█████  ████████ ░░██████  ████ █████░░██████  ████ █████                      
//    ░░░░░     ░░░░░░  ░░░░░░     ░░░░░  ░░░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░  ░░░░ ░░░░░                                                                                                        ░░░░░░  


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


int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    Vtop_processing_element_array* pe_array = new Vtop_processing_element_array;
    
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    pe_array->trace(tfp, 10); 
    tfp->open("waveform_array_verify.vcd");
    
    const int GRID_HEIGHT = 4;
    const int GRID_WIDTH = 4;
    const int TILE_SIZE = 4;
    const int NUM_BIT_PLANES = 4;
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> act_dist(-128, 127);    // 8-bit signed range
    std::uniform_int_distribution<int> weight_dist(0, 15);     // 4-bit unsigned range
    
    SharedTestConfig shared_config;
    vluint64_t sim_time = 0;
    
    std::cout << "\n" << std::endl;
    std::cout << "===== NUMERICAL VERIFICATION: =====\n" << std::endl;
    std::cout << "Generating test data and calculating expected results...\n" << std::endl;
    
    // Generate Random Weights for Each Row (Shared By All PEs in That Row)
    for (int row = 0; row < GRID_HEIGHT; row++) {
        for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                shared_config.row_weights[row][i][j] = weight_dist(rng);
            }
        }
        
        // Convert Weights to Bit Planes for This Row
        convert_weights_to_bit_planes(
            shared_config.row_weights[row],
            shared_config.row_bit_planes[row]
        );
    }
    
    // Generate Random Activations for Each Column (Shared by All PEs in That Column)
    for (int col = 0; col < GRID_WIDTH; col++) {
        for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                shared_config.col_activations[col][i][j] = act_dist(rng);
            }
        }
    }
    
    // Calculate Expected Results for Each PE
    for (int row = 0; row < GRID_HEIGHT; row++) {
        for (int col = 0; col < GRID_WIDTH; col++) {
            calculate_expected_results(
                shared_config.row_weights[row],
                shared_config.col_activations[col],
                shared_config.expected_results[row][col]
            );
        }
    }
    
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
    
    pe_array->clk = 0;
    pe_array->eval();
    pe_array->clk = 1;
    pe_array->eval();
    tfp->dump(sim_time++);
    
    pe_array->rst_n = 1;
    
    // Pre-load Buffers
    for (int row = 0; row < GRID_HEIGHT; row++) {
        for (int bp = 0; bp < NUM_BIT_PLANES; bp++) {
            for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
                    pe_array->row_weight_buffer[row][bp][i][j] = shared_config.row_bit_planes[row][bp][i][j];
                }
            }
        }
    }
    
    for (int col = 0; col < GRID_WIDTH; col++) {
        for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                pe_array->col_activation_buffer[col][i][j] = shared_config.col_activations[col][i][j];
            }
        }
    }
    
    // Configure All PEs at Once
    pe_array->broadcast_enable = 1;
    pe_array->broadcast_all = 1;
    
    pe_array->clk = 0;
    pe_array->eval();
    pe_array->clk = 1;
    pe_array->eval();
    tfp->dump(sim_time++);
    
    pe_array->broadcast_enable = 0;
    pe_array->broadcast_all = 0;
    
    // Start Computation
    pe_array->global_start = 1;
    pe_array->clk = 0;
    pe_array->eval();
    pe_array->clk = 1;
    pe_array->eval();
    tfp->dump(sim_time++);
    
    pe_array->global_start = 0;
    
    // Wait for Computation to Complete
    int exe_cycles = 0;
    while (!pe_array->all_pes_done && exe_cycles < 10) {
        pe_array->clk = 0;
        pe_array->eval();
        pe_array->clk = 1;
        pe_array->eval();
        tfp->dump(sim_time++);
        exe_cycles++;
    }
    
    std::cout << "Computation Completed in " << exe_cycles << " Cycles" << std::endl;
    std::cout << "Verifying Results...\n" << std::endl;
    
    // Verify Results from All PEs
    std::vector<std::pair<int, int>> pe_coords;
    for (int row = 0; row < GRID_HEIGHT; row++) {
        for (int col = 0; col < GRID_WIDTH; col++) {
            pe_coords.push_back(std::make_pair(row, col));
        }
    }
    
    const int num_pes_to_test = GRID_HEIGHT * GRID_WIDTH;
    std::vector<bool> pe_verified(num_pes_to_test, false);
    bool all_verified = true;
    
    // Batch Verification for All PEs
    for (int pe_idx = 0; pe_idx < num_pes_to_test; pe_idx++) {
        int row = pe_coords[pe_idx].first;
        int col = pe_coords[pe_idx].second;
        
        // Select the PE to Read Its Results
        pe_array->pe_row_select = row;
        pe_array->pe_col_select = col;
        
        // Read the Result
        pe_array->clk = 0;
        pe_array->eval();
        pe_array->clk = 1;
        pe_array->eval();
        
        // Verify the Results - Only Report Errors
        bool all_correct = true;
        int errors = 0;
        
        for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                int32_t expected = shared_config.expected_results[row][col][i][j];
                int32_t actual = static_cast<int32_t>(pe_array->pe_result_tile[i][j]);
                
                if (actual != expected) {
                    if (errors < 3) {  
                        std::cout << "ERROR in PE [" << row << "][" << col << "] at position ["
                                  << i << "][" << j << "]: Got " << actual
                                  << ", expected " << expected << std::endl;
                    }
                    errors++;
                    all_correct = false;
                }
            }
        }
        
        if (all_correct) {
            pe_verified[pe_idx] = true;
        } else {
            std::cout << "PE [" << row << "][" << col << "] has " << errors << " errors" << std::endl;
            all_verified = false;
        }
    }
    
    // Verification Summary
    std::cout << "\n===== VERIFICATION SUMMARY =====\n" << std::endl;
    
    // Display Verification Grid
    for (int row = 0; row < GRID_HEIGHT; row++) {
        for (int col = 0; col < GRID_WIDTH; col++) {
            int idx = row * GRID_WIDTH + col;
            std::cout << (pe_verified[idx] ? "✓" : "✗");
        }
        std::cout << " ← Row " << row << std::endl;
    }
    
    std::cout << "\nOverall Verification: " << (all_verified ? "PASSED ✓" : "FAILED ✗") << std::endl;
    std::cout << "\n" << std::endl;
    
    // Clean Up
    tfp->close();
    delete tfp;
    delete pe_array;
    
    return all_verified ? 0 : 1;
} 