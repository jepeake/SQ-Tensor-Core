#include <stdlib.h>
#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vtop_processing_element.h"
#include <random>
#include <ctime>


//  ███████████                                                        ███                     
// ░░███░░░░░███                                                      ░░░                      
//  ░███    ░███ ████████   ██████   ██████   ██████   █████   █████  ████  ████████    ███████
//  ░██████████ ░░███░░███ ███░░███ ███░░███ ███░░███ ███░░   ███░░  ░░███ ░░███░░███  ███░░███
//  ░███░░░░░░   ░███ ░░░ ░███ ░███░███ ░░░ ░███████ ░░█████ ░░█████  ░███  ░███ ░███ ░███ ░███
//  ░███         ░███     ░███ ░███░███  ███░███░░░   ░░░░███ ░░░░███ ░███  ░███ ░███ ░███ ░███
//  █████        █████    ░░██████ ░░██████ ░░██████  ██████  ██████  █████ ████ █████░░███████
// ░░░░░        ░░░░░      ░░░░░░   ░░░░░░   ░░░░░░  ░░░░░░  ░░░░░░  ░░░░░ ░░░░ ░░░░░  ░░░░░███
//                                                                                     ███ ░███
//                                                                                    ░░██████ 
//                                                                                     ░░░░░░  
//  ██████████ ████                                                █████                       
// ░░███░░░░░█░░███                                               ░░███                        
//  ░███  █ ░  ░███   ██████  █████████████    ██████  ████████   ███████                      
//  ░██████    ░███  ███░░███░░███░░███░░███  ███░░███░░███░░███ ░░░███░                       
//  ░███░░█    ░███ ░███████  ░███ ░███ ░███ ░███████  ░███ ░███   ░███                        
//  ░███ ░   █ ░███ ░███░░░   ░███ ░███ ░███ ░███░░░   ░███ ░███   ░███ ███                    
//  ██████████ █████░░██████  █████░███ █████░░██████  ████ █████  ░░█████                     
// ░░░░░░░░░░ ░░░░░  ░░░░░░  ░░░░░ ░░░ ░░░░░  ░░░░░░  ░░░░ ░░░░░    ░░░░░                      
                                                                                            
                                                                                            
                                                                                            
//  ███████████                   █████    █████                                  █████        
// ░█░░░███░░░█                  ░░███    ░░███                                  ░░███         
// ░   ░███  ░   ██████   █████  ███████   ░███████   ██████  ████████    ██████  ░███████     
//     ░███     ███░░███ ███░░  ░░░███░    ░███░░███ ███░░███░░███░░███  ███░░███ ░███░░███    
//     ░███    ░███████ ░░█████   ░███     ░███ ░███░███████  ░███ ░███ ░███ ░░░  ░███ ░███    
//     ░███    ░███░░░   ░░░░███  ░███ ███ ░███ ░███░███░░░   ░███ ░███ ░███  ███ ░███ ░███    
//     █████   ░░██████  ██████   ░░█████  ████████ ░░██████  ████ █████░░██████  ████ █████   
//    ░░░░░     ░░░░░░  ░░░░░░     ░░░░░  ░░░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░  ░░░░ ░░░░░    


void display_matrices_and_expected_results(
    int TILE_SIZE, 
    int NUM_BIT_PLANES,
    uint8_t* weight_tiles,  
    int8_t* activation_tile) { 
    
    std::cout << "\n==== TEST MATRIX DATA ====\n" << std::endl;
    
    std::cout << "Full Weight Matrix:" << std::endl;
    int full_weights[TILE_SIZE][TILE_SIZE];
    
    for (int i = 0; i < TILE_SIZE; i++) {
        for (int j = 0; j < TILE_SIZE; j++) {
            full_weights[i][j] = 0;
            for (int bp = 0; bp < NUM_BIT_PLANES; bp++) {
                uint8_t bit_val = weight_tiles[bp*TILE_SIZE*TILE_SIZE + i*TILE_SIZE + j];
                full_weights[i][j] += bit_val << bp;
            }
        }
    }
    
    for (int i = 0; i < TILE_SIZE; i++) {
        std::cout << "    ";
        for (int j = 0; j < TILE_SIZE; j++) {
            std::cout << full_weights[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    for (int bp = 0; bp < NUM_BIT_PLANES; bp++) {
        std::cout << "Weight Matrix (Bit Plane " << bp << "):" << std::endl;
        for (int i = 0; i < TILE_SIZE; i++) {
            std::cout << "    ";
            for (int j = 0; j < TILE_SIZE; j++) {
                std::cout << (int)weight_tiles[bp*TILE_SIZE*TILE_SIZE + i*TILE_SIZE + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    std::cout << "Activation Matrix:" << std::endl;
    for (int i = 0; i < TILE_SIZE; i++) {
        std::cout << "    ";
        for (int j = 0; j < TILE_SIZE; j++) {
            std::cout << (int)activation_tile[i*TILE_SIZE + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Standard Matrix Multiplication Result (activation × weight):" << std::endl;
    int32_t matmul_results[TILE_SIZE][TILE_SIZE];
    
    for (int i = 0; i < TILE_SIZE; i++) {
        for (int j = 0; j < TILE_SIZE; j++) {
            matmul_results[i][j] = 0;
        }
    }
    
    for (int i = 0; i < TILE_SIZE; i++) {
        for (int j = 0; j < TILE_SIZE; j++) {
            for (int k = 0; k < TILE_SIZE; k++) {
                int8_t act_val = activation_tile[i*TILE_SIZE + k];
                int32_t weight_val = full_weights[k][j];
                matmul_results[i][j] += act_val * weight_val;
            }
        }
    }
    
    for (int i = 0; i < TILE_SIZE; i++) {
        std::cout << "    ";
        for (int j = 0; j < TILE_SIZE; j++) {
            std::cout << matmul_results[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n==== END TEST DATA ====\n" << std::endl;
}

// Converts full weights into bit planes
void convert_weights_to_bit_planes(
    int TILE_SIZE,
    int NUM_BIT_PLANES,
    uint8_t* full_weights,
    uint8_t* bit_planes) {
    
    for (int bp = 0; bp < NUM_BIT_PLANES; bp++) {
        for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                uint8_t weight = full_weights[i*TILE_SIZE + j];
                bit_planes[bp*TILE_SIZE*TILE_SIZE + i*TILE_SIZE + j] = (weight >> bp) & 0x1;
            }
        }
    }
}

vluint64_t sim_time = 0;

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    Vtop_processing_element* pe = new Vtop_processing_element;
    
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    pe->trace(tfp, 99); 
    tfp->open("waveform.vcd");
    
    const int TILE_SIZE = 4;
    const int ACT_WIDTH = 8;
    const int WEIGHT_WIDTH = 1; 
    const int NUM_BIT_PLANES = 4;
    const int RESULT_WIDTH = 32;
    
    std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> act_dist(-128, 127);    // 8-bit signed range
    std::uniform_int_distribution<int> weight_dist(0, 15);     // 4-bit unsigned range
    
    pe->clk = 0;
    pe->rst_n = 0;
    pe->start = 0;
    pe->activation_threshold = 0;  
    
    // Reset Sequence
    for (int i = 0; i < 5; i++) {
        pe->clk = 0;
        pe->eval();
        pe->clk = 1;
        pe->eval();
        tfp->dump(sim_time++);
    }
    
    pe->rst_n = 1;
    pe->clk = 0;
    pe->eval();
    pe->clk = 1;
    pe->eval();
    tfp->dump(sim_time++);
    
    // Generate Random Full Weights 
    uint8_t full_weights[TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < TILE_SIZE; i++) {
        for (int j = 0; j < TILE_SIZE; j++) {
            full_weights[i*TILE_SIZE + j] = weight_dist(rng);
        }
    }
    
    // Convert Full Weights to Bit Planes
    uint8_t bit_planes[NUM_BIT_PLANES * TILE_SIZE * TILE_SIZE];
    convert_weights_to_bit_planes(TILE_SIZE, NUM_BIT_PLANES, full_weights, bit_planes);
    
    // Configure the Weight Bit Planes
    for (int bp = 0; bp < NUM_BIT_PLANES; bp++) {
        for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                pe->weight_tiles[bp][i][j] = bit_planes[bp*TILE_SIZE*TILE_SIZE + i*TILE_SIZE + j];
            }
        }
    }
    
    // Generate Random Activation Values
    int8_t activation_values[TILE_SIZE * TILE_SIZE];
    for (int i = 0; i < TILE_SIZE; i++) {
        for (int j = 0; j < TILE_SIZE; j++) {
            activation_values[i*TILE_SIZE + j] = act_dist(rng);
        }
    }
    
    // Configure the Activation Values
    for (int i = 0; i < TILE_SIZE; i++) {
        for (int j = 0; j < TILE_SIZE; j++) {
            pe->activation_tile[i][j] = activation_values[i*TILE_SIZE + j];
        }
    }
        
    display_matrices_and_expected_results(TILE_SIZE, NUM_BIT_PLANES, 
                                         bit_planes, 
                                         activation_values);
    
    // Start Processing
    pe->start = 1;
    pe->clk = 0;
    pe->eval();
    pe->clk = 1;
    pe->eval();
    tfp->dump(sim_time++);
    
    pe->start = 0;
    
    bool done = false;
    int max_cycles = 100; 
    int cycle_count = 0;
    
    while (!done && cycle_count < max_cycles) {
        pe->clk = 0;
        pe->eval();
        
        pe->clk = 1;
        pe->eval();
        tfp->dump(sim_time++);
        cycle_count++;
        
        if (pe->done) {
            done = true;
            std::cout << "\nDone Signal Detected at Time " << sim_time << " - Ending Simulation" << std::endl;
            
            std::cout << "Final Results:" << std::endl;
            for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
                    std::cout << "result_tile[" << i << "][" << j << "] = " 
                              << pe->result_tile[i][j] << std::endl;
                }
            }
            
            // Verify
            bool all_correct = true;
            for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
                    int32_t expected = 0;
                    for (int k = 0; k < TILE_SIZE; k++) {
                        int8_t act_val = activation_values[i*TILE_SIZE + k];
                        uint8_t weight_val = 0;
                        for (int bp = 0; bp < NUM_BIT_PLANES; bp++) {
                            weight_val += (bit_planes[bp*TILE_SIZE*TILE_SIZE + k*TILE_SIZE + j] << bp);
                        }
                        expected += act_val * weight_val;
                    }
                    
                    if (pe->result_tile[i][j] != expected) {
                        std::cout << "ERROR at [" << i << "][" << j << "]: Got " 
                                  << pe->result_tile[i][j] << ", expected " << expected << std::endl;
                        all_correct = false;
                    }
                }
            }
            
            std::cout << "\n=================================================================" << std::endl;
            if (all_correct) {
                std::cout << "All Results Matched Expected Values" << std::endl;
            } else {
                std::cout << "Some Results Did Not Match Expected Values" << std::endl;
            }
            std::cout << "Total Simulation Time: " << sim_time << " cycles" << std::endl;
            std::cout << "=================================================================\n" << std::endl;

            tfp->close();
            delete tfp;
            delete pe;
            
            std::cout << "Exiting Simulation" << std::endl;
            exit(all_correct ? 0 : 1);
        }
    }
    
    if (!done) {
        std::cout << "Simulation Timed Out After " << max_cycles << " Cycles" << std::endl;
    }
    
    std::cout << "Simulation " << (done ? "Completed Successfully" : "Timed Out") << std::endl;
    std::cout << "Total Simulation Time: " << sim_time << " cycles" << std::endl;
        
    tfp->close();
    delete tfp;
    delete pe;
    
    return done ? 0 : 1;
} 