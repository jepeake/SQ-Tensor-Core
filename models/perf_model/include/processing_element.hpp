#pragma once
#include "tile.hpp"
#include "weight_memory.hpp"
#include "stats.hpp"
#include "config.hpp"
#include <vector>
#include <cmath>


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


namespace perf_model {


class ProcessingElement {
private:
    size_t tile_size;
    PEStats stats;
    size_t adder_tree_width;
    
public:
    explicit ProcessingElement(size_t ts);
    Tile<int32_t> mpGEMM(const std::vector<Tile<uint8_t>>& weight_tiles, 
                         const Tile<int16_t>& activation_tile,
                         size_t num_bits,
                         int16_t activation_threshold = 0);

    const PEStats& get_stats() const { return stats; }
    void reset_stats() { stats.clear(); }

    void set_simulated_total_cycles(size_t cycles) { stats.total_cycles = cycles; }
    void calculate_adder_tree_width(double expected_weight_sparsity, double expected_activation_sparsity);
    
    // Methods for Synchronisation Stats
    void add_stall_cycles(size_t cycles) { stats.stall_cycles += cycles; }
    void add_barrier_wait(size_t cycles) { 
        stats.barrier_waits++; 
        stats.barrier_wait_cycles += cycles; 
    }
    void add_fifo_wait_cycles(size_t cycles) { stats.fifo_wait_cycles += cycles; }
    void set_max_fifo_occupancy(size_t occupancy) { 
        stats.max_fifo_occupancy = std::max(stats.max_fifo_occupancy, occupancy); 
    }
};


} // namespace perf_model