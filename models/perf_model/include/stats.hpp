#pragma once
#include <cstddef>
#include <vector>
#include <algorithm>
#include <queue>
#include <string>


//   █████████   █████               █████           
//  ███░░░░░███ ░░███               ░░███            
// ░███    ░░░  ███████    ██████   ███████    █████ 
// ░░█████████ ░░░███░    ░░░░░███ ░░░███░    ███░░  
//  ░░░░░░░░███  ░███      ███████   ░███    ░░█████ 
//  ███    ░███  ░███ ███ ███░░███   ░███ ███ ░░░░███
// ░░█████████   ░░█████ ░░████████  ░░█████  ██████ 
//  ░░░░░░░░░     ░░░░░   ░░░░░░░░    ░░░░░  ░░░░░░  
 

namespace perf_model {

// Enum for Synchronisation Modes
enum class SynchronisationMode {
    GLOBAL_STALLING = 0,          // Mode 0: Global Stalling
    GLOBAL_BARRIER_PER_GEMM = 1,  // Mode 1: Global Barrier per GEMM
    GLOBAL_BARRIER_PER_BATCH = 2, // Mode 2: Global Barrier per Batch
    ASYNC_LOCAL_FIFO = 3,         // Mode 3: Asynchronous Tiles with Local FIFOs
    ASYNC_SHARED_BUFFER = 4       // Mode 4: Asynchronous Tiles with Shared Output Buffer
};

struct PEStats {
    size_t total_cycles;

    // Per-Cycle Stats (Operations that Happen in Parallel Count as a Single Operation)
    size_t masking_operations{0};     
    size_t shifting_operations{0};     
    size_t addition_operations{0};     
    
    // Total Stats (Counts Regardless of Parallelism)
    size_t total_mask_ops{0};         
    size_t total_shifts{0};           
    size_t total_additions{0};       
    
    // Sparsity & Adder Tree Stats
    size_t max_adder_tree_inputs{0};  // Maximum Number of Inputs to Adder Tree Across All Computations
    size_t adder_tree_width{0};       // Width of the Adder Tree (Max Inputs per Cycle)
    size_t extra_adder_cycles{0};     // Extra Cycles Needed When Inputs > Adder_Tree_Width
    
    // Synchronisation Stats
    size_t stall_cycles{0};           // Cycles Spent Stalled (for Mode 0)
    size_t barrier_waits{0};          // Number of Times PE Had to Wait at a Barrier
    size_t barrier_wait_cycles{0};    // Total Cycles Spent Waiting at Barriers
    size_t fifo_wait_cycles{0};       // Cycles Spent Waiting for FIFO Space (Mode 3)
    std::queue<size_t> fifo;          // Simulated FIFO for Mode 3
    size_t max_fifo_occupancy{0};     // Maximum Number of Items in FIFO
    
    void clear() {
        total_cycles = 0;
        masking_operations = 0;
        shifting_operations = 0;
        addition_operations = 0;
        total_mask_ops = 0;
        total_shifts = 0;
        total_additions = 0;
        max_adder_tree_inputs = 0;
        adder_tree_width = 0;
        extra_adder_cycles = 0;
        stall_cycles = 0;
        barrier_waits = 0;
        barrier_wait_cycles = 0;
        fifo_wait_cycles = 0;
        
        // Clear the FIFO Queue
        std::queue<size_t> empty;
        std::swap(fifo, empty);
        max_fifo_occupancy = 0;
    }
};

struct SystemStats {
    std::vector<PEStats> pe_stats;
    
    // System-Wide Parallel Stats
    size_t total_parallel_cycles{0};
    size_t total_parallel_mask_ops{0};
    size_t total_parallel_shifts{0};
    size_t total_parallel_additions{0};
    
    // Global Barrier Stats
    size_t global_barriers{0};
    size_t slowest_pe_indices{0};
    
    // Synchronisation Stats
    SynchronisationMode sync_mode{SynchronisationMode::GLOBAL_BARRIER_PER_GEMM};
    size_t global_stalls{0};             // Count of Global Stalls (Mode 0)
    size_t batch_barriers{0};            // Number of Batch Barriers (Mode 2)
    size_t output_buffer_size{0};        // Size of Shared Output Buffer (Mode 4)
    size_t fifo_depth{0};                // Configured FIFO Depth (Mode 3)
    size_t fifo_overflows{0};            // Count of FIFO Overflows (Mode 3)
    size_t max_skew_cycles{0};           // Maximum Skew Observed Between PEs
    
    void clear() {
        pe_stats.clear();
        total_parallel_cycles = 0;
        total_parallel_mask_ops = 0;
        total_parallel_shifts = 0;
        total_parallel_additions = 0;
        global_barriers = 0;
        slowest_pe_indices = 0;
        global_stalls = 0;
        batch_barriers = 0;
        output_buffer_size = 0;
        fifo_depth = 0;
        fifo_overflows = 0;
        max_skew_cycles = 0;
    }

    void calculateSystemStats() {
        if (pe_stats.empty()) return;
        
        // In Parallel Execution with Barriers, We Take the Maximum Cycles Across All PEs
        total_parallel_cycles = 0;
        total_parallel_mask_ops = 0;
        total_parallel_shifts = 0;
        total_parallel_additions = 0;

        // Find the PE with the Maximum Cycle Count (Slowest PE)
        size_t max_cycles = 0;
        size_t slowest_pe = 0;
        
        for (size_t i = 0; i < pe_stats.size(); ++i) {
            const auto& pe = pe_stats[i];
            if (pe.total_cycles > max_cycles) {
                max_cycles = pe.total_cycles;
                slowest_pe = i;
            }
            
            total_parallel_mask_ops = std::max(total_parallel_mask_ops, pe.masking_operations);
            total_parallel_shifts = std::max(total_parallel_shifts, pe.shifting_operations);
            total_parallel_additions = std::max(total_parallel_additions, pe.addition_operations);
        }
        
        total_parallel_cycles = max_cycles;
        slowest_pe_indices = slowest_pe;
        
        // Calculate Maximum Skew Between PEs
        if (!pe_stats.empty()) {
            size_t min_cycles = pe_stats[0].total_cycles;
            for (const auto& pe : pe_stats) {
                min_cycles = std::min(min_cycles, pe.total_cycles);
            }
            max_skew_cycles = max_cycles - min_cycles;
        }
    }
    
    // Get a String Description of the Current Synchronisation Mode
    std::string getSyncModeDescription() const {
        switch (sync_mode) {
            case SynchronisationMode::GLOBAL_STALLING:
                return "Mode 0: Global Stalling";
            case SynchronisationMode::GLOBAL_BARRIER_PER_GEMM:
                return "Mode 1: Global Barrier per GEMM";
            case SynchronisationMode::GLOBAL_BARRIER_PER_BATCH:
                return "Mode 2: Global Barrier per Batch";
            case SynchronisationMode::ASYNC_LOCAL_FIFO:
                return "Mode 3: Asynchronous Tiles with Local FIFOs";
            case SynchronisationMode::ASYNC_SHARED_BUFFER:
                return "Mode 4: Asynchronous Tiles with Shared Output Buffer";
            default:
                return "Unknown Synchronisation Mode";
        }
    }
};


} // namespace perf_model