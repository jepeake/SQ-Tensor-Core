#include "../include/pe_array.hpp"


//  ███████████  ██████████      █████████                                           
// ░░███░░░░░███░░███░░░░░█     ███░░░░░███                                          
//  ░███    ░███ ░███  █ ░     ░███    ░███  ████████  ████████   ██████   █████ ████
//  ░██████████  ░██████       ░███████████ ░░███░░███░░███░░███ ░░░░░███ ░░███ ░███ 
//  ░███░░░░░░   ░███░░█       ░███░░░░░███  ░███ ░░░  ░███ ░░░   ███████  ░███ ░███ 
//  ░███         ░███ ░   █    ░███    ░███  ░███      ░███      ███░░███  ░███ ░███ 
//  █████        ██████████    █████   █████ █████     █████    ░░████████ ░░███████ 
// ░░░░░        ░░░░░░░░░░    ░░░░░   ░░░░░ ░░░░░     ░░░░░      ░░░░░░░░   ░░░░░███ 
//                                                                          ███ ░███ 
//                                                                         ░░██████  
//                                                                          ░░░░░░   


namespace perf_model {


void PEArray::applyGlobalStalling(std::vector<size_t>& pe_last_finish, 
                           const std::vector<std::vector<size_t>>& pe_tile_assignments) {
}

void PEArray::applyGlobalBarrierPerGEMM(std::vector<size_t>& pe_last_finish, 
                             const std::vector<std::vector<size_t>>& pe_tile_assignments) {
    // Find PEs That Processed At Least One Tile
    std::vector<size_t> active_pes;
    for (size_t i = 0; i < num_pes; i++) {
        if (!pe_tile_assignments[i].empty()) {
            active_pes.push_back(i);
        }
    }
    
    if (!active_pes.empty()) {
        // For Each Active PE, Adjust Its Finish Time Based On Extra Adder Cycles
        for (size_t pe_idx : active_pes) {
            size_t total_extra_cycles = 0;
            
            // Sum Up The Extra Adder Cycles For All Tiles Processed By This PE
            for (size_t tile_idx : pe_tile_assignments[pe_idx]) {
                // Get The Extra Adder Cycles From Each Tile's Processed Results
                total_extra_cycles += pes[pe_idx]->get_stats().extra_adder_cycles;
            }
            
            // Add These Extra Cycles To The PE's Finish Time
            pe_last_finish[pe_idx] += total_extra_cycles;
        }
        
        // Apply Global Barrier Synchronisation - All PEs Must Wait For The Slowest One
        size_t max_finish_time = *std::max_element(pe_last_finish.begin(), pe_last_finish.end());
        global_barriers = active_pes.size(); // Count One Barrier Per Active PE
        
        // Set All Active PEs To Finish At The Same Time (Global Barrier)
        for (size_t pe_idx : active_pes) {
            size_t wait_time = max_finish_time - pe_last_finish[pe_idx];
            if (wait_time > 0) {
                pes[pe_idx]->add_barrier_wait(wait_time);
            }
            pe_last_finish[pe_idx] = max_finish_time;
        }
    }
}


void PEArray::applyGlobalBarrierPerBatch(std::vector<size_t>& pe_last_finish, 
                              const std::vector<std::vector<size_t>>& pe_tile_assignments) {
    // This Is Similar To The GEMM Barrier, But Counts As An Additional Batch Barrier
    std::vector<size_t> active_pes;
    for (size_t i = 0; i < num_pes; i++) {
        if (!pe_tile_assignments[i].empty()) {
            active_pes.push_back(i);
        }
    }
    
    if (!active_pes.empty()) {
        // For Each Active PE, Adjust Its Finish Time Based On Extra Adder Cycles
        for (size_t pe_idx : active_pes) {
            size_t total_extra_cycles = 0;
            
            // Sum Up The Extra Adder Cycles For All Tiles Processed By This PE
            for (size_t tile_idx : pe_tile_assignments[pe_idx]) {
                // Get The Extra Adder Cycles From Each Tile's Processed Results
                total_extra_cycles += pes[pe_idx]->get_stats().extra_adder_cycles;
            }
            
            // Add These Extra Cycles To The PE's Finish Time
            pe_last_finish[pe_idx] += total_extra_cycles;
        }
        
        // Apply Global Barrier Synchronisation - All PEs Must Wait For The Slowest One
        size_t max_finish_time = *std::max_element(pe_last_finish.begin(), pe_last_finish.end());
        global_barriers++; // Add One More Batch Barrier
        
        // Set All Active PEs To Finish At The Same Time (Global Barrier)
        for (size_t pe_idx : active_pes) {
            size_t wait_time = max_finish_time - pe_last_finish[pe_idx];
            if (wait_time > 0) {
                pes[pe_idx]->add_barrier_wait(wait_time);
            }
            pe_last_finish[pe_idx] = max_finish_time;
        }
    }
}


void PEArray::applyAsyncLocalFIFO(std::vector<size_t>& pe_last_finish, 
                       const std::vector<std::vector<size_t>>& pe_tile_assignments) {
    // For Each PE, Calculate FIFO Usage And Potential Overflow
    for (size_t i = 0; i < num_pes; i++) {
        if (pe_tile_assignments[i].empty()) continue;
        
        size_t fifo_wait_cycles = 0;
        
        // Calculate Basic Finish Time With Extra Adder Cycles
        size_t total_extra_cycles = 0;
        for (size_t j = 0; j < pe_tile_assignments[i].size(); j++) {
            total_extra_cycles += pes[i]->get_stats().extra_adder_cycles;
        }
        
        pe_last_finish[i] += total_extra_cycles;
        
        // Find The Fastest PE Finish Time To Determine When Tiles Start Being Drained
        size_t min_finish_time = pe_last_finish[0];
        for (size_t j = 1; j < num_pes; j++) {
            if (!pe_tile_assignments[j].empty()) {
                min_finish_time = std::min(min_finish_time, pe_last_finish[j]);
            }
        }
        
        // Estimate FIFO Behavior Based On Relative Finish Times
        size_t drain_rate = 1; // Assume One Tile Drained Per Cycle
        size_t tiles_completed = pe_tile_assignments[i].size();
        
        // Adjust For FIFO Overflow
        if (tiles_completed > fifo_depth) {
            // Calculate How Many Cycles PE Must Stall Waiting For FIFO Space
            size_t overflow_tiles = tiles_completed - fifo_depth;
            fifo_wait_cycles = overflow_tiles / drain_rate;
            
            // Update Finish Time And Record Stats
            pe_last_finish[i] += fifo_wait_cycles;
            pes[i]->add_fifo_wait_cycles(fifo_wait_cycles);
        }
        
        // Record Maximum FIFO Usage
        pes[i]->set_max_fifo_occupancy(std::min(tiles_completed, fifo_depth));
    }
    
    // No Global Barriers In This Mode
    global_barriers = 0;
}


void PEArray::applyAsyncSharedBuffer(std::vector<size_t>& pe_last_finish, 
                          const std::vector<std::vector<size_t>>& pe_tile_assignments) {
    // For Mode 4, PEs Continue Working Independently
    // First, Apply Extra Adder Cycles To Each PE's Finish Time
    for (size_t i = 0; i < num_pes; i++) {
        if (pe_tile_assignments[i].empty()) continue;
        
        size_t total_extra_cycles = 0;
        for (size_t j = 0; j < pe_tile_assignments[i].size(); j++) {
            total_extra_cycles += pes[i]->get_stats().extra_adder_cycles;
        }
        
        pe_last_finish[i] += total_extra_cycles;
    }
    
    // No Global Barriers In This Mode
    global_barriers = 0;
}


} // namespace perf_model 