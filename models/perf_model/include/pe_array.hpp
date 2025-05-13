#pragma once
#include "processing_element.hpp"
#include "tile.hpp"
#include "config.hpp"
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath> 
#include <algorithm>
#include <unordered_map>


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


class PEArray {
private:
    std::vector<std::unique_ptr<ProcessingElement>> pes;
    size_t num_pes;
    size_t tile_size;
    size_t global_barriers;
    SynchronisationMode sync_mode;
    size_t batch_size;
    size_t fifo_depth;
    size_t output_buffer_size;
    
    // For Mode 4: Output Buffer Tracking
    std::unordered_map<size_t, bool> output_buffer_entries;
    
    // Helper Methods for Different Synchronisation Modes
    void applyGlobalStalling(std::vector<size_t>& pe_last_finish, 
                           const std::vector<std::vector<size_t>>& pe_tile_assignments);
    
    void applyGlobalBarrierPerGEMM(std::vector<size_t>& pe_last_finish, 
                                 const std::vector<std::vector<size_t>>& pe_tile_assignments);
    
    void applyGlobalBarrierPerBatch(std::vector<size_t>& pe_last_finish, 
                                  const std::vector<std::vector<size_t>>& pe_tile_assignments);
    
    void applyAsyncLocalFIFO(std::vector<size_t>& pe_last_finish, 
                           const std::vector<std::vector<size_t>>& pe_tile_assignments);
    
    void applyAsyncSharedBuffer(std::vector<size_t>& pe_last_finish, 
                              const std::vector<std::vector<size_t>>& pe_tile_assignments);

public:
    PEArray(size_t num_processing_elements, size_t ts) 
        : num_pes(num_processing_elements), tile_size(ts), global_barriers(0) {
        
        // Initialise PEs
        for (size_t i = 0; i < num_pes; i++) {
            pes.push_back(std::make_unique<ProcessingElement>(tile_size));
        }
        
        // Set Synchronisation Mode from Config
        sync_mode = static_cast<SynchronisationMode>(config::synchronisation_mode);
        batch_size = config::batch_size;
        
        // Calculate FIFO Depth for Mode 3
        // ceil((Tmax - Tmin) / Tmin)
        // Tmin: with full sparsity
        // Tmax: with no sparsity
        double sparsity_factor = (1.0 - config::expected_weight_sparsity) * 
                                (1.0 - config::expected_activation_sparsity);
        double t_min = sparsity_factor * tile_size;
        double t_max = tile_size;
        fifo_depth = std::max(size_t(4), static_cast<size_t>(std::ceil((t_max - t_min) / t_min)));
        
        // Calculate Output Buffer Size for Mode 4
        output_buffer_size = ts * ts * num_processing_elements;
    }

    std::vector<Tile<int32_t>> processTiles(
        const std::vector<std::pair<std::vector<Tile<uint8_t>>, Tile<int16_t>>>& tiles,
        size_t num_bits,
        int16_t activation_threshold = 0) {
        
        size_t total_tiles = tiles.size();
        std::vector<Tile<int32_t>> results(total_tiles);
        
        // Timing Parameters
        const size_t L_mask = 1;  // Masking Stage Latency
        size_t L_add = static_cast<size_t>(std::ceil(std::log2(tile_size * num_bits)));
        const size_t L_total = L_mask + L_add;  // Total Computation Latency per Tile
        // Memory Load Latency per Tile – this is the time to load one tile from (off-chip) memory.
        const size_t L_load = 3;

        // Track Finish Time (Cycle Count) for Each PE
        std::vector<size_t> pe_last_finish(num_pes, 0);
        
        // Track Which Tiles are Assigned to Which PEs
        std::vector<std::vector<size_t>> pe_tile_assignments(num_pes);
        
        // Reset FIFO Tracking for Mode 3
        for (auto& pe : pes) {
            pe->reset_stats();
        }
        
        // Reset Output Buffer Tracking for Mode 4
        output_buffer_entries.clear();
        
        // Schedule Each Tile Round-Robin over the Available PEs with Double Buffering
        // For the First Tile on a Given PE the Effective Cycle Cost Includes L_load + L_total
        // For Subsequent Tiles, the Load Stage is Overlapped with the Compute Stage so
        // the Added Cost is the Maximum of the Two (i.e. the Longer Stage)
        for (size_t i = 0; i < total_tiles; i++) {
            size_t pe_index = i % num_pes;
            size_t current_cost = 0;
            if (pe_last_finish[pe_index] == 0) {
                // No Previous Tile: Must Pay Both the Memory Load and the Computation Cost
                current_cost = L_load + L_total;
            } else {
                // Double Buffering Hides the Load Cost if the Compute Stage is Longer
                current_cost = std::max(L_total, L_load);
            }

            size_t start_cycle = pe_last_finish[pe_index];
            size_t finish_time = start_cycle + current_cost;
            
            const auto& work_item = tiles[i];
            
            // Track Which Tile is Assigned to Which PE
            pe_tile_assignments[pe_index].push_back(i);
            
            // Synchronously Perform the Matrix Multiply on the Assigned PE.
            results[i] = pes[pe_index]->mpGEMM(
                work_item.first,
                work_item.second, 
                num_bits,
                activation_threshold
            );
            
            pe_last_finish[pe_index] = finish_time;
            
            // For Mode 0 (Global Stalling), Check for Extra Adder Cycles after Each Tile
            if (sync_mode == SynchronisationMode::GLOBAL_STALLING) {
                size_t extra_cycles = pes[pe_index]->get_stats().extra_adder_cycles;
                if (extra_cycles > 0) {
                    // Apply Global Stall for This Tile
                    for (size_t j = 0; j < num_pes; j++) {
                        if (j != pe_index) {
                            // Add Stall Cycles to All Other PEs
                            pes[j]->add_stall_cycles(extra_cycles);
                        }
                        pe_last_finish[j] += extra_cycles;
                    }
                }
            }
            
            // For Mode 2 (Global Barrier per Batch), Check if We've Reached a Batch Boundary
            if (sync_mode == SynchronisationMode::GLOBAL_BARRIER_PER_BATCH && 
                (i + 1) % (batch_size * num_pes) == 0) {
                
                // Find the Slowest PE in This Batch
                size_t max_time = 0;
                for (size_t j = 0; j < num_pes; j++) {
                    max_time = std::max(max_time, pe_last_finish[j]);
                }
                
                // Apply Batch Barrier - All PEs Wait for the Slowest One
                for (size_t j = 0; j < num_pes; j++) {
                    size_t wait_time = max_time - pe_last_finish[j];
                    if (wait_time > 0) {
                        pes[j]->add_barrier_wait(wait_time);
                    }
                    pe_last_finish[j] = max_time;
                }
                
                global_barriers++;
            }
        }
        
        // Apply the Appropriate Synchronisation Mode to the Final Results
        switch (sync_mode) {
            case SynchronisationMode::GLOBAL_STALLING:
                // Already handled during processing
                break;
                
            case SynchronisationMode::GLOBAL_BARRIER_PER_GEMM:
                applyGlobalBarrierPerGEMM(pe_last_finish, pe_tile_assignments);
                break;
                
            case SynchronisationMode::GLOBAL_BARRIER_PER_BATCH:
                // Final Batch Barrier if Needed
                if (total_tiles % (batch_size * num_pes) != 0) {
                    applyGlobalBarrierPerBatch(pe_last_finish, pe_tile_assignments);
                }
                break;
                
            case SynchronisationMode::ASYNC_LOCAL_FIFO:
                applyAsyncLocalFIFO(pe_last_finish, pe_tile_assignments);
                break;
                
            case SynchronisationMode::ASYNC_SHARED_BUFFER:
                applyAsyncSharedBuffer(pe_last_finish, pe_tile_assignments);
                break;
        }
        
        // Update Each PE's Stats to Reflect the Simulated Pipelined Total Cycles
        for (size_t i = 0; i < num_pes; i++) {
            pes[i]->set_simulated_total_cycles(pe_last_finish[i]);
        }
        
        return results;
    }

    // Process an Entire Output Tile with a Single PE - Local PE Accumulation Mode
    Tile<int32_t> processOutputTile(
        size_t pe_index,
        const std::vector<std::pair<std::vector<Tile<uint8_t>>, Tile<int16_t>>>& work_items,
        size_t num_bits,
        int16_t activation_threshold = 0) {
        
        if (pe_index >= num_pes) {
            throw std::out_of_range("PE index out of range");
        }
        
        // Result Accumulator for This Output Tile
        Tile<int32_t> result(tile_size, tile_size, 0); // Initialize to Zeros
        
        // Timing Parameters
        const size_t L_mask = 1;  // Masking Stage Latency
        size_t L_add = static_cast<size_t>(std::ceil(std::log2(tile_size * num_bits)));
        const size_t L_total = L_mask + L_add;  // Total Computation Latency per Tile
        const size_t L_load = 3;  // Memory Load Latency per Tile
        
        // For Local PE Accumulation, We Need to Track the PE's Finish Time
        size_t pe_finish_time = 0;
        size_t total_extra_cycles = 0;
        
        // Reset the PE's Stats Before Processing
        pes[pe_index]->reset_stats();
        
        // Process Each Work Item (Weight-Activation Tile Pair) Sequentially
        for (size_t i = 0; i < work_items.size(); i++) {
            // Calculate the Current Cost for This Work Item
            size_t current_cost = 0;
            if (i == 0) {
                // First Work Item: Must Pay Both Memory Load and Computation Cost
                current_cost = L_load + L_total;
            } else {
                // Subsequent Items: Double Buffering Hides Load Cost if Compute is Longer
                current_cost = std::max(L_total, L_load);
            }
            
            // Update the Finish Time
            pe_finish_time += current_cost;
            
            // Process the Current Work Item
            const auto& work_item = work_items[i];
            
            // Perform the Matrix Multiply for This Input Tile Pair
            Tile<int32_t> partial_result = pes[pe_index]->mpGEMM(
                work_item.first,
                work_item.second,
                num_bits,
                activation_threshold
            );
            
            // Handle Extra Adder Cycles Based on Synchronisation Mode
            size_t extra_cycles = pes[pe_index]->get_stats().extra_adder_cycles;
            
            if (sync_mode == SynchronisationMode::GLOBAL_STALLING) {
                // In Global Stalling Mode, Extra Cycles Affect All PEs
                // But for Local PE Accumulation, This is Just the One PE
                pe_finish_time += extra_cycles;
            } else {
                // For Other Modes, Track Extra Cycles to Apply Later
                total_extra_cycles += extra_cycles;
            }
            
            // Accumulate the Partial Result into the Final Result
            for (size_t row = 0; row < tile_size; row++) {
                for (size_t col = 0; col < tile_size; col++) {
                    result.at(row, col) += partial_result.at(row, col);
                }
            }
        }
        
        // Apply Extra Cycles Based on the Synchronisation Mode
        if (sync_mode != SynchronisationMode::GLOBAL_STALLING) {
            pe_finish_time += total_extra_cycles;
        }
        
        // Set the Final Cycle Count for This PE
        pes[pe_index]->set_simulated_total_cycles(pe_finish_time);
        
        // One Global Barrier per Output Tile for Barrier-Based Modes
        if (sync_mode == SynchronisationMode::GLOBAL_BARRIER_PER_GEMM) {
            global_barriers = 1;
        } else if (sync_mode == SynchronisationMode::GLOBAL_BARRIER_PER_BATCH) {
            global_barriers = std::max(size_t(1), work_items.size() / batch_size);
        }
        
        return result;
    }
    
    // Retrieve the Vector of Stats for Each Processing Element
    std::vector<PEStats> getAllStats() const {
        std::vector<PEStats> all_stats;
        all_stats.reserve(num_pes);
        for (const auto &pe : pes) {
            all_stats.push_back(pe->get_stats());
        }
        return all_stats;
    }
    
    // Get the Number of Global Barriers that Were Applied
    size_t getGlobalBarriers() const {
        return global_barriers;
    }
    
    // Get the Synchronisation Mode
    SynchronisationMode getSyncMode() const {
        return sync_mode;
    }
    
    // Get the FIFO Depth for Mode 3
    size_t getFIFODepth() const {
        return fifo_depth;
    }
    
    // Get the Output Buffer Size for Mode 4
    size_t getOutputBufferSize() const {
        return output_buffer_size;
    }
    
    // Get the Batch Size for Mode 2
    size_t getBatchSize() const {
        return batch_size;
    }
};


} // namespace perf_model 