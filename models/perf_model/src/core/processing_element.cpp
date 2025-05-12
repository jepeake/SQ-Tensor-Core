#include "../include/processing_element.hpp"
#include "../include/config.hpp"
#include <iostream>
#include <iomanip>
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
//  ██████████ ████                                                █████              ░░██████             
// ░░███░░░░░█░░███                                               ░░███                 ░░░░░░        
//  ░███  █ ░  ░███   ██████  █████████████    ██████  ████████   ███████                      
//  ░██████    ░███  ███░░███░░███░░███░░███  ███░░███░░███░░███ ░░░███░                       
//  ░███░░█    ░███ ░███████  ░███ ░███ ░███ ░███████  ░███ ░███   ░███                        
//  ░███ ░   █ ░███ ░███░░░   ░███ ░███ ░███ ░███░░░   ░███ ░███   ░███ ███                    
//  ██████████ █████░░██████  █████░███ █████░░██████  ████ █████  ░░█████                     
// ░░░░░░░░░░ ░░░░░  ░░░░░░  ░░░░░ ░░░ ░░░░░  ░░░░░░  ░░░░ ░░░░░    ░░░░░                      


namespace perf_model {


ProcessingElement::ProcessingElement(size_t ts) : tile_size(ts) {

    stats.clear();
    
    // Initialise Adder Tree Width from Config or Calculate Based on Sparsity
    if (config::adder_tree_width > 0) {
        adder_tree_width = config::adder_tree_width;
    } else {
        calculate_adder_tree_width(config::expected_weight_sparsity, config::expected_activation_sparsity);
    }
    
    stats.adder_tree_width = adder_tree_width;
}


void ProcessingElement::calculate_adder_tree_width(double expected_weight_sparsity, double expected_activation_sparsity) {

    // Calculate Expected Number of Non-Zero Values After Multiplying Weight & Activation Tiles
    // Expected Inputs: tile_size * (1-weight_sparsity) * (1-activation_sparsity) * num_bits

    double expected_density = (1.0 - expected_weight_sparsity) * (1.0 - expected_activation_sparsity);
    double max_bits = config::num_pes > 0 ? 8 : 8; // Default to 8b
    double expected_inputs = tile_size * expected_density * max_bits;
    adder_tree_width = static_cast<size_t>(std::ceil(expected_inputs));
}


Tile<int32_t> ProcessingElement::mpGEMM(
    const std::vector<Tile<uint8_t>>& weight_tiles,
    const Tile<int16_t>& activation_tile,
    size_t num_bits,
    int16_t activation_threshold) {  
    
    Tile<int32_t> result(tile_size, tile_size);
    
    // Keep Track of Maximum Number of Adder Tree Inputs
    stats.max_adder_tree_inputs = 0;
    stats.extra_adder_cycles = 0;
    
    for (size_t i = 0; i < tile_size; i++) {
        for (size_t j = 0; j < tile_size; j++) {
            int32_t final_acc = 0;
            
            // Count Actual Number of Non-Zero Inputs That Will Enter the Adder Tree
            size_t nonzero_inputs = 0;
            std::vector<int32_t> all_shifted_activations;
            
            for (size_t k = 0; k < tile_size; k++) {
                // Check Activation Sparsity
                if (std::abs(activation_tile.at(i, k)) <= activation_threshold) {
                    continue;  // Skip Computation for Sparse Activations
                }
                
                // Mask
                // In Hardware, all masks occur in parallel across all weight-activation pairs, across all bit-planes
                stats.masking_operations = 1;
                std::vector<int32_t> selected_activations(num_bits, 0);
                for (size_t bit = 0; bit < num_bits; bit++) {
                    stats.total_mask_ops++;
                    if (weight_tiles[bit].at(k, j) == 1) {
                        selected_activations[bit] = activation_tile.at(i, k);
                        // Count each non-zero weight-activation pair as an input to the adder tree
                        nonzero_inputs++;
                    }
                }
                
                // Shift
                // In Hardware, all shifts occur in parallel (rewiring)
                stats.shifting_operations = 0;
                
                for (size_t bit = 0; bit < num_bits; bit++) {
                    stats.total_shifts++;
                    if (selected_activations[bit] != 0) {
                        int32_t shifted_value = (selected_activations[bit] << bit);
                        all_shifted_activations.push_back(shifted_value);
                    }
                }
            }
            
            // Update Max Inputs Seen
            stats.max_adder_tree_inputs = std::max(stats.max_adder_tree_inputs, nonzero_inputs);
            
            // Calculate Extra Cycles Needed for Adder Tree if Inputs Exceed Width
            size_t total_passes = (nonzero_inputs + adder_tree_width - 1) / adder_tree_width;
            size_t extra_cycles = total_passes > 1 ? total_passes - 1 : 0;
            
            // Adder Tree Stats 
            size_t adder_stages = static_cast<size_t>(std::ceil(std::log2(std::max(nonzero_inputs, size_t(1)))));
            
            stats.total_additions += nonzero_inputs > 0 ? nonzero_inputs - 1 : 0;
            stats.addition_operations = adder_stages + extra_cycles;
            stats.extra_adder_cycles += extra_cycles;
            
            // Add All Values - In Hardware This Would Be Done in Multiple Passes if Needed
            for (const auto& value : all_shifted_activations) {
                final_acc += value;
            }
            
            result.at(i, j) = final_acc;
        }
    }

    // Update Total Cycles to Include the Extra Adder Tree Cycles
    stats.total_cycles = stats.masking_operations + 
                         stats.shifting_operations + 
                         stats.addition_operations;

    return result;
}


} // namespace perf_model