#pragma once
#include <cstddef>

namespace spmpGEMM {

struct PEStats {
    size_t total_cycles;

    // Per-cycle Stats (operations that happen in parallel count as a single operation)
    size_t masking_operations{0};     
    size_t shifting_operations{0};     
    size_t addition_operations{0};     
    
    // Total Stats (counts regardless of parallelism)
    size_t total_mask_ops{0};         
    size_t total_shifts{0};           
    size_t total_additions{0};       
    
    void clear() {
        total_cycles = 0;
        masking_operations = 0;
        shifting_operations = 0;
        addition_operations = 0;
        total_mask_ops = 0;
        total_shifts = 0;
        total_additions = 0;
    }
};

} // namespace spmpGEMM