#pragma once
#include <cstddef>
#include <vector>

namespace perf_model {

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

struct SystemStats {
    std::vector<PEStats> pe_stats;
    
    // System-wide parallel stats
    size_t total_parallel_cycles{0};
    size_t total_parallel_mask_ops{0};
    size_t total_parallel_shifts{0};
    size_t total_parallel_additions{0};
    
    void clear() {
        pe_stats.clear();
        total_parallel_cycles = 0;
        total_parallel_mask_ops = 0;
        total_parallel_shifts = 0;
        total_parallel_additions = 0;
    }

    void calculateSystemStats() {
        if (pe_stats.empty()) return;
        
        // In parallel execution, we take the maximum cycles across all PEs
        total_parallel_cycles = 0;
        total_parallel_mask_ops = 0;
        total_parallel_shifts = 0;
        total_parallel_additions = 0;

        for (const auto& pe : pe_stats) {
            total_parallel_cycles = std::max(total_parallel_cycles, pe.total_cycles);
            total_parallel_mask_ops = std::max(total_parallel_mask_ops, pe.masking_operations);
            total_parallel_shifts = std::max(total_parallel_shifts, pe.shifting_operations);
            total_parallel_additions = std::max(total_parallel_additions, pe.addition_operations);
        }
    }
};

} // namespace perf_model