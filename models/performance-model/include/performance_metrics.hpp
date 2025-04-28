#pragma once

namespace panda {

struct PerformanceMetrics {
    // Overall Latency in ns
    double system_latency_ns;
    // Throughput defined as Total Operations (FLOPs) per Second
    double throughput_ops;
    // Memory Bandwidth in bytes per second
    double memory_bandwidth_bytes_per_sec;
    // Arithmetic Intensity in FLOPs per Byte
    double arithmetic_intensity;
    
    // Hardware Costs
    double total_energy_pj;  
    double total_area_um2;       
    
    // Per-component Energy and Area
    double adder_energy_pj;      
    double mask_energy_pj;       
    double adder_area_um2;       
    double mask_area_um2;        
};

} // namespace panda 