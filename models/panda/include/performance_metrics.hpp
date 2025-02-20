#pragma once

namespace panda {

struct PerformanceMetrics {
    // Overall Latency in ns
    double system_latency_ns;
    // Throughput defined as Total Operations (MACs) per Second
    double throughput_ops;
    // Memory Bandwidth in bytes per second
    double memory_bandwidth_bytes_per_sec;
};

} // namespace panda 