#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/simd_engine.hpp"
#include "../include/stats.hpp"
#include "../include/performance_metrics.hpp"  // Include the header for PerformanceMetrics

namespace py = pybind11;

PYBIND11_MODULE(panda, m) {
    m.doc() = "PANDA Python Bindings"; 

    // Bind the Tile
    py::class_<panda::Tile<int32_t>>(m, "Tile")
        .def(py::init<>())
        .def(py::init<size_t, size_t>())
        .def_readwrite("rows", &panda::Tile<int32_t>::rows)
        .def_readwrite("cols", &panda::Tile<int32_t>::cols)
        .def_readwrite("data", &panda::Tile<int32_t>::data)
        .def("at", static_cast<int32_t& (panda::Tile<int32_t>::*)(size_t, size_t)>(&panda::Tile<int32_t>::at));

    // Bind the PEStats
    py::class_<panda::PEStats>(m, "PEStats")
        .def_readonly("total_cycles", &panda::PEStats::total_cycles)
        .def_readonly("masking_operations", &panda::PEStats::masking_operations)
        .def_readonly("shifting_operations", &panda::PEStats::shifting_operations)
        .def_readonly("addition_operations", &panda::PEStats::addition_operations)
        .def_readonly("total_mask_ops", &panda::PEStats::total_mask_ops)
        .def_readonly("total_shifts", &panda::PEStats::total_shifts)
        .def_readonly("total_additions", &panda::PEStats::total_additions);

    // Bind the SystemStats
    py::class_<panda::SystemStats>(m, "SystemStats")
        .def_readonly("pe_stats", &panda::SystemStats::pe_stats)
        .def_readonly("total_parallel_cycles", &panda::SystemStats::total_parallel_cycles)
        .def_readonly("total_parallel_mask_ops", &panda::SystemStats::total_parallel_mask_ops)
        .def_readonly("total_parallel_shifts", &panda::SystemStats::total_parallel_shifts)
        .def_readonly("total_parallel_additions", &panda::SystemStats::total_parallel_additions);

    // Bind the PerformanceMetrics 
    py::class_<panda::PerformanceMetrics>(m, "PerformanceMetrics")
        .def_readonly("system_latency_ns", &panda::PerformanceMetrics::system_latency_ns)
        .def_readonly("throughput_ops", &panda::PerformanceMetrics::throughput_ops)
        .def_readonly("memory_bandwidth_bytes_per_sec", &panda::PerformanceMetrics::memory_bandwidth_bytes_per_sec)
        .def_readonly("arithmetic_intensity", &panda::PerformanceMetrics::arithmetic_intensity);

    // Bind the SIMDEngine 
    py::class_<panda::SIMDEngine>(m, "SIMDEngine")
        .def(py::init<const std::string&>())
        .def("compute", &panda::SIMDEngine::compute,
             py::arg("activations"),
             py::arg("activation_threshold") = 0)
        .def("get_stats", &panda::SIMDEngine::getStats)
        .def("get_matrix_rows", &panda::SIMDEngine::getMatrixRows)
        .def("get_matrix_cols", &panda::SIMDEngine::getMatrixCols)
        .def("get_tile_size", &panda::SIMDEngine::getTileSize)
        .def("get_num_pes", &panda::SIMDEngine::getNumPEs)
        .def("get_performance_metrics", &panda::SIMDEngine::getPerformanceMetrics,
             py::arg("clock_frequency_hz"));
}