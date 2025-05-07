#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/simd_engine.hpp"
#include "../include/stats.hpp"
#include "../include/performance_metrics.hpp"  // Include the header for PerformanceMetrics

namespace py = pybind11;

PYBIND11_MODULE(perf_model, m) {
    m.doc() = "SQ-TC Python Bindings"; 

    // Bind the Tile
    py::class_<perf_model::Tile<int32_t>>(m, "Tile")
        .def(py::init<>())
        .def(py::init<size_t, size_t>())
        .def_readwrite("rows", &perf_model::Tile<int32_t>::rows)
        .def_readwrite("cols", &perf_model::Tile<int32_t>::cols)
        .def_readwrite("data", &perf_model::Tile<int32_t>::data)
        .def("at", static_cast<int32_t& (perf_model::Tile<int32_t>::*)(size_t, size_t)>(&perf_model::Tile<int32_t>::at));

    // Bind the PEStats
    py::class_<perf_model::PEStats>(m, "PEStats")
        .def_readonly("total_cycles", &perf_model::PEStats::total_cycles)
        .def_readonly("masking_operations", &perf_model::PEStats::masking_operations)
        .def_readonly("shifting_operations", &perf_model::PEStats::shifting_operations)
        .def_readonly("addition_operations", &perf_model::PEStats::addition_operations)
        .def_readonly("total_mask_ops", &perf_model::PEStats::total_mask_ops)
        .def_readonly("total_shifts", &perf_model::PEStats::total_shifts)
        .def_readonly("total_additions", &perf_model::PEStats::total_additions);

    // Bind the SystemStats
    py::class_<perf_model::SystemStats>(m, "SystemStats")
        .def_readonly("pe_stats", &perf_model::SystemStats::pe_stats)
        .def_readonly("total_parallel_cycles", &perf_model::SystemStats::total_parallel_cycles)
        .def_readonly("total_parallel_mask_ops", &perf_model::SystemStats::total_parallel_mask_ops)
        .def_readonly("total_parallel_shifts", &perf_model::SystemStats::total_parallel_shifts)
        .def_readonly("total_parallel_additions", &perf_model::SystemStats::total_parallel_additions);

    // Bind the PerformanceMetrics 
    py::class_<perf_model::PerformanceMetrics>(m, "PerformanceMetrics")
        .def_readonly("system_latency_ns", &perf_model::PerformanceMetrics::system_latency_ns)
        .def_readonly("throughput_ops", &perf_model::PerformanceMetrics::throughput_ops)
        .def_readonly("ops_per_cycle", &perf_model::PerformanceMetrics::ops_per_cycle)
        .def_readonly("memory_bandwidth_bytes_per_sec", &perf_model::PerformanceMetrics::memory_bandwidth_bytes_per_sec)
        .def_readonly("arithmetic_intensity", &perf_model::PerformanceMetrics::arithmetic_intensity)
        .def_readonly("total_energy_pj", &perf_model::PerformanceMetrics::total_energy_pj)
        .def_readonly("total_area_um2", &perf_model::PerformanceMetrics::total_area_um2)
        .def_readonly("adder_energy_pj", &perf_model::PerformanceMetrics::adder_energy_pj)
        .def_readonly("mask_energy_pj", &perf_model::PerformanceMetrics::mask_energy_pj)
        .def_readonly("adder_area_um2", &perf_model::PerformanceMetrics::adder_area_um2)
        .def_readonly("mask_area_um2", &perf_model::PerformanceMetrics::mask_area_um2);

    // Bind the SIMDEngine 
    py::class_<perf_model::SIMDEngine>(m, "SIMDEngine")
        .def(py::init<const std::string&>())
        .def("compute", &perf_model::SIMDEngine::compute,
             py::arg("activations"),
             py::arg("activation_threshold") = 0)
        .def("get_stats", &perf_model::SIMDEngine::getStats)
        .def("get_matrix_rows", &perf_model::SIMDEngine::getMatrixRows)
        .def("get_matrix_cols", &perf_model::SIMDEngine::getMatrixCols)
        .def("get_tile_size", &perf_model::SIMDEngine::getTileSize)
        .def("get_num_pes", &perf_model::SIMDEngine::getNumPEs)
        .def("get_num_matmuls", &perf_model::SIMDEngine::getNumMatMuls)
        .def("get_performance_metrics", &perf_model::SIMDEngine::getPerformanceMetrics,
             py::arg("clock_frequency_hz"));
}