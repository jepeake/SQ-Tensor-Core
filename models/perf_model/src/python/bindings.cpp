#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/simd_engine.hpp"
#include "../include/stats.hpp"
#include "../include/performance_metrics.hpp"

namespace py = pybind11;

PYBIND11_MODULE(perf_model, m) {
    m.doc() = "SQ-TC Python Bindings"; 

    // Bind the Synchronisation Mode
    py::enum_<perf_model::SynchronisationMode>(m, "SynchronisationMode")
        .value("GLOBAL_STALLING", perf_model::SynchronisationMode::GLOBAL_STALLING)
        .value("GLOBAL_BARRIER_PER_GEMM", perf_model::SynchronisationMode::GLOBAL_BARRIER_PER_GEMM)
        .value("GLOBAL_BARRIER_PER_BATCH", perf_model::SynchronisationMode::GLOBAL_BARRIER_PER_BATCH)
        .value("ASYNC_LOCAL_FIFO", perf_model::SynchronisationMode::ASYNC_LOCAL_FIFO)
        .value("ASYNC_SHARED_BUFFER", perf_model::SynchronisationMode::ASYNC_SHARED_BUFFER)
        .export_values();

    // Bind the Tile
    py::class_<perf_model::Tile<int32_t>>(m, "Tile")
        .def(py::init<>())
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, int32_t>())
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
        .def_readonly("total_additions", &perf_model::PEStats::total_additions)
        .def_readonly("max_adder_tree_inputs", &perf_model::PEStats::max_adder_tree_inputs)
        .def_readonly("adder_tree_width", &perf_model::PEStats::adder_tree_width)
        .def_readonly("extra_adder_cycles", &perf_model::PEStats::extra_adder_cycles)
        .def_readonly("stall_cycles", &perf_model::PEStats::stall_cycles)
        .def_readonly("barrier_waits", &perf_model::PEStats::barrier_waits)
        .def_readonly("barrier_wait_cycles", &perf_model::PEStats::barrier_wait_cycles)
        .def_readonly("fifo_wait_cycles", &perf_model::PEStats::fifo_wait_cycles)
        .def_readonly("max_fifo_occupancy", &perf_model::PEStats::max_fifo_occupancy);

    // Bind the SystemStats
    py::class_<perf_model::SystemStats>(m, "SystemStats")
        .def_readonly("pe_stats", &perf_model::SystemStats::pe_stats)
        .def_readonly("total_parallel_cycles", &perf_model::SystemStats::total_parallel_cycles)
        .def_readonly("total_parallel_mask_ops", &perf_model::SystemStats::total_parallel_mask_ops)
        .def_readonly("total_parallel_shifts", &perf_model::SystemStats::total_parallel_shifts)
        .def_readonly("total_parallel_additions", &perf_model::SystemStats::total_parallel_additions)
        .def_readonly("global_barriers", &perf_model::SystemStats::global_barriers)
        .def_readonly("slowest_pe_indices", &perf_model::SystemStats::slowest_pe_indices)
        .def_readonly("sync_mode", &perf_model::SystemStats::sync_mode)
        .def_readonly("global_stalls", &perf_model::SystemStats::global_stalls)
        .def_readonly("batch_barriers", &perf_model::SystemStats::batch_barriers)
        .def_readonly("output_buffer_size", &perf_model::SystemStats::output_buffer_size)
        .def_readonly("fifo_depth", &perf_model::SystemStats::fifo_depth)
        .def_readonly("fifo_overflows", &perf_model::SystemStats::fifo_overflows)
        .def_readonly("max_skew_cycles", &perf_model::SystemStats::max_skew_cycles)
        .def("get_sync_mode_description", &perf_model::SystemStats::getSyncModeDescription);

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
             
    // Create a High-Level Engine Wrapper Class
    py::class_<perf_model::SIMDEngine>(m, "Engine")
        .def(py::init<const std::string&>())
        .def("compute", &perf_model::SIMDEngine::compute,
             py::arg("activations"),
             py::arg("activation_threshold") = 0)
        .def("get_stats", &perf_model::SIMDEngine::getStats)
        .def("get_total_cycles", [](perf_model::SIMDEngine& self) {
            return self.getStats().total_parallel_cycles;
        })
        .def("get_global_barriers", [](perf_model::SIMDEngine& self) {
            return self.getStats().global_barriers;
        })
        .def("get_slowest_pe_index", [](perf_model::SIMDEngine& self) {
            return self.getStats().slowest_pe_indices;
        })
        .def("get_sync_mode", [](perf_model::SIMDEngine& self) {
            return self.getStats().sync_mode;
        })
        .def("get_sync_mode_description", [](perf_model::SIMDEngine& self) {
            return self.getStats().getSyncModeDescription();
        })
        .def("get_max_skew_cycles", [](perf_model::SIMDEngine& self) {
            return self.getStats().max_skew_cycles;
        })
        .def("get_matrix_rows", &perf_model::SIMDEngine::getMatrixRows)
        .def("get_matrix_cols", &perf_model::SIMDEngine::getMatrixCols)
        .def("get_tile_size", &perf_model::SIMDEngine::getTileSize)
        .def("get_num_pes", &perf_model::SIMDEngine::getNumPEs)
        .def("get_num_matmuls", &perf_model::SIMDEngine::getNumMatMuls)
        .def("get_performance_metrics", &perf_model::SIMDEngine::getPerformanceMetrics,
             py::arg("clock_frequency_hz"));
}