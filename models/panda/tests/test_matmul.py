import numpy as np
from panda import SIMDEngine
from preprocessing.preprocess_weights import preprocess_weights
from typing import List, Dict
import textwrap
import os
import contextlib
import sys


# ----- Pretty Printing -----

def print_matrix(name: str, matrix: np.ndarray, indent: int = 0):
    """Pretty print a matrix with name and formatting"""
    indent_str = " " * indent
    print(f"\n{indent_str}{name}:")
    print(f"{indent_str}{'=' * (len(name) + 1)}")
    
    max_width = max(len(str(x)) for x in matrix.flatten())
    
    for row in matrix:
        print(indent_str + " ".join(f"{x:>{max_width}}" for x in row))

def print_tile(name: str, tile: np.ndarray, indent: int = 0):
    """Pretty print a tile with borders"""
    indent_str = " " * indent
    print(f"\n{indent_str}{name}:")
    print(f"{indent_str}┌{'─' * (tile.shape[1] * 4 + 1)}┐")
    
    for row in tile:
        print(f"{indent_str}│ " + " ".join(f"{x:>2}" for x in row) + " │")
    
    print(f"{indent_str}└{'─' * (tile.shape[1] * 4 + 1)}┘")

def print_pe_assignment(pe_idx: int, weight_tiles: List[np.ndarray], act_tile: np.ndarray, indent: int = 0):
    """Print PE assignment details with tiles"""
    indent_str = " " * indent
    separator = f"{indent_str}{'─' * 50}"
    
    print(f"\n{separator}")
    print(f"{indent_str}Processing Element {pe_idx}")
    print(f"{indent_str}{separator}")
    
    print_tile("Activation Tile", act_tile, indent + 2)
    
    for bit, w_tile in enumerate(weight_tiles):
        print_tile(f"Weight Tile (Bit {bit})", w_tile, indent + 2)

def print_matrix_info(engine: SIMDEngine):
    """Print matrix and tiling information"""
    matrix_rows = engine.get_matrix_rows()
    matrix_cols = engine.get_matrix_cols()
    tile_size = engine.get_tile_size()
    num_pes = engine.get_num_pes()
    
    num_row_tiles = (matrix_rows + tile_size - 1) // tile_size
    num_col_tiles = (matrix_cols + tile_size - 1) // tile_size
    total_tiles = num_row_tiles * num_col_tiles
    
    print("\nMatrix Configuration")
    print("═" * 50)
    print(f"Matrix Size: {matrix_rows}x{matrix_cols}")
    print(f"Tile Size: {tile_size}x{tile_size}")
    print(f"Number of Row Tiles: {num_row_tiles}")
    print(f"Number of Column Tiles: {num_col_tiles}")
    print(f"Total Tiles: {total_tiles}")
    print(f"Available PEs: {num_pes}")

def print_tile_assignment(tile_row: int, tile_col: int, k: int, pe_idx: int, indent: int = 0):
    """Print tile assignment information including both activation and weight tile locations"""
    indent_str = " " * indent
    print(f"\n{indent_str}Tile Assignment:")
    print(f"{indent_str}├─ Activation Tile Location: ({tile_row}, {k})")
    print(f"{indent_str}├─ Weight Tile Location: ({k}, {tile_col})")
    print(f"{indent_str}├─ K-index: {k}")
    print(f"{indent_str}└─ Assigned to PE: {pe_idx}")

def print_pe_stats(pe_idx: int, stats, indent: int = 0):
    indent_str = " " * indent
    
    print(f"\n{indent_str}┌{'─' * 40}┐")
    print(f"{indent_str}│ Processing Element {pe_idx:<20}│")
    print(f"{indent_str}├{'─' * 40}┤")
    print(f"{indent_str}│ Cycle-level Operations:                │")
    print(f"{indent_str}│   ├─ Masking:  {stats.masking_operations:<4} cycles             │")
    print(f"{indent_str}│   ├─ Shifting: {stats.shifting_operations:<4} cycles             │")
    print(f"{indent_str}│   └─ Addition: {stats.addition_operations:<4} cycles             │")
    print(f"{indent_str}├{'─' * 40}┤")
    print(f"{indent_str}│ Total Operations:                      │")
    print(f"{indent_str}│   ├─ Mask Ops:  {stats.total_mask_ops:<6}                 │")
    print(f"{indent_str}│   ├─ Shifts:    {stats.total_shifts:<6}                 │")
    print(f"{indent_str}│   └─ Additions: {stats.total_additions:<6}                 │")
    print(f"{indent_str}├{'─' * 40}┤")
    print(f"{indent_str}│ Total Cycles: {stats.total_cycles:<21}    │")
    print(f"{indent_str}└{'─' * 40}┘")

def print_system_stats(stats, indent: int = 0):
    """Pretty print system-wide statistics with box drawing characters"""
    indent_str = " " * indent
    
    print(f"\n{indent_str}┌{'─' * 50}┐")
    print(f"{indent_str}│ System-wide Statistics (Parallel Execution)      │")
    print(f"{indent_str}├{'─' * 50}┤")
    print(f"{indent_str}│ Maximum Parallel Operations:                     │")
    print(f"{indent_str}│   ├─ Masking:  {stats.total_parallel_mask_ops:<4} cycles                       │")
    print(f"{indent_str}│   ├─ Shifting: {stats.total_parallel_shifts:<4} cycles                       │")
    print(f"{indent_str}│   └─ Addition: {stats.total_parallel_additions:<4} cycles                       │")
    print(f"{indent_str}├{'─' * 50}┤")
    print(f"{indent_str}│ Total Parallel Execution Time: {stats.total_parallel_cycles:<14}    │")
    print(f"{indent_str}└{'─' * 50}┘")

def format_throughput(ops):
    if ops >= 1e9:
        return f"{ops / 1e9:.2f} Gops/s"
    elif ops >= 1e6:
        return f"{ops / 1e6:.2f} Mops/s"
    elif ops >= 1e3:
        return f"{ops / 1e3:.2f} Kops/s"
    else:
        return f"{ops:.2f} ops/s"

def format_bandwidth(bps):
    """Format bandwidth to human-readable string."""
    if bps >= 1e9:
        return f"{bps / 1e9:.2f} GB/s"
    elif bps >= 1e6:
        return f"{bps / 1e6:.2f} MB/s"
    elif bps >= 1e3:
        return f"{bps / 1e3:.2f} KB/s"
    else:
        return f"{bps:.2f} B/s"

def print_performance_metrics(metrics, indent: int = 0):
    indent_str = " " * indent

    throughput_str = format_throughput(metrics.throughput_ops)
    bandwidth_str = format_bandwidth(metrics.memory_bandwidth_bytes_per_sec)
    latency_str = f"{metrics.system_latency_ns:.2f} ns"

    print(f"\n{indent_str}┌{'─' * 50}┐")
    print(f"{indent_str}│ Performance Metrics                     │")
    print(f"{indent_str}├{'─' * 50}┤")
    print(f"{indent_str}│ Overall Latency         : {latency_str:>15} │")
    print(f"{indent_str}│ Throughput              : {throughput_str:>15} │")
    print(f"{indent_str}│ Memory Bandwidth        : {bandwidth_str:>15} │")
    print(f"{indent_str}└{'─' * 50}┘")

# ----- End of Pretty Printing -----


# ----- Test -----

@contextlib.contextmanager
def suppress_all_output():
    """
    Suppress stdout and stderr (including output from C++ code) by redirecting
    the underlying file descriptors to os.devnull.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

def run_matmul_test(matrix_size, tile_size, num_bits, activation_threshold=0, verbose=True):
    weights = np.random.randint(0, 15, size=(matrix_size, matrix_size), dtype=np.int8)
    activations = np.random.randint(-128, 127, size=(matrix_size, matrix_size), dtype=np.int32)

    if verbose:
        print("\nInput Matrices")
        print("═" * 50)
        print_matrix("Weight Matrix", weights)
        print_matrix("Activation Matrix", activations)

    preprocess_weights(weights, num_bits=num_bits, tile_size=tile_size)

    engine = SIMDEngine("weight_bits.bin")

    if verbose:
        print_matrix_info(engine)

    if verbose:
        result_tile = engine.compute(activations.flatten().tolist(), activation_threshold)
    else:
        with suppress_all_output():
            result_tile = engine.compute(activations.flatten().tolist(), activation_threshold)

    result_array = np.array(result_tile.data).reshape(matrix_size, matrix_size)
    software_reference = np.matmul(activations, weights)

    stats = engine.get_stats()

    if verbose:
        print("\nComputation Results")
        print("═" * 50)
        print_matrix("Hardware Result", result_array)
        print_matrix("Software Reference", software_reference)

        print("\nProcessing Element Stats")
        print("═" * 50)
        num_row_tiles = (matrix_size + tile_size - 1) // tile_size
        num_col_tiles = (matrix_size + tile_size - 1) // tile_size
        for tile_row in range(num_row_tiles):
            for tile_col in range(num_col_tiles):
                for k in range(num_col_tiles):
                    assigned_pe = (tile_row * num_col_tiles * num_col_tiles + tile_col * num_col_tiles + k) % engine.get_num_pes()
                    print_tile_assignment(tile_row, tile_col, k, assigned_pe, indent=2)
                    print_pe_stats(assigned_pe, stats.pe_stats[assigned_pe], indent=4)

        print("\nSystem-wide Stats")
        print("═" * 50)
        print_system_stats(stats, indent=2)
        
        clock_frequency_hz = 1e9  # 1 GHz
        performance_metrics = engine.get_performance_metrics(clock_frequency_hz)
        print_performance_metrics(performance_metrics, indent=2)

    return result_array, software_reference, stats

if __name__ == "__main__":
    matrix_size = 16
    tile_size = 4
    num_bits = 4
    run_matmul_test(matrix_size, tile_size, num_bits, verbose=True) 