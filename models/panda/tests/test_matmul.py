import numpy as np
from panda import SIMDEngine
from preprocessing.preprocess_weights import preprocess_weights
from typing import List, Dict
import textwrap
import os
import contextlib
import sys
import json

# ----- Pretty Printing -----

def print_matrix(name: str, matrix: np.ndarray, indent: int = 0):
    indent_str = " " * indent
    print(f"\n{indent_str}{name}:")
    print(f"{indent_str}{'=' * (len(name) + 1)}")
    
    max_width = max(len(str(x)) for x in matrix.flatten())
    
    for row in matrix:
        print(indent_str + " ".join(f"{x:>{max_width}}" for x in row))

def print_tile(name: str, tile: np.ndarray, indent: int = 0):
    indent_str = " " * indent
    print(f"\n{indent_str}{name}:")
    print(f"{indent_str}┌{'─' * (tile.shape[1] * 4 + 1)}┐")
    
    for row in tile:
        print(f"{indent_str}│ " + " ".join(f"{x:>2}" for x in row) + " │")
    
    print(f"{indent_str}└{'─' * (tile.shape[1] * 4 + 1)}┘")

def print_pe_assignment(pe_idx: int, weight_tiles: List[np.ndarray], act_tile: np.ndarray, indent: int = 0):
    indent_str = " " * indent
    separator = f"{indent_str}{'─' * 50}"
    
    print(f"\n{separator}")
    print(f"{indent_str}Processing Element {pe_idx}")
    print(f"{indent_str}{separator}")
    
    print_tile("Activation Tile", act_tile, indent + 2)
    
    for bit, w_tile in enumerate(weight_tiles):
        print_tile(f"Weight Tile (Bit {bit})", w_tile, indent + 2)

def print_matrix_info(engine: SIMDEngine):
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
    indent_str = " " * indent
    box_width = 50 
    
    print(f"\n{indent_str}┌{'─' * box_width}┐")
    print(f"{indent_str}│ System Statistics (Parallel Execution)           │")
    print(f"{indent_str}├{'─' * box_width}┤")
    
    content = f" Total Execution Time: {stats.total_parallel_cycles} cycles"
    formatted_line = content.ljust(box_width)
    
    print(f"{indent_str}│{formatted_line}│")
    print(f"{indent_str}└{'─' * box_width}┘")

def format_throughput(ops):
    if ops >= 1e9:
        return f"{ops / 1e9:.2f} GFLOPs/s"
    elif ops >= 1e6:
        return f"{ops / 1e6:.2f} MFLOPs/s"
    elif ops >= 1e3:
        return f"{ops / 1e3:.2f} KFLOPs/s"
    else:
        return f"{ops:.2f} FLOPs/s"

def format_bandwidth(bps):
    if bps >= 1e9:
        return f"{bps / 1e9:.2f} GB/s"
    elif bps >= 1e6:
        return f"{bps / 1e6:.2f} MB/s"
    elif bps >= 1e3:
        return f"{bps / 1e3:.2f} KB/s"
    else:
        return f"{bps:.2f} B/s"

def print_performance_metrics(metrics, indent=0):
    indent_str = " " * indent
    print(f"{indent_str}Performance Metrics:")
    print(f"{indent_str}  System Latency: {metrics.system_latency_ns/1e3:.2f} μs")
    print(f"{indent_str}  Throughput: {metrics.throughput_ops/1e9:.2f} GFLOPS")
    print(f"{indent_str}  Memory Bandwidth: {metrics.memory_bandwidth_bytes_per_sec/1e9:.2f} GB/s")
    print(f"{indent_str}  Arithmetic Intensity: {metrics.arithmetic_intensity:.2f} FLOPS/Byte")
    
    # Hardware costs with automatic unit conversion
    print(f"\n{indent_str}Hardware Costs:")
    
    # Energy conversions
    total_energy = metrics.total_energy_pj
    energy_unit = "pJ"
    if total_energy > 1000000:
        total_energy /= 1000000
        energy_unit = "μJ"
    elif total_energy > 1000:
        total_energy /= 1000
        energy_unit = "nJ"
    
    # Area conversions
    total_area = metrics.total_area_um2
    area_unit = "μm²"
    if total_area > 1000000:
        total_area /= 1000000
        area_unit = "mm²"
    elif total_area > 1000:
        total_area /= 1000
        area_unit = "mm² × 10⁻³"  # 1000 μm² = 0.001 mm²
    
    print(f"{indent_str}  Total Energy: {total_energy:.2f} {energy_unit}")
    print(f"{indent_str}  Total Area: {total_area:.2f} {area_unit}")
    
    # Per-component breakdown with percentage
    print(f"\n{indent_str}Cost Breakdown:")
    
    # For adder energy
    adder_energy = metrics.adder_energy_pj
    adder_energy_unit = "pJ"
    if metrics.adder_energy_pj > 1000000:
        adder_energy /= 1000000
        adder_energy_unit = "μJ"
    elif metrics.adder_energy_pj > 1000:
        adder_energy /= 1000
        adder_energy_unit = "nJ"
    
    # For mask energy
    mask_energy = metrics.mask_energy_pj
    mask_energy_unit = "pJ"
    if metrics.mask_energy_pj > 1000000:
        mask_energy /= 1000000
        mask_energy_unit = "μJ"
    elif metrics.mask_energy_pj > 1000:
        mask_energy /= 1000
        mask_energy_unit = "nJ"
    
    # For adder area
    adder_area = metrics.adder_area_um2
    adder_area_unit = "μm²"
    if metrics.adder_area_um2 > 1000000:
        adder_area /= 1000000
        adder_area_unit = "mm²"
    elif metrics.adder_area_um2 > 1000:
        adder_area /= 1000
        adder_area_unit = "mm² × 10⁻³"
    
    # For mask area
    mask_area = metrics.mask_area_um2
    mask_area_unit = "μm²"
    if metrics.mask_area_um2 > 1000000:
        mask_area /= 1000000
        mask_area_unit = "mm²"
    elif metrics.mask_area_um2 > 1000:
        mask_area /= 1000
        mask_area_unit = "mm² × 10⁻³"
    
    print(f"{indent_str}  Adder Energy: {adder_energy:.2f} {adder_energy_unit} ({metrics.adder_energy_pj/metrics.total_energy_pj*100:.1f}%)")
    print(f"{indent_str}  Masking Energy: {mask_energy:.2f} {mask_energy_unit} ({metrics.mask_energy_pj/metrics.total_energy_pj*100:.1f}%)")
    print(f"{indent_str}  Adder Area: {adder_area:.2f} {adder_area_unit} ({metrics.adder_area_um2/metrics.total_area_um2*100:.1f}%)")
    print(f"{indent_str}  Masking Area: {mask_area:.2f} {mask_area_unit} ({metrics.mask_area_um2/metrics.total_area_um2*100:.1f}%)")
    
    # Calculate power consumption
    system_time_sec = metrics.system_latency_ns * 1e-9
    
    # Dynamic power (from the energy values we already have)
    dynamic_power_w = metrics.total_energy_pj * 1e-12 / system_time_sec if system_time_sec > 0 else 0
    
    # Static power estimation (based on area)
    # Typical leakage power density for 65nm - 90nm process: ~0.2-0.3 W/mm²
    # For more advanced nodes like 28nm or smaller: ~0.1 W/mm²
    leakage_power_density = 0.1  # W/mm²
    static_power_w = (metrics.total_area_um2 * 1e-6) * leakage_power_density  # Convert μm² to mm²
    
    # Total power
    total_power_w = dynamic_power_w + static_power_w
    
    # Format power values appropriately
    def format_power(power_w):
        if power_w < 1e-6:
            return f"{power_w * 1e9:.2f} nW"
        elif power_w < 1e-3:
            return f"{power_w * 1e6:.2f} μW"
        elif power_w < 1:
            return f"{power_w * 1e3:.2f} mW"
        else:
            return f"{power_w:.2f} W"
    
    print(f"\n{indent_str}Power Estimation:")
    print(f"{indent_str}  Dynamic Power: {format_power(dynamic_power_w)} ({dynamic_power_w/total_power_w*100:.1f}%)")
    print(f"{indent_str}  Static Power: {format_power(static_power_w)} ({static_power_w/total_power_w*100:.1f}%)")
    print(f"{indent_str}  Total Power: {format_power(total_power_w)}")
    
    # Energy efficiency (FLOPS/Watt)
    # Convert pJ to W: 1 W = 1 J/s = 10^12 pJ/s
    flops_per_joule = metrics.throughput_ops / total_power_w if total_power_w > 0 else 0
    
    # Use appropriate unit based on magnitude
    efficiency_value = flops_per_joule
    efficiency_unit = "FLOPS/W"
    
    if efficiency_value >= 1e9:
        efficiency_value /= 1e9
        efficiency_unit = "GFLOPS/W"
    elif efficiency_value >= 1e6:
        efficiency_value /= 1e6
        efficiency_unit = "MFLOPS/W"
    
    print(f"\n{indent_str}Energy Efficiency: {efficiency_value:.2f} {efficiency_unit}")

def print_grouped_pe_assignments_and_stats(engine: SIMDEngine, stats, matrix_size: int, tile_size: int):
    num_row_tiles = (matrix_size + tile_size - 1) // tile_size
    num_col_tiles = (matrix_size + tile_size - 1) // tile_size
    num_pes = engine.get_num_pes()

    # Create a Dictionary to Collect Assignments per PE
    # In pe_array - scheduling each job is assigned by: assigned_pe = global_job_index % num_pes,
    # where global_job_index increments across all jobs
    pe_assignments = {pe: [] for pe in range(num_pes)}
    global_job = 0
    for tile_row in range(num_row_tiles):
        for tile_col in range(num_col_tiles):
            for k in range(num_col_tiles):
                assigned_pe = global_job % num_pes
                assignment_str = f"Job {global_job}: Activation Tile ({tile_row}, {k}), Weight Tile ({k}, {tile_col})"
                pe_assignments[assigned_pe].append(assignment_str)
                global_job += 1

    for pe in range(num_pes):
        print(f"\nProcessing Element {pe}")
        print("─" * 50)
        if pe_assignments[pe]:
            print("Assigned Jobs:")
            for assign in pe_assignments[pe]:
                print("  " + assign)
        else:
            print("No assignments")

        pe_stat = stats.pe_stats[pe]
        # print("\nAggregated Operation Cycles:")
        # print(f"  Masking   : {pe_stat.masking_operations} cycles")
        # print(f"  Shifting  : {pe_stat.shifting_operations} cycles")
        # print(f"  Addition  : {pe_stat.addition_operations} cycles")
        print("\n Latency:")
        print(f"  Masking   : {pe_stat.masking_operations} cycles (per tile)")
        print(f"  Shifting  : {pe_stat.shifting_operations} cycles (per tile)")
        print(f"  Addition  : {pe_stat.addition_operations} cycles (per tile)")
        print(f"  Total     : {pe_stat.total_cycles} cycles (pipelined)")
        print("─" * 50)

# ----- End of Pretty Printing -----

# ----- Test -----

@contextlib.contextmanager
def suppress_all_output():
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

def run_matmul_test(matrix_size, tile_size, num_bits, activation_threshold=0, weight_sparsity=0.0, verbose=True):
    # Generate weights with specified sparsity
    weights = np.random.randint(0, 15, size=(matrix_size, matrix_size), dtype=np.int8)
    
    # Apply sparsity by setting some values to zero
    if weight_sparsity > 0:
        sparsity_mask = np.random.choice(
            [0, 1], 
            size=weights.shape, 
            p=[weight_sparsity, 1-weight_sparsity]
        )
        weights = weights * sparsity_mask
        
    activations = np.random.randint(-128, 127, size=(matrix_size, matrix_size), dtype=np.int32)

    print("\nInput Matrices Summary")
    print("═" * 50)
    print(f"Weight Matrix: shape {weights.shape}, sparsity {weight_sparsity:.2f}")
    print(f"Activation Matrix: shape {activations.shape}")

    if verbose:
        print("\nDetailed Input Matrices")
        print_matrix("Weight Matrix", weights)
        print_matrix("Activation Matrix", activations)

    preprocess_weights(weights, num_bits=num_bits, tile_size=tile_size)

    engine = SIMDEngine("weight_bits.bin")

    print_matrix_info(engine)

    if verbose:
        result_tile = engine.compute(activations.flatten().tolist(), activation_threshold)
    else:
        with suppress_all_output():
            result_tile = engine.compute(activations.flatten().tolist(), activation_threshold)

    result_array = np.array(result_tile.data).reshape(matrix_size, matrix_size)
    software_reference = np.matmul(activations, weights)

    stats = engine.get_stats()

    print("\nComputation Results")
    print("═" * 50)
    print_matrix("Hardware Result", result_array)
    print_matrix("Software Reference", software_reference)

    # Compute job assignments per processing element
    num_row_tiles = (matrix_size + tile_size - 1) // tile_size
    num_col_tiles = (matrix_size + tile_size - 1) // tile_size
    num_pes = engine.get_num_pes()

    # The scheduling is assumed to assign jobs round-robin:
    pe_assignments = {pe: [] for pe in range(num_pes)}
    global_job = 0
    for tile_row in range(num_row_tiles):
        for tile_col in range(num_col_tiles):
            for k in range(num_col_tiles):
                assigned_pe = global_job % num_pes
                # Format shows the job number and the tile indices
                assignment_str = f"[Job {global_job}: ActTile=({tile_row},{k}), WTile=({k},{tile_col})]"
                pe_assignments[assigned_pe].append(assignment_str)
                global_job += 1

    # Print the processing element stats along with its job assignments on the same line
    print("\nProcessing Element Stats Summary")
    print("═" * 50)
    for idx, pe_stat in enumerate(stats.pe_stats):
        assignments_str = " ".join(pe_assignments[idx])
        print(f"PE {idx}: Total Cycles = {pe_stat.total_cycles}, "
              f"Mask Ops = {pe_stat.total_mask_ops}, "
              f"Shifts = {pe_stat.total_shifts}, "
              f"Additions = {pe_stat.total_additions}, "
              f"Assigned Jobs = {assignments_str}")

    if verbose:
        print_grouped_pe_assignments_and_stats(engine, stats, matrix_size, tile_size)

    print("\nSystem Stats Summary")
    print("═" * 50)
    print_system_stats(stats, indent=2)
    clock_frequency_hz = 1e9  # 1 GHz
    performance_metrics = engine.get_performance_metrics(clock_frequency_hz)
    print_performance_metrics(performance_metrics, indent=2)

    return result_array, software_reference, stats

if __name__ == "__main__":
    import sys
    this_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(this_dir, "..", "src", "core", "panda_config.json")
    
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
    except Exception as e:
        print(f"Error reading configuration file ({config_path}): {e}")
        config_data = {}
    
    matrix_size = config_data.get("matrix_size", 16)
    tile_size = config_data.get("tile_size", 4)
    num_bits = 4
    weight_sparsity = config_data.get("weight_sparsity", 0.0)

    verbose = "--verbose" in sys.argv
    run_matmul_test(matrix_size, tile_size, num_bits, weight_sparsity=weight_sparsity, verbose=verbose) 