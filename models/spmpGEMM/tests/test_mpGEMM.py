import numpy as np
from spmp_gemm import SIMDEngine
from preprocessing.preprocess_weights import preprocess_weights
from typing import List, Dict
import textwrap


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
    print(f"{separator}")
    
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

# ----- End of Pretty Printing -----


# ----- Test -----

def test_mpGEMM():
    N = 8
    TILE_SIZE = 4
    NUM_BITS = 4
    
    weights = np.random.randint(0, 15, size=(N, N), dtype=np.int8)
    activations = np.random.randint(-128, 127, size=(N, N), dtype=np.int32)
    
    print("\nInput Matrices")
    print("═" * 50)
    print_matrix("Weight Matrix", weights)
    print_matrix("Activation Matrix", activations)
    
    preprocess_weights(weights, num_bits=NUM_BITS, tile_size=TILE_SIZE)
    
    engine = SIMDEngine("weight_bits.bin")
    
    print_matrix_info(engine)
    
    result = engine.compute(activations.flatten().tolist(), activation_threshold=0)
    result_array = np.array(result.data).reshape(N, N)
    
    print("\nComputation Results")
    print("═" * 50)
    print_matrix("Hardware Result", result_array)
    print_matrix("Software Reference", np.matmul(activations, weights))
    
    stats = engine.get_stats()
    
    print("\nProcessing Element Stats")
    print("═" * 50)
    
    num_row_tiles = (N + TILE_SIZE - 1) // TILE_SIZE
    num_col_tiles = (N + TILE_SIZE - 1) // TILE_SIZE
    
    pe_idx = 0
    for tile_row in range(num_row_tiles):
        for tile_col in range(num_col_tiles):
            for k in range(num_col_tiles):
                print_tile_assignment(tile_row, tile_col, k, pe_idx, indent=2)
                print_pe_stats(pe_idx, stats.pe_stats[pe_idx], indent=4)
                pe_idx += 1
    
    print("\nSystem-wide Stats")
    print("═" * 50)
    print_system_stats(stats, indent=2)

if __name__ == "__main__":
    test_mpGEMM() 