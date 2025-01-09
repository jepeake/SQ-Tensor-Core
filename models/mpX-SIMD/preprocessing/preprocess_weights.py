import numpy as np
import struct
import os

#  _  _  ____  __  ___  _  _  ____    ____  ____  ____  ____  ____   __    ___  ____  ____  ____   __  ____ 
# / )( \(  __)(  )/ __)/ )( \(_  _)  (  _ \(  _ \(  __)(  _ \(  _ \ /  \  / __)(  __)/ ___)/ ___) /  \(  _ \
# \ /\ / ) _)  )(( (_ \) __ (  )(     ) __/ )   / ) _)  ) __/ )   /(  O )( (__  ) _) \___ \\___ \(  O ))   /
# (_/\_)(____)(__)\___/\_)(_/ (__)   (__)  (__\_)(____)(__)  (__\_) \__/  \___)(____)(____/(____/ \__/(__\_)

def preprocess_weights(weights: np.ndarray, num_bits: int, tile_size: int) -> bytes:
    """
    Preprocess weight matrix into bit-serial tiled format for hardware
    Returns binary format that can be loaded by C++ hardware model
    """
#     print(
#     f"""
#  _  _  ____  __  ___  _  _  ____    ____  ____  ____  ____  ____   __    ___  ____  ____  ____   __  ____ 
# / )( \(  __)(  )/ __)/ )( \(_  _)  (  _ \(  _ \(  __)(  _ \(  _ \ /  \  / __)(  __)/ ___)/ ___) /  \(  _ \\
# \ /\ / ) _)  )(( (_ \) __ (  )(     ) __/ )   / ) _)  ) __/ )   /(  O )( (__  ) _) \___ \\___ \(  O ))   /
# (_/\_)(____)(__)\___/\_)(_/ (__)   (__)  (__\_)(____)(__)  (__\_) \__/  \___)(____)(____/(____/ \__/(__\_)
#     """)
    rows, cols = weights.shape
    print("\nInput Weights Matrix:")
    print("-" * 40)
    print(weights)
    print("-" * 40)
    
    num_row_tiles = (rows + tile_size - 1) // tile_size
    num_col_tiles = (cols + tile_size - 1) // tile_size
    print(f"No. of Tiles: {num_row_tiles}x{num_col_tiles}")
    
    # Decompose into Bit Matrices
    bit_matrices = np.zeros((num_bits, num_row_tiles * num_col_tiles, tile_size, tile_size), dtype=np.uint8)
    print(f"Bit Matrices Shape: {bit_matrices.shape}")
    
    for bit in range(num_bits):
        tile_idx = 0
        for tile_row in range(num_row_tiles):
            row_start = tile_row * tile_size
            row_end = min(row_start + tile_size, rows)
            
            for tile_col in range(num_col_tiles):
                col_start = tile_col * tile_size
                col_end = min(col_start + tile_size, cols)

                tile = weights[row_start:row_end, col_start:col_end]
                bit_tile = (tile >> bit) & 1
                if bit_tile.shape != (tile_size, tile_size):
                    padded = np.zeros((tile_size, tile_size), dtype=np.uint8)
                    padded[:bit_tile.shape[0], :bit_tile.shape[1]] = bit_tile
                    bit_tile = padded
                    
                bit_matrices[bit, tile_idx] = bit_tile
                print(f"\nTile [{tile_row},{tile_col}]:")
                print("-" * 20)
                print(bit_tile)
                tile_idx += 1
    
    # Pack into Binary Format
    packed_data = np.packbits(bit_matrices)
    output_file = "weight_bits.bin"
    with open(output_file, "wb") as f:
        # Write Header: rows, cols, bits, tile_size
        f.write(struct.pack("<IIII", rows, cols, num_bits, tile_size))
        # Write Bit Matrices
        packed_data.tofile(f)
    
    file_size = os.path.getsize(output_file)
    print(f"\nOutput File Stats:")
    print(f"- File Size: {file_size:,} bytes")
    print(f"- Header Size: {4 * 4} bytes")
    print(f"- Data Size: {packed_data.nbytes:,} bytes")
    
    return packed_data