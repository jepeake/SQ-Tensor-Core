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
    rows, cols = weights.shape
    
    num_row_tiles = (rows + tile_size - 1) // tile_size
    num_col_tiles = (cols + tile_size - 1) // tile_size
    
    # Decompose into Bit Matrices
    bit_matrices = np.zeros((num_bits, num_row_tiles * num_col_tiles, tile_size, tile_size), dtype=np.uint8)
    
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
                tile_idx += 1
    
    output_file = "weight_bits.bin"
    with open(output_file, "wb") as f:
        # Write Header
        f.write(struct.pack("<IIII", rows, cols, num_bits, tile_size))
        
        # Write bit matrices
        for bit in range(num_bits):
            for tile_idx in range(num_row_tiles * num_col_tiles):
                tile = bit_matrices[bit, tile_idx]
                packed_bytes = []
                bit_count = 0
                current_byte = 0
                
                # Pack Bits into Bytes
                for row in range(tile_size):
                    for col in range(tile_size):
                        current_byte = (current_byte << 1) | tile[row, col]
                        bit_count += 1
                        
                        if bit_count == 8:
                            packed_bytes.append(current_byte)
                            current_byte = 0
                            bit_count = 0
                
                # Handle Remaining Bits in Last Byte
                if bit_count > 0:
                    current_byte <<= (8 - bit_count)
                    packed_bytes.append(current_byte)
                
                f.write(bytes(packed_bytes))
    
    return bytes(packed_bytes)