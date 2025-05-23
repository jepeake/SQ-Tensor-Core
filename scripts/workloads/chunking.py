import argparse
import struct
import json
from typing import Dict, List, Tuple, NamedTuple
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm


#    █████████  █████                            █████       ███                     
#   ███░░░░░███░░███                            ░░███       ░░░                      
#  ███     ░░░  ░███████   █████ ████ ████████   ░███ █████ ████  ████████    ███████
# ░███          ░███░░███ ░░███ ░███ ░░███░░███  ░███░░███ ░░███ ░░███░░███  ███░░███
# ░███          ░███ ░███  ░███ ░███  ░███ ░███  ░██████░   ░███  ░███ ░███ ░███ ░███
# ░░███     ███ ░███ ░███  ░███ ░███  ░███ ░███  ░███░░███  ░███  ░███ ░███ ░███ ░███
#  ░░█████████  ████ █████ ░░████████ ████ █████ ████ █████ █████ ████ █████░░███████
#   ░░░░░░░░░  ░░░░ ░░░░░   ░░░░░░░░ ░░░░ ░░░░░ ░░░░ ░░░░░ ░░░░░ ░░░░ ░░░░░  ░░░░░███
#                                                                            ███ ░███
#                                                                           ░░██████ 
#                                                                            ░░░░░░  

# Script to Break Down Sampled Matrices into 256x256 Chunks


@dataclass
class ChunkMetadata:
    """Metadata for a matrix chunk"""
    original_name: str               # Original Tensor Name
    original_shape: Tuple[int, int]  # Original Matrix Dimensions
    chunk_index: int                 # Index of this Chunk (0, 1, 2, ...)
    total_chunks: int                # Total Number of Chunks from this Matrix
    row_start: int                   # Starting Row in Original Matrix
    row_end: int                     # Ending Row in Original Matrix (Exclusive)
    col_start: int                   # Starting Column in Original Matrix
    col_end: int                     # Ending Column in Original Matrix (Exclusive)
    chunk_shape: Tuple[int, int]     # Actual Chunk Dimensions (May be < 256x256 for Edges)
    chunk_file: str                  # Filename of this Chunk


@dataclass
class MatrixHeader:
    magic: int
    num_dims: int
    shape: List[int]


def read_binary_matrix(file_path: Path) -> np.ndarray:
    """Read a matrix from binary format"""
    with open(file_path, "rb") as f:
        # Read Header
        magic = struct.unpack('I', f.read(4))[0]
        if magic != 0x4D545258:  # 'MTRX'
            raise ValueError(f"Invalid magic number in {file_path}")
        
        num_dims = struct.unpack('I', f.read(4))[0]
        shape = []
        for _ in range(num_dims):
            shape.append(struct.unpack('I', f.read(4))[0])
        
        # Calculate Total Size and Read Data
        total_size = np.prod(shape)
        data = np.frombuffer(f.read(total_size * 4), dtype=np.float32)
        
        return data.reshape(shape)


def write_binary_chunk(chunk: np.ndarray, file_path: Path) -> None:
    """Write a matrix chunk in binary format"""
    with open(file_path, "wb") as f:
        # Write Header
        f.write(struct.pack('I', 0x4D545258))  # Magic: 'MTRX'
        f.write(struct.pack('I', len(chunk.shape)))  # Number of Dimensions
        for dim in chunk.shape:
            f.write(struct.pack('I', dim))  # Each Dimension Size
        
        # Write Data
        chunk_f32 = chunk.astype(np.float32)
        f.write(chunk_f32.tobytes('C'))


def load_sampled_matrices(input_dir: Path, format_type: str) -> List[Tuple[str, np.ndarray, dict]]:
    """Load Sampled Matrices from Either Binary or Numpy Format"""
    matrices = []
    
    if format_type == "binary":
        # Read Metadata
        metadata_file = input_dir / "metadata.txt"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        print(f"Reading Metadata: {metadata_file}")
        
        matrix_info = {}
        current_matrix = None
        
        with open(metadata_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                original_line = line
                line = line.strip()
                
                if line.startswith('num_matrices:'):
                    num_matrices = int(line.split(':')[1].strip())
                    print(f"Expected {num_matrices} Matrices")
                    continue
                
                if line.startswith('matrix_'):
                    current_matrix = line.rstrip(':')
                    matrix_info[current_matrix] = {}
                    continue
                
                # Check for Indented Lines (Metadata Fields)
                if original_line.startswith('  ') and current_matrix and ':' in line:
                    try:
                        key, value = line.split(': ', 1)
                        matrix_info[current_matrix][key] = value
                    except ValueError:
                        print(f"Warning: Could not parse line {line_num}: {original_line.strip()}")
                        continue
        
        print(f"Parsed Metadata for {len(matrix_info)} Matrices")
        
        for matrix_key, info in matrix_info.items():
            if 'file' in info and 'name' in info:
                matrix_file = input_dir / info['file']
                if matrix_file.exists():
                    try:
                        print(f"Loading {info['name']} from {info['file']}")
                        matrix = read_binary_matrix(matrix_file)
                        matrices.append((info['name'], matrix, info))
                    except Exception as e:
                        print(f"Error loading {matrix_file}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"Matrix file not found: {matrix_file}")
            else:
                print(f"Missing required fields for {matrix_key}: file={info.get('file', 'MISSING')}, name={info.get('name', 'MISSING')}")
                    
    elif format_type == "numpy":
        # Load from .npz File
        npz_file = input_dir / "matrices.npz"
        metadata_file = input_dir / "metadata.txt"
        
        if not npz_file.exists():
            raise FileNotFoundError(f"NumPy file not found: {npz_file}")
        
        # Load Matrices
        npz_data = np.load(npz_file)
        
        # Parse Metadata if Available
        metadata = {}
        if metadata_file.exists():
            current_matrix = None
            with open(metadata_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('matrix_'):
                        current_matrix = line.rstrip(':')
                        metadata[current_matrix] = {}
                    elif line.startswith('  ') and current_matrix:
                        try:
                            key, value = line.strip().split(': ', 1)
                            metadata[current_matrix][key] = value
                        except ValueError:
                            continue
        
        # Combine Matrices with Metadata
        for key in npz_data.files:
            matrix = npz_data[key]
            info = metadata.get(key, {'name': key})
            matrices.append((info.get('name', key), matrix, info))
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
    return matrices


def chunk_matrix(
    matrix: np.ndarray, 
    matrix_name: str, 
    chunk_size: int = 256
) -> List[Tuple[np.ndarray, ChunkMetadata]]:
    """Break a Matrix into Chunks of Specified Size"""
    
    if len(matrix.shape) != 2:
        raise ValueError(f"Only 2D matrices supported, got shape {matrix.shape}")
    
    rows, cols = matrix.shape
    chunks = []
    chunk_index = 0
    
    # Calculate Total Number of Chunks
    rows_chunks = (rows + chunk_size - 1) // chunk_size
    cols_chunks = (cols + chunk_size - 1) // chunk_size
    total_chunks = rows_chunks * cols_chunks
    
    # Generate Chunks
    for row_idx in range(rows_chunks):
        for col_idx in range(cols_chunks):
            # Calculate Chunk Boundaries
            row_start = row_idx * chunk_size
            row_end = min(row_start + chunk_size, rows)
            col_start = col_idx * chunk_size
            col_end = min(col_start + chunk_size, cols)
            
            # Extract Chunk
            chunk = matrix[row_start:row_end, col_start:col_end]
            
            chunk_file = f"chunk_{chunk_index:04d}.bin"
            metadata = ChunkMetadata(
                original_name=matrix_name,
                original_shape=(rows, cols),
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                chunk_shape=chunk.shape,
                chunk_file=chunk_file
            )
            
            chunks.append((chunk, metadata))
            chunk_index += 1
    
    return chunks


def save_chunks(
    all_chunks: List[Tuple[np.ndarray, ChunkMetadata]], 
    output_dir: Path
) -> None:
    """Save All Chunks and Metadata"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Each Chunk as Binary File
    for chunk, metadata in tqdm(all_chunks, desc="Saving Chunks"):
        chunk_path = output_dir / metadata.chunk_file
        write_binary_chunk(chunk, chunk_path)
    
    # Save Comprehensive Metadata as JSON
    metadata_list = [asdict(metadata) for _, metadata in all_chunks]
    
    with open(output_dir / "chunks_metadata.json", "w") as f:
        json.dump({
            "total_chunks": len(all_chunks),
            "chunk_size": 256,  
            "chunks": metadata_list
        }, f, indent=2)
    
    # Save Human-Readable Summary
    with open(output_dir / "chunks_summary.txt", "w") as f:
        f.write(f"Matrix Chunks Summary\n")
        f.write(f"===================\n\n")
        f.write(f"Total chunks: {len(all_chunks)}\n")
        f.write(f"Chunk size: 256x256 (max)\n\n")
        
        # Group by Original Matrix
        by_matrix = {}
        for _, metadata in all_chunks:
            if metadata.original_name not in by_matrix:
                by_matrix[metadata.original_name] = []
            by_matrix[metadata.original_name].append(metadata)
        
        f.write("Original Matrices:\n")
        f.write("-" * 80 + "\n")
        
        for matrix_name, chunks in by_matrix.items():
            orig_shape = chunks[0].original_shape
            f.write(f"\n{matrix_name}\n")
            f.write(f"  Original shape: {orig_shape[0]}x{orig_shape[1]}\n")
            f.write(f"  Number of chunks: {len(chunks)}\n")
            f.write(f"  Chunk files: chunk_{chunks[0].chunk_index:04d}.bin to chunk_{chunks[-1].chunk_index:04d}.bin\n")


def reconstruct_matrix_from_chunks(
    chunks_metadata_file: Path,
    chunks_dir: Path,
    matrix_name: str
) -> np.ndarray:
    """Reconstruct a Matrix from its Chunks (Utility Function for Verification)"""
    
    with open(chunks_metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Find Chunks for this Matrix
    matrix_chunks = [
        chunk for chunk in metadata['chunks'] 
        if chunk['original_name'] == matrix_name
    ]
    
    if not matrix_chunks:
        raise ValueError(f"No chunks found for matrix: {matrix_name}")
    
    # Get Original Shape
    orig_shape = matrix_chunks[0]['original_shape']
    reconstructed = np.zeros(orig_shape, dtype=np.float32)
    
    # Load and Place Each Chunk
    for chunk_meta in matrix_chunks:
        chunk_file = chunks_dir / chunk_meta['chunk_file']
        chunk = read_binary_matrix(chunk_file)
        
        # Place Chunk in Reconstructed Matrix
        row_start, row_end = chunk_meta['row_start'], chunk_meta['row_end']
        col_start, col_end = chunk_meta['col_start'], chunk_meta['col_end']
        reconstructed[row_start:row_end, col_start:col_end] = chunk
    
    return reconstructed


def main():
    parser = argparse.ArgumentParser(
        description="Break Down Sampled Matrices into 256x256 Chunks"
    )
    parser.add_argument("input_dir", help="Directory Containing Sampled Matrices")
    parser.add_argument("--output-dir", "-o", default="./matrix_chunks",
                       help="Base Output Directory for Chunks (default: ./matrix_chunks)")
    parser.add_argument("--format", choices=["binary", "numpy"], default="binary",
                       help="Input Format of Sampled Matrices (default: binary)")
    parser.add_argument("--chunk-size", type=int, default=256,
                       help="Size of Square Chunks (default: 256)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify Reconstruction by Loading First Matrix")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    
    # Extract model name from the input directory path
    model_name = input_path.name  # Gets the last component of the path
    
    # Create output directory with model name
    output_path = Path(args.output_dir) / model_name
    
    print(f"Loading Sampled Matrices: {input_path}")
    matrices = load_sampled_matrices(input_path, args.format)
    print(f"Loaded {len(matrices)} Matrices")
    
    if not matrices:
        print("No Matrices Found")
        return
    
    all_chunks = []
    print(f"Breaking Matrices into {args.chunk_size}x{args.chunk_size} Chunks")
    
    for matrix_name, matrix, info in tqdm(matrices, desc="Processing Matrices"):
        if len(matrix.shape) != 2:
            continue
            
        chunks = chunk_matrix(matrix, matrix_name, args.chunk_size)
        all_chunks.extend(chunks)
        print(f"  {matrix_name}: {matrix.shape} → {len(chunks)} chunks")
    
    print(f"Chunks Generated: {len(all_chunks)}")
    
    print(f"Saving Chunks to: {output_path}")
    save_chunks(all_chunks, output_path)
    
    if args.verify and matrices:
        print("\nVerifying Reconstruction")
        first_matrix_name = matrices[0][0]
        original = matrices[0][1]
        
        try:
            reconstructed = reconstruct_matrix_from_chunks(
                output_path / "chunks_metadata.json",
                output_path,
                first_matrix_name
            )
            
            if np.allclose(original, reconstructed, rtol=1e-6):
                print(f"✓ Verification Passed for {first_matrix_name}")
            else:
                print(f"✗ Verification Failed for {first_matrix_name}")
                print(f"  Max Difference: {np.max(np.abs(original - reconstructed))}")
        except Exception as e:
            print(f"✗ Verification Error: {e}")
    
if __name__ == "__main__":
    main() 