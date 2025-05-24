import argparse
import re
import struct
import random
from typing import Dict, List, Tuple, NamedTuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from tqdm import tqdm

try:
    import onnx
    import onnx.numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ERROR: ONNX not available. Install with: pip install onnx")
    exit(1)


#   █████████                                      ████                           ███████    ██████   █████ ██████   █████ █████ █████
#  ███░░░░░███                                    ░░███                         ███░░░░░███ ░░██████ ░░███ ░░██████ ░░███ ░░███ ░░███ 
# ░███    ░░░   ██████   █████████████   ████████  ░███   ██████  ████████     ███     ░░███ ░███░███ ░███  ░███░███ ░███  ░░███ ███  
# ░░█████████  ░░░░░███ ░░███░░███░░███ ░░███░░███ ░███  ███░░███░░███░░███   ░███      ░███ ░███░░███░███  ░███░░███░███   ░░█████   
#  ░░░░░░░░███  ███████  ░███ ░███ ░███  ░███ ░███ ░███ ░███████  ░███ ░░░    ░███      ░███ ░███ ░░██████  ░███ ░░██████    ███░███  
#  ███    ░███ ███░░███  ░███ ░███ ░███  ░███ ░███ ░███ ░███░░░   ░███        ░░███     ███  ░███  ░░█████  ░███  ░░█████   ███ ░░███ 
# ░░█████████ ░░████████ █████░███ █████ ░███████  █████░░██████  █████        ░░░███████░   █████  ░░█████ █████  ░░█████ █████ █████
#  ░░░░░░░░░   ░░░░░░░░ ░░░░░ ░░░ ░░░░░  ░███░░░  ░░░░░  ░░░░░░  ░░░░░           ░░░░░░░    ░░░░░    ░░░░░ ░░░░░    ░░░░░ ░░░░░ ░░░░░ 
#                                        ░███                                                                                         
#                                        █████                                                                                        
#                                       ░░░░░                                                                                                                                                                                                                  
            
# Script to Sample Representative Weight Matrices from ONNX Model Files


@dataclass
class LayerInfo:
    name: str
    layer_type: str  
    layer_index: int  
    matmul_count: int  
    tensor_names: List[str]  


@dataclass
class TensorMetadata:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    layer_info: LayerInfo


def load_onnx_weights(model_path: str) -> Dict[str, np.ndarray]:
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Library Not Available")
    
    print(f"Loading ONNX model from: {model_path}")
    model = onnx.load(model_path)
    
    weights = {}
    
    # Extract Weights From Model Initialisers
    for initializer in model.graph.initializer:
        name = initializer.name
        
        # Convert ONNX Tensor To Numpy Array
        try:
            array = onnx.numpy_helper.to_array(initializer)
            
            # Only Include 2D+ Tensors (Matrices)
            if len(array.shape) >= 2:
                weights[name] = array
                print(f"  Loaded: {name} -> {array.shape} ({array.dtype})")
            else:
                print(f"  Skipped: {name} -> {array.shape} (Not A Matrix)")
                
        except Exception as e:
            print(f"  Failed To Load {name}: {e}")
            continue
    
    print(f"Successfully Loaded {len(weights)} Weight Matrices")
    return weights


def parse_onnx_tensor_name(name: str) -> Tuple[str, int, str]:
    """Parse ONNX Tensor Names To Identify Layer Types
    
    ONNX Models Often Have Different Naming Conventions Than Safetensors.
    Common Patterns Include:
    - /encoder/layers.0/attention/self/query/MatMul;weight
    - encoder.layer.0.attention.attention.query.weight
    - blocks.0.attn.qkv.weight
    - /model/layers.0/self_attn/q_proj/MatMul;weight
    - onnx::MatMul_1753 (Auto-Generated Names)
    """
    
    # Clean Up ONNX-Specific Prefixes And Suffixes
    clean_name = name.replace('/MatMul;weight', '.weight')
    clean_name = clean_name.replace('/', '.')
    clean_name = clean_name.lstrip('.')
    
    # ONNX Vision Transformer Patterns
    vit_patterns = [
        (r'(?:encoder\.)?(?:layer|blocks?)\.(\d+)\.(?:self_)?(?:attn|attention)\.(.+)', 'attention'),
        (r'(?:encoder\.)?(?:layer|blocks?)\.(\d+)\.(?:mlp|intermediate)\.(.+)', 'mlp'),
        (r'(?:encoder\.)?(?:layer|blocks?)\.(\d+)\.(?:output)\.(.+)', 'mlp'),  # ViT output layer
        (r'(?:embeddings|patch_embed|pos_embed)\.(.+)', 'embedding'),
        (r'(?:classifier|head|pooler)\.(.+)', 'output'),
        (r'(?:encoder\.)?(?:layer|blocks?)\.(\d+)\..*(?:norm|layernorm)\.(.+)', 'norm'),
    ]
    
    # Standard Transformer Patterns  
    standard_patterns = [
        (r'(?:model\.)?layers?\.(\d+)\.(?:self_)?(?:attn|attention)\.(.+)', 'attention'),
        (r'(?:model\.)?layers?\.(\d+)\.(?:mlp|feed_forward)\.(.+)', 'mlp'),
        (r'(?:model\.)?(?:embed_tokens|token_embeddings|embeddings)\.(.+)', 'embedding'),
        (r'(?:model\.)?(?:lm_head|output|head)\.(.+)', 'output'),
        (r'(?:model\.)?layers?\.(\d+)\..*(?:norm|ln)\.(.+)', 'norm'),
        (r'(?:model\.)?layers?\.(\d+)\.(.+)', 'other'),
    ]
    
    all_patterns = vit_patterns + standard_patterns
    
    # Try Standard Patterns First
    for pattern, layer_type in all_patterns:
        match = re.search(pattern, clean_name, re.IGNORECASE)
        if match:
            if layer_type in ['embedding', 'output']:
                return layer_type, 0, match.group(1)
            else:
                try:
                    layer_idx = int(match.group(1))
                    component = match.group(2) if len(match.groups()) > 1 else ''
                    return layer_type, layer_idx, component
                except (ValueError, IndexError):
                    continue
    
    # Handle ONNX Auto-Generated Names 
    # These Need To Be Identified By Context And Matrix Shapes
    if 'onnx::' in name or 'MatMul' in name:
        # Extract Number From ONNX Name For Rough Ordering
        number_match = re.search(r'(\d+)', name)
        if number_match:
            tensor_id = int(number_match.group(1))
            # Use A Simple Heuristic: Group Sequential Tensors
            # This Is Approximate But Better Than Marking Everything As "Unknown"
            estimated_layer = (tensor_id % 100) // 6  # Rough Estimate For ViT Layers
            return 'onnx_generated', estimated_layer, f'tensor_{tensor_id}'
    
    # If All Else Fails, Mark As Unknown But Try To Extract Any Numbers
    number_match = re.search(r'(\d+)', clean_name)
    layer_idx = int(number_match.group(1)) if number_match else 0
    
    return 'unknown', layer_idx, clean_name


def estimate_matmul_operations_with_shapes(tensor_dict: Dict[str, np.ndarray]) -> Dict[str, LayerInfo]:
    """Enhanced Matmul Estimation That Uses Both Names And Matrix Shapes
    
    For ViT Models, We Can Identify Layer Types By Matrix Dimensions:
    - 768x768: Attention matrices (Q, K, V, output projection)
    - 768x3072: MLP intermediate (FC1)  
    - 3072x768: MLP output (FC2)
    - 1000x768: Classifier head
    - Other Shapes: Embeddings, etc.
    """
    
    layers = defaultdict(lambda: {'tensors': [], 'type': 'unknown', 'matmuls': 0})
    shape_based_analysis = {}
    
    print("\nAnalyzing tensor names and shapes for layer identification...")
    
    # First Pass: Analyze Shapes To Understand The Model Architecture
    shape_counts = defaultdict(int)
    for name, tensor in tensor_dict.items():
        if len(tensor.shape) >= 2:
            shape_key = 'x'.join(map(str, tensor.shape))
            shape_counts[shape_key] += 1
    
    print(f"Common matrix shapes found:")
    for shape, count in sorted(shape_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {shape}: {count} matrices")
    
    # Identify ViT Architecture Dimensions
    vit_dim = None
    mlp_dim = None
    
    # Look For Common ViT Patterns
    for shape_str, count in shape_counts.items():
        shape_parts = shape_str.split('x')
        if len(shape_parts) == 2:
            h, w = int(shape_parts[0]), int(shape_parts[1])
            
            # Square Matrices Likely Attention (768x768, 512x512, Etc.)
            if h == w and count >= 12:  
                vit_dim = h
                print(f"  Detected ViT dimension: {vit_dim}")
                
            # MLP Dimensions (Typically 4x The Embedding Dim)
            if h == w * 4 and count >= 3:  
                mlp_dim = h
                print(f"  Detected MLP dimension: {mlp_dim}")
                
    # Second Pass: Group Tensors With Enhanced Logic
    for name, tensor in tensor_dict.items():
        if len(tensor.shape) < 2:
            continue
            
        # Get Initial Classification From Name Parsing
        layer_type, layer_idx, component = parse_onnx_tensor_name(name)
        
        # Enhance Classification Using Matrix Shapes For ViT
        if vit_dim:
            h, w = tensor.shape[0], tensor.shape[1]
            
            # Override Layer Type Based On Shape Analysis
            if h == w == vit_dim:
                layer_type = 'attention'  # Square Matrices = Attention
            elif h == vit_dim and w == mlp_dim:
                layer_type = 'mlp'  # ViT -> MLP Intermediate
            elif h == mlp_dim and w == vit_dim:
                layer_type = 'mlp'  # MLP -> ViT Output
            elif w == vit_dim and h > vit_dim * 2:
                layer_type = 'output'  # Classifier (1000x768 etc.)
            elif h == vit_dim and 'embed' in name.lower():
                layer_type = 'embedding'
                
        # For ONNX Generated Names, Estimate Layer Based On Tensor Ordering
        if layer_type == 'onnx_generated':
            # Use Tensor Shape To Determine Type
            h, w = tensor.shape[0], tensor.shape[1]
            if vit_dim and h == w == vit_dim:
                layer_type = 'attention'
            elif vit_dim and mlp_dim and ((h == vit_dim and w == mlp_dim) or (h == mlp_dim and w == vit_dim)):
                layer_type = 'mlp'
                
        layer_key = f"{layer_type}_{layer_idx}"
        
        layers[layer_key]['tensors'].append(name)
        layers[layer_key]['type'] = layer_type
        layers[layer_key]['index'] = layer_idx
        
        print(f"  {name} {tensor.shape} -> {layer_key} ({layer_type})")
    
    # Third Pass: Estimate Matmul Operations With Better Heuristics
    layer_infos = {}
    
    print("\nEstimating matmul operations per layer...")
    
    for layer_key, layer_data in layers.items():
        layer_type = layer_data['type']
        layer_idx = layer_data['index']
        tensors = layer_data['tensors']
        
        # For ONNX Models, Count All 2D Tensors (Not Just Those With "Weight" In Name)
        matrix_tensors = [name for name in tensors if len(tensor_dict[name].shape) >= 2]
        
        # Also Count Traditional Weight Tensors For Hybrid Models
        weight_tensors = [name for name in tensors if 'weight' in name.lower()]
        
        # Enhanced Matmul Estimation
        if layer_type == 'attention':
            # ViT Attention: Q, K, V, Output = 4 Matmuls Minimum
            # Count Actual Matrices To Be More Accurate
            square_matrices = [name for name in tensors if len(tensor_dict[name].shape) == 2 
                             and tensor_dict[name].shape[0] == tensor_dict[name].shape[1]]
            matmul_count = max(4, len(square_matrices), len(matrix_tensors))
            
        elif layer_type == 'mlp':
            # MLP: FC1 + FC2 = 2 Matmuls Minimum
            # Count Rectangular Matrices For MLP Layers
            rectangular_matrices = [name for name in tensors if len(tensor_dict[name].shape) == 2 
                                  and tensor_dict[name].shape[0] != tensor_dict[name].shape[1]]
            matmul_count = max(2, len(rectangular_matrices), len(matrix_tensors))
            
        elif layer_type == 'embedding':     
            matmul_count = 1
            
        elif layer_type == 'output':
            matmul_count = 1
            
        elif layer_type == 'onnx_generated':
            # For ONNX Generated Names, Estimate Based On Tensor Count And Shapes
            matmul_count = len(matrix_tensors)  # Each Matrix Tensor Likely Represents One Matmul
            
        else:
            # For Unknown Layers, Count Actual Matrix Tensors
            matmul_count = max(1, len(matrix_tensors))
        
        layer_infos[layer_key] = LayerInfo(
            name=layer_key,
            layer_type=layer_type,
            layer_index=layer_idx,
            matmul_count=matmul_count,
            tensor_names=tensors
        )
        
        print(f"  {layer_key}: {len(tensors)} tensors, {len(matrix_tensors)} matrices, {len(weight_tensors)} weights, {matmul_count} estimated matmuls")
    
    return layer_infos


def sample_matrices_proportional(
    tensor_dict: Dict[str, np.ndarray],
    layer_infos: Dict[str, LayerInfo],
    num_samples: int,
    seed: int = 42
) -> List[Tuple[str, np.ndarray, TensorMetadata]]:
    """Sample Matrices Proportionally To Their Computational Importance"""
    
    rng = random.Random(seed)
    np.random.seed(seed)
    
    # Calculate Total
    total_matmuls = sum(layer.matmul_count for layer in layer_infos.values())
    
    if total_matmuls == 0:
        print("Warning: No matmul operations detected")
        return []
    
    print(f"\nSampling {num_samples} matrices proportional to {total_matmuls} total matmul operations...")
    
    # Allocate Samples Proportionally To Each Layer
    samples_per_layer = {}
    remaining_samples = num_samples
    
    # Sort Layers By Matmul Count (Most Important First)
    sorted_layers = sorted(layer_infos.items(), key=lambda x: x[1].matmul_count, reverse=True)
    
    for layer_key, layer_info in sorted_layers:
        if remaining_samples <= 0:
            samples_per_layer[layer_key] = 0
            continue
            
        # Calculate Proportion Of Total Computation
        proportion = layer_info.matmul_count / total_matmuls
        layer_samples = max(1, round(proportion * num_samples))  # At Least 1 Sample
        layer_samples = min(layer_samples, remaining_samples)
        
        # For ONNX Models, All Tensors Are Potentially Weights (No "Weight" Naming Convention)
        # So We Consider All 2D+ Tensors As Valid For Sampling
        available_tensors = [name for name in layer_info.tensor_names 
                           if len(tensor_dict[name].shape) >= 2]
        
        # Special Handling For Embedding And Output Layers That Might Have Meaningful Names
        if layer_info.layer_type in ['embedding', 'output']:
            weight_tensors = [name for name in available_tensors if 'weight' in name.lower()]
            if weight_tensors:
                available_tensors = weight_tensors
        
        layer_samples = min(layer_samples, len(available_tensors))
        
        samples_per_layer[layer_key] = layer_samples
        remaining_samples -= layer_samples
        
        print(f"  {layer_key}: {layer_info.matmul_count} matmuls ({proportion:.1%}) -> {layer_samples} samples (from {len(available_tensors)} tensors)")
    
    # Sample Matrices From Each Layer
    sampled_matrices = []
    
    for layer_key, num_layer_samples in samples_per_layer.items():
        if num_layer_samples == 0:
            continue
            
        layer_info = layer_infos[layer_key]
        
        # Get Available Tensors (All 2D+ Tensors For ONNX)
        available_tensors = [name for name in layer_info.tensor_names 
                           if len(tensor_dict[name].shape) >= 2]
        
        # For Embedding/Output Layers, Prefer Weight Tensors If Available
        if layer_info.layer_type in ['embedding', 'output']:
            weight_tensors = [name for name in available_tensors if 'weight' in name.lower()]
            if weight_tensors:
                available_tensors = weight_tensors
        
        if not available_tensors:
            print(f"  Warning: No suitable tensors found in {layer_key}")
            continue

        # Randomly Sample From Available Tensors
        selected_tensors = rng.sample(available_tensors, min(num_layer_samples, len(available_tensors)))
        
        for tensor_name in selected_tensors:
            tensor = tensor_dict[tensor_name]
            metadata = TensorMetadata(
                name=tensor_name,
                shape=tensor.shape,
                dtype=str(tensor.dtype),
                layer_info=layer_info
            )
            sampled_matrices.append((tensor_name, tensor, metadata))
            print(f"    Sampled: {tensor_name} {tensor.shape}")
    
    print(f"\nTotal matrices sampled: {len(sampled_matrices)}")
    return sampled_matrices


def save_matrices_for_cpp(
    sampled_matrices: List[Tuple[str, np.ndarray, TensorMetadata]],
    output_dir: str,
    format_type: str = 'binary'
) -> None:
    """Save Sampled Matrices In C++ Compatible Format"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if format_type == 'binary':
        save_binary_format(sampled_matrices, output_path)
    elif format_type == 'numpy':
        save_numpy_format(sampled_matrices, output_path)
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def save_binary_format(
    sampled_matrices: List[Tuple[str, np.ndarray, TensorMetadata]], 
    output_path: Path
) -> None:
    """Save Matrices In Binary Format With C++ Compatibility"""
    
    # Save Metadata
    with open(output_path / "metadata.txt", "w") as f:
        f.write(f"num_matrices: {len(sampled_matrices)}\n")
        f.write(f"source: onnx_model\n")
        
        for i, (name, tensor, metadata) in enumerate(sampled_matrices):
            f.write(f"\nmatrix_{i}:\n")
            f.write(f"  name: {name}\n")
            f.write(f"  shape: {' '.join(map(str, tensor.shape))}\n")
            f.write(f"  dtype: {metadata.dtype}\n")
            f.write(f"  layer_type: {metadata.layer_info.layer_type}\n")
            f.write(f"  layer_index: {metadata.layer_info.layer_index}\n")
            f.write(f"  estimated_matmuls: {metadata.layer_info.matmul_count}\n")
            f.write(f"  file: matrix_{i}.bin\n")
    
    # Save Binary Matrices
    for i, (name, tensor, metadata) in enumerate(sampled_matrices):
        # Convert To Float32 For Consistency
        tensor_f32 = tensor.astype(np.float32)
        
        with open(output_path / f"matrix_{i}.bin", "wb") as f:
            # Write Header With Magic Number        
            f.write(struct.pack('I', 0x4D545258))  # Magic: 'MTRX'
            f.write(struct.pack('I', len(tensor_f32.shape)))  # Number Of Dimensions
            
            # Write Dimensions
            for dim in tensor_f32.shape:
                f.write(struct.pack('I', dim))
            
            # Write Matrix Data In Row-Major Order (C-Style)
            f.write(tensor_f32.tobytes('C'))


def save_numpy_format(
    sampled_matrices: List[Tuple[str, np.ndarray, TensorMetadata]], 
    output_path: Path
) -> None:
    """Save Matrices In Compressed Numpy Format"""
    
    matrix_dict = {}
    metadata_dict = {}
    
    for i, (name, tensor, metadata) in enumerate(sampled_matrices):
        key = f"matrix_{i}"
        matrix_dict[key] = tensor.astype(np.float32)
        metadata_dict[key] = {
            'name': name,
            'shape': tensor.shape,
            'dtype': metadata.dtype,
            'layer_type': metadata.layer_info.layer_type,
            'layer_index': metadata.layer_info.layer_index,
            'estimated_matmuls': metadata.layer_info.matmul_count,
        }
    
    # Save Compressed Matrices
    np.savez_compressed(output_path / "matrices.npz", **matrix_dict)
    
    # Save Metadata
    with open(output_path / "metadata.txt", "w") as f:
        f.write(f"num_matrices: {len(sampled_matrices)}\n")
        f.write(f"source: onnx_model\n")
        
        for key, meta in metadata_dict.items():
            f.write(f"\n{key}:\n")
            for k, v in meta.items():
                f.write(f"  {k}: {v}\n")


def print_sampling_report(
    layer_infos: Dict[str, LayerInfo],
    sampled_matrices: List[Tuple[str, np.ndarray, TensorMetadata]]
) -> None:
    """Print Detailed Sampling Analysis Report"""
        
    print("\n" + "="*80)
    print("ONNX MODEL SAMPLING REPORT")
    print("="*80)
    
    print(f"\nLayer Analysis:")
    print(f"{'Layer':<25} {'Type':<12} {'MatMuls':<8} {'Tensors':<8} {'Weights':<8}")
    print("-" * 70)
    
    total_matmuls = sum(layer.matmul_count for layer in layer_infos.values())
    total_tensors = sum(len(layer.tensor_names) for layer in layer_infos.values())
    
    for layer_key, layer_info in sorted(layer_infos.items(), key=lambda x: (x[1].layer_type, x[1].layer_index)):
        weight_count = len([name for name in layer_info.tensor_names if 'weight' in name.lower()])
        print(f"{layer_key:<25} {layer_info.layer_type:<12} {layer_info.matmul_count:<8} {len(layer_info.tensor_names):<8} {weight_count:<8}")
    
    print(f"\nSummary:")
    print(f"  Total Layers Analysed: {len(layer_infos)}")
    print(f"  Total Tensors Found: {total_tensors}")
    print(f"  Total Estimated Matmul Ops: {total_matmuls}")
    print(f"  Total Matrices Sampled: {len(sampled_matrices)}")
    
    # Group Samples By Layer Type
    samples_by_type = defaultdict(int)
    for _, _, metadata in sampled_matrices:
        samples_by_type[metadata.layer_info.layer_type] += 1
    
    print(f"\nSamples By Layer Type:")
    for layer_type, count in sorted(samples_by_type.items()):
        percentage = (count / len(sampled_matrices)) * 100
        print(f"  {layer_type}: {count} samples ({percentage:.1f}%)")
    
    print(f"\nDetailed Sample List:")
    print(f"{'Matrix Name':<50} {'Shape':<15} {'Layer':<25} {'Type':<12}")
    print("-" * 105)
    
    for name, tensor, metadata in sampled_matrices:
        shape_str = 'x'.join(map(str, tensor.shape))
        layer_name = f"{metadata.layer_info.layer_type}_{metadata.layer_info.layer_index}"
        print(f"{name:<50} {shape_str:<15} {layer_name:<25} {metadata.layer_info.layer_type:<12}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample Representative Weight Matrices From ONNX Model Files Proportional To Computational Importance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sampler_onnx.py model.onnx --num-samples 512
  python sampler_onnx.py model.onnx -n 1024 --format numpy --verbose
  python sampler_onnx.py quantized_model.onnx -o ./samples --seed 123
        """
    )
    parser.add_argument("model_path", help="Path to the .onnx model file")
    parser.add_argument("--num-samples", "-n", type=int, default=512, 
                       help="Number of matrices to sample (default: 512)")
    parser.add_argument("--output-dir", "-o", default="./sampled_weights",
                       help="Base output directory for sampled matrices (default: ./sampled_weights)")
    parser.add_argument("--format", choices=["binary", "numpy"], default="binary",
                       help="Output format: binary (C++ friendly) or numpy (default: binary)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed sampling report")
    
    args = parser.parse_args()
    
    if not ONNX_AVAILABLE:
        print("ERROR: ONNX library not available. Install with: pip install onnx")
        return 1
    
    # Validate Input File
    model_file = Path(args.model_path)
    if not model_file.exists():
        print(f"ERROR: Model file not found: {args.model_path}")
        return 1
    
    if not model_file.suffix.lower() == '.onnx':
        print(f"WARNING: File doesn't have .onnx extension: {args.model_path}")
    
    # Extract Model Name For Output Directory
    model_name = model_file.stem
    output_dir = Path(args.output_dir) / model_name
    
    print(f"🚀 Starting ONNX Matrix Sampling")
    print(f"📁 Model: {args.model_path}")
    print(f"📊 Target samples: {args.num_samples}")
    print(f"💾 Output: {output_dir}")
    print(f"🎲 Seed: {args.seed}")
    
    try:
        # Load ONNX Model Weights
        tensor_dict = load_onnx_weights(args.model_path)
        
        if not tensor_dict:
            print("ERROR: No weight matrices found in ONNX model")
            return 1
        
        # Analyze Layers And Estimate Computational Importance
        layer_infos = estimate_matmul_operations_with_shapes(tensor_dict)
        
        if not layer_infos:
            print("ERROR: No layers identified from ONNX model")
            return 1
        
        # Sample Matrices Proportionally
        sampled_matrices = sample_matrices_proportional(
            tensor_dict, layer_infos, args.num_samples, args.seed
        )
        
        if not sampled_matrices:
            print("ERROR: No matrices were sampled")
            return 1
        
        # Save Results
        print(f"\n💾 Saving {len(sampled_matrices)} matrices to {output_dir}")
        save_matrices_for_cpp(sampled_matrices, str(output_dir), args.format)
        
        # Print Report
        if args.verbose:
            print_sampling_report(layer_infos, sampled_matrices)
        
        print(f"\n✅ Successfully Sampled {len(sampled_matrices)} Matrices From ONNX Model")
        print(f"📁 Output saved to: {output_dir.absolute()}")   
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Failed To Process ONNX Model: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
