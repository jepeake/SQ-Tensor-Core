import argparse
import re
import struct
import random
from typing import Dict, List, Tuple, NamedTuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from safetensors.numpy import load_file
from tqdm import tqdm

try:
    import torch
    from safetensors.torch import load_file as torch_load_file
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


#   █████████                                      ████                    
#  ███░░░░░███                                    ░░███                    
# ░███    ░░░   ██████   █████████████   ████████  ░███   ██████  ████████ 
# ░░█████████  ░░░░░███ ░░███░░███░░███ ░░███░░███ ░███  ███░░███░░███░░███
#  ░░░░░░░░███  ███████  ░███ ░███ ░███  ░███ ░███ ░███ ░███████  ░███ ░░░ 
#  ███    ░███ ███░░███  ░███ ░███ ░███  ░███ ░███ ░███ ░███░░░   ░███     
# ░░█████████ ░░████████ █████░███ █████ ░███████  █████░░██████  █████    
#  ░░░░░░░░░   ░░░░░░░░ ░░░░░ ░░░ ░░░░░  ░███░░░  ░░░░░  ░░░░░░  ░░░░░     
#                                        ░███                              
#                                        █████                             
#                                       ░░░░░                
            
# Script to Sample Representative Weight Matrices from Tensor Files


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


def load_safetensors(file_path: str) -> Dict[str, np.ndarray]:
    try:
        return load_file(file_path)
    except TypeError as e:
        if "bfloat16" in str(e) and TORCH_AVAILABLE:
            print("Detected bfloat16 tensors, loading with PyTorch backend...")
            torch_dict = torch_load_file(file_path)
            numpy_dict = {}
            for k, v in torch_dict.items():
                if v.dtype == torch.bfloat16:
                    numpy_dict[k] = v.to(torch.float32).cpu().numpy()
                else:
                    numpy_dict[k] = v.cpu().numpy()
            return numpy_dict
        else:
            raise


def parse_tensor_name(name: str) -> Tuple[str, int, str]:
    patterns = [
        (r'layers?\.(\d+)\.(?:self_)?(?:attn|attention)\.(.+)', 'attention'),
        (r'layers?\.(\d+)\.(?:mlp|feed_forward)\.(.+)', 'mlp'),
        (r'(?:embed_tokens|token_embeddings|embeddings)\.(.+)', 'embedding'),
        (r'(?:lm_head|output|head)\.(.+)', 'output'),
        (r'layers?\.(\d+)\..*(?:norm|ln)\.(.+)', 'norm'),
        (r'layers?\.(\d+)\.(.+)', 'other'),
    ]
    
    for pattern, layer_type in patterns:
        match = re.search(pattern, name)
        if match:
            if layer_type in ['embedding', 'output']:
                return layer_type, 0, match.group(1)
            else:
                layer_idx = int(match.group(1))
                component = match.group(2) if len(match.groups()) > 1 else ''
                return layer_type, layer_idx, component
    
    return 'unknown', 0, name


def estimate_matmul_operations(tensor_dict: Dict[str, np.ndarray]) -> Dict[str, LayerInfo]:
    layers = defaultdict(lambda: {'tensors': [], 'type': 'unknown', 'matmuls': 0})
    
    for name, tensor in tensor_dict.items():
        if len(tensor.shape) < 2:  
            continue
            
        layer_type, layer_idx, component = parse_tensor_name(name)
        layer_key = f"{layer_type}_{layer_idx}"
        
        layers[layer_key]['tensors'].append(name)
        layers[layer_key]['type'] = layer_type
        layers[layer_key]['index'] = layer_idx
    
    layer_infos = {}
    
    for layer_key, layer_data in layers.items():
        layer_type = layer_data['type']
        layer_idx = layer_data['index']
        tensors = layer_data['tensors']
        
        matmul_count = 0
        
        if layer_type == 'attention':
            weight_tensors = [name for name in tensors if 'weight' in name.lower()]
            matmul_count = len(weight_tensors)  
            
        elif layer_type == 'mlp':
            weight_tensors = [name for name in tensors if 'weight' in name.lower()]
            matmul_count = len(weight_tensors)  
            
        elif layer_type == 'embedding':     
            matmul_count = 1
            
        elif layer_type == 'output':
            matmul_count = 1
            
        else:
            weight_tensors = [name for name in tensors if 'weight' in name.lower()]
            matmul_count = len(weight_tensors)
        
        layer_infos[layer_key] = LayerInfo(
            name=layer_key,
            layer_type=layer_type,
            layer_index=layer_idx,
            matmul_count=max(1, matmul_count),  
            tensor_names=tensors
        )
    
    return layer_infos


def sample_matrices_proportional(
    tensor_dict: Dict[str, np.ndarray],
    layer_infos: Dict[str, LayerInfo],
    num_samples: int,
    seed: int = 42
) -> List[Tuple[str, np.ndarray, TensorMetadata]]:
    
    rng = random.Random(seed)
    
    total_matmuls = sum(layer.matmul_count for layer in layer_infos.values())
    
    if total_matmuls == 0:
        print("Warning: No matmul operations detected")
        return []
    
    samples_per_layer = {}
    remaining_samples = num_samples
    
    sorted_layers = sorted(layer_infos.items(), key=lambda x: x[1].matmul_count, reverse=True)
    
    for layer_key, layer_info in sorted_layers:
        if remaining_samples <= 0:
            samples_per_layer[layer_key] = 0
            continue
            
        proportion = layer_info.matmul_count / total_matmuls
        layer_samples = max(1, round(proportion * num_samples))  
        layer_samples = min(layer_samples, remaining_samples)
        layer_samples = min(layer_samples, len(layer_info.tensor_names))  
        
        samples_per_layer[layer_key] = layer_samples
        remaining_samples -= layer_samples
    
    sampled_matrices = []
    
    for layer_key, num_layer_samples in samples_per_layer.items():
        if num_layer_samples == 0:
            continue
            
        layer_info = layer_infos[layer_key]
        
        weight_tensors = [name for name in layer_info.tensor_names 
                         if 'weight' in name.lower() and len(tensor_dict[name].shape) >= 2]
        
        if not weight_tensors:
            continue

        selected_tensors = rng.sample(weight_tensors, min(num_layer_samples, len(weight_tensors)))
        
        for tensor_name in selected_tensors:
            tensor = tensor_dict[tensor_name]
            metadata = TensorMetadata(
                name=tensor_name,
                shape=tensor.shape,
                dtype=str(tensor.dtype),
                layer_info=layer_info
            )
            sampled_matrices.append((tensor_name, tensor, metadata))
    
    return sampled_matrices


def save_matrices_for_cpp(
    sampled_matrices: List[Tuple[str, np.ndarray, TensorMetadata]],
    output_dir: str,
    format_type: str = 'binary'
) -> None:
    
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
    
    with open(output_path / "metadata.txt", "w") as f:
        f.write(f"num_matrices: {len(sampled_matrices)}\n")
        for i, (name, tensor, metadata) in enumerate(sampled_matrices):
            f.write(f"\nmatrix_{i}:\n")
            f.write(f"  name: {name}\n")
            f.write(f"  shape: {' '.join(map(str, tensor.shape))}\n")
            f.write(f"  dtype: {metadata.dtype}\n")
            f.write(f"  layer_type: {metadata.layer_info.layer_type}\n")
            f.write(f"  layer_index: {metadata.layer_info.layer_index}\n")
            f.write(f"  file: matrix_{i}.bin\n")
    
    for i, (name, tensor, metadata) in enumerate(sampled_matrices):
        tensor_f32 = tensor.astype(np.float32)
        
        with open(output_path / f"matrix_{i}.bin", "wb") as f:
            f.write(struct.pack('I', 0x4D545258))  
            f.write(struct.pack('I', len(tensor_f32.shape)))        
            for dim in tensor_f32.shape:
                f.write(struct.pack('I', dim))  
            
            tensor_f32.tobytes('C')  
            f.write(tensor_f32.tobytes('C'))


def save_numpy_format(
    sampled_matrices: List[Tuple[str, np.ndarray, TensorMetadata]], 
    output_path: Path
) -> None:
    
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
        }
    
    np.savez_compressed(output_path / "matrices.npz", **matrix_dict)
    
    with open(output_path / "metadata.txt", "w") as f:
        f.write(f"num_matrices: {len(sampled_matrices)}\n")
        for key, meta in metadata_dict.items():
            f.write(f"\n{key}:\n")
            for k, v in meta.items():
                f.write(f"  {k}: {v}\n")


def print_sampling_report(
    layer_infos: Dict[str, LayerInfo],
    sampled_matrices: List[Tuple[str, np.ndarray, TensorMetadata]]
) -> None:
        
    print("\nLayer Analysis:")
    print(f"{'Layer':<20} {'Type':<12} {'MatMuls':<8} {'Tensors':<8}")
    print("-" * 50)
    
    total_matmuls = sum(layer.matmul_count for layer in layer_infos.values())
    
    for layer_key, layer_info in sorted(layer_infos.items(), key=lambda x: x[1].layer_index):
        print(f"{layer_key:<20} {layer_info.layer_type:<12} {layer_info.matmul_count:<8} {len(layer_info.tensor_names):<8}")
    
    print(f"\nTotal estimated matmul operations: {total_matmuls}")
    print(f"Total matrices sampled: {len(sampled_matrices)}")
    
    print("\nSampled Matrices:")
    print(f"{'Matrix':<50} {'Shape':<15} {'Layer':<20} {'Type':<12}")
    print("-" * 100)
    
    for name, tensor, metadata in sampled_matrices:
        shape_str = 'x'.join(map(str, tensor.shape))
        layer_name = f"{metadata.layer_info.layer_type}_{metadata.layer_info.layer_index}"
        print(f"{name:<50} {shape_str:<15} {layer_name:<20} {metadata.layer_info.layer_type:<12}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample representative weight matrices from a safetensors file proportional to matmul operations"
    )
    parser.add_argument("model_path", help="Path to the .safetensors file")
    parser.add_argument("--num-samples", "-n", type=int, default=1024, 
                       help="Number of matrices to sample (default: 32)")
    parser.add_argument("--output-dir", "-o", default="./sampled_weights",
                       help="Base output directory for sampled matrices (default: ./sampled_weights)")
    parser.add_argument("--format", choices=["binary", "numpy"], default="binary",
                       help="Output format: binary (C++ friendly) or numpy (default: binary)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed sampling report")
    
    args = parser.parse_args()
    
    model_file = Path(args.model_path)
    model_name = model_file.stem  
    
    output_dir = Path(args.output_dir) / model_name
    
    tensor_dict = load_safetensors(args.model_path)
    print(f"Loaded {len(tensor_dict)} tensors")
    
    layer_infos = estimate_matmul_operations(tensor_dict)
    
    print(f"Sampling {args.num_samples} Matrices")
    sampled_matrices = sample_matrices_proportional(
        tensor_dict, layer_infos, args.num_samples, args.seed
    )
    
    if not sampled_matrices:
        print("No Matrices Sampled.")
        return
    
    print(f"Saving {len(sampled_matrices)} Matrices to {output_dir}")
    save_matrices_for_cpp(sampled_matrices, str(output_dir), args.format)
    
    if args.verbose:
        print_sampling_report(layer_infos, sampled_matrices)

if __name__ == "__main__":
    main()
