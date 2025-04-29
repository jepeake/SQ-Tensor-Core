#!/usr/bin/env python3

import argparse
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from safetensors.numpy import load_file

try:
    import torch
    from safetensors.torch import load_file as torch_load_file
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def safe_load_tensor(file_path: str) -> Dict[str, np.ndarray]:
    try:
        return load_file(file_path)
    except TypeError as e:
        if "bfloat16" in str(e) and TORCH_AVAILABLE:
            print("Detected bfloat16 tensors, loading with PyTorch backend...")
            torch_dict = torch_load_file(file_path)
            return {k: v.to(torch.float32).cpu().numpy() for k, v in torch_dict.items()}
        elif "bfloat16" in str(e):
            raise ImportError(
                "Model contains bfloat16 tensors which require PyTorch. "
                "Please install PyTorch: pip install torch"
            ) from e
        else:
            raise


def sample_tensor(tensor: np.ndarray, max_elements: int, rng: random.Random = random) -> np.ndarray:
    """Sample a Tensor to a Maximum Number of Elements."""
    if tensor.size <= max_elements:
        return tensor.flatten()
    
    # For Very Large Tensors, Use Stratified Sampling to Better Capture Distribution
    if tensor.size > max_elements * 10:
        # Create Multiple Samples from Different Regions of the Tensor
        samples = []
        total_samples = 0
        
        # Split Tensor into Regions and Sample from Each
        regions = min(10, tensor.size // max_elements)
        elements_per_region = max_elements // regions
        
        # For 2D+ Tensors, Try to Sample from Different Areas
        if tensor.ndim >= 2:
            flat_indices = []
            shape = tensor.shape
            for i in range(regions):
                # Sample from Different Parts of the Tensor
                start_idx = (i * tensor.size) // regions
                end_idx = ((i + 1) * tensor.size) // regions
                
                # Convert Flat Indices to Multidimensional
                region_indices = rng.sample(range(start_idx, end_idx), 
                                           min(elements_per_region, end_idx - start_idx))
                flat_indices.extend(region_indices)
            
            # Convert Flat Indices to a Mask for Advanced Indexing
            flat_tensor = tensor.flatten()
            samples = flat_tensor[flat_indices]
            return samples
        else:
            # For 1D Tensors, Simple Random Sampling
            indices = rng.sample(range(tensor.size), max_elements)
            return tensor.flatten()[indices]
    else:
        # Simple Random Sampling for Smaller Tensors
        indices = rng.sample(range(tensor.size), max_elements)
        return tensor.flatten()[indices]


def analyze_sparsity(tensor: np.ndarray, block_size: int = 8) -> Dict[str, float]:
    """Analyze Element-Wise and Block-Wise Sparsity of a Tensor."""
    # Element-Wise Sparsity (Exact Zeros)
    zero_elements = np.count_nonzero(tensor == 0)
    element_sparsity = zero_elements / tensor.size
    
    # Element-Wise Sparsity with Small Threshold
    small_threshold = 1e-6
    small_elements = np.count_nonzero(np.abs(tensor) < small_threshold)
    small_element_sparsity = small_elements / tensor.size
    
    # For block sparsity, we need at least 2D tensor
    if tensor.ndim < 2:
        # For 1D Tensors, Reshape to 2D if Possible
        size = tensor.size
        new_side = int(np.sqrt(size))
        padded = np.zeros((new_side, new_side), dtype=tensor.dtype)
        padded.flat[:size] = tensor.flat
        tensor = padded
    
    # Ensure tensor is 2D now
    if tensor.ndim > 2:
        # Use First Two Dimensions
        original_shape = tensor.shape
        tensor = tensor.reshape(original_shape[0], -1)
    
    # Pad Tensor to be Multiple of Block Size
    h, w = tensor.shape
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(tensor, ((0, pad_h), (0, pad_w)), mode='constant')
        tensor = padded
    
    # Get New Dimensions After Padding
    h, w = tensor.shape
    
    # Calculate Block-Wise Sparsity
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size
    total_blocks = n_blocks_h * n_blocks_w
    zero_blocks = 0
    near_zero_blocks = 0
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = tensor[i:i+block_size, j:j+block_size]
            if np.all(block == 0):
                zero_blocks += 1
            if np.all(np.abs(block) < small_threshold):
                near_zero_blocks += 1
    
    block_sparsity = zero_blocks / total_blocks if total_blocks > 0 else 0
    near_zero_block_sparsity = near_zero_blocks / total_blocks if total_blocks > 0 else 0
    
    return {
        "element_sparsity": element_sparsity,
        "small_element_sparsity": small_element_sparsity,
        "block_sparsity": block_sparsity,
        "near_zero_block_sparsity": near_zero_block_sparsity
    }


def get_weight_histogram(samples: np.ndarray, n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Histogram of Weight Values."""
    # For Highly Concentrated Distributions, Use Log Bins
    if np.std(samples) < 0.01 * np.abs(np.mean(samples)):
        # Use Log-Spaced Bins for Concentrated Distributions
        if np.min(samples) <= 0:
            # Handle Zeros and Negative Values by Shifting
            min_val = np.min(samples)
            if min_val < 0:
                # For Negative Values, Use Symmetric Log Bins
                max_abs = max(abs(np.min(samples)), abs(np.max(samples)))
                edges = np.concatenate([
                    -np.logspace(np.log10(max_abs), np.log10(1e-10), n_bins//2)[::-1],
                    [0],
                    np.logspace(np.log10(1e-10), np.log10(max_abs), n_bins//2)
                ])
            else:
                # For Zeros, Use a Special First Bin and Log-Spaced Remainder
                min_nonzero = np.min(samples[samples > 0]) if np.any(samples > 0) else 1e-10
                max_val = np.max(samples)
                edges = np.concatenate([
                    [0],
                    np.logspace(np.log10(min_nonzero), np.log10(max_val), n_bins-1)
                ])
        else:
            # All Positive Values, Use Standard Log Bins
            min_val = np.min(samples)
            max_val = np.max(samples)
            if min_val == max_val:
                edges = np.linspace(min_val - 0.1, max_val + 0.1, n_bins)
            else:
                edges = np.logspace(np.log10(max(min_val, 1e-10)), np.log10(max(max_val, 1e-9)), n_bins)
    else:
        # Use Linear Bins for Well-Spread Distributions
        min_val = np.min(samples)
        max_val = np.max(samples)
        if min_val == max_val:
            edges = np.linspace(min_val - 0.1, max_val + 0.1, n_bins)
        else:
            edges = np.linspace(min_val, max_val, n_bins)
    
    hist, bin_edges = np.histogram(samples, bins=edges)
    return hist, bin_edges


def categorize_tensor(name: str) -> Tuple[str, str, str]:
    """Categorize Tensor by Layer Type, Position, and Parameter Type."""
    layer_type = "other"
    position = "other"
    param_type = "other"
    
    name_lower = name.lower()
    
    # Determine Layer Type
    if "embed" in name_lower:
        layer_type = "embedding"
    elif "attention" in name_lower:
        layer_type = "attention"
        if "self" in name_lower:
            layer_type = "self_attention"
        elif "output" in name_lower:
            layer_type = "attention_output"
    elif "ffn" in name_lower or "mlp" in name_lower or "feed_forward" in name_lower:
        layer_type = "ffn"
    elif "norm" in name_lower or "layernorm" in name_lower:
        layer_type = "normalization"
    elif "proj" in name_lower:
        layer_type = "projection"
    elif "decoder" in name_lower:
        layer_type = "decoder"
    elif "encoder" in name_lower:
        layer_type = "encoder"
    elif "head" in name_lower or "classifier" in name_lower or "lm_head" in name_lower:
        layer_type = "head"
    
    # Determine Position
    if any(x in name_lower for x in ["0.", ".0.", "layer.0", "block.0", "0/"]):
        position = "early"
    elif any(x in name_lower for x in ["1.", ".1.", "layer.1", "block.1", "1/"]):
        position = "early"
    elif any(x in name_lower for x in ["2.", ".2.", "layer.2", "block.2", "2/"]):
        position = "early"
    elif any(x in name_lower.split(".")[-2:] for x in ["last", "final"]):
        position = "late"
    elif re.search(r'layer.([2-9][0-9]|[1-9][0-9][0-9])', name_lower):
        position = "late"
    elif "backbone" in name_lower:
        position = "middle"
    elif "embed" in name_lower:
        position = "early"
    elif "head" in name_lower:
        position = "late"
    else:
        position = "middle"
    
    # Determine Parameter Type
    if "weight" in name_lower:
        param_type = "weight"
    elif "bias" in name_lower:
        param_type = "bias"
    elif "embedding" in name_lower:
        param_type = "embedding"
    elif "wq" in name_lower or "w_q" in name_lower or "query" in name_lower:
        param_type = "query"
    elif "wk" in name_lower or "w_k" in name_lower or "key" in name_lower:
        param_type = "key"
    elif "wv" in name_lower or "w_v" in name_lower or "value" in name_lower:
        param_type = "value"
    
    return layer_type, position, param_type


def plot_weight_distributions(
    all_samples: np.ndarray,
    samples_by_layer_type: Dict[str, np.ndarray],
    output_dir: str,
    n_bins: int = 100,
    include_zero_centered: bool = True,
    include_log_scale: bool = True
):
    """Create Detailed Weight Distribution Plots for the Model and by Layer Type.
    
    Args:
        all_samples: Array of All Sampled Weights Across the Model
        samples_by_layer_type: Dictionary Mapping Layer Type to Arrays of Weight Samples
        output_dir: Directory to Save Plots
        n_bins: Number of Bins for Histograms
        include_zero_centered: Whether to Include Zoomed Plots Centered Around Zero
        include_log_scale: Whether to Include Plots with Log Scale Y-Axis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Whole Model Weight Distribution (Normal Scale)
    plt.figure(figsize=(12, 8))
    hist, bin_edges = get_weight_histogram(all_samples, n_bins)
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', alpha=0.7)
    plt.title("Global Weight Distribution", fontsize=14)
    plt.xlabel("Weight Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_weight_distribution.png"), dpi=300)
    plt.close()
    
    # Plot Whole Model Weight Distribution (Log Scale)
    if include_log_scale:
        plt.figure(figsize=(12, 8))
        hist, bin_edges = get_weight_histogram(all_samples, n_bins)
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', alpha=0.7)
        plt.title("Global Weight Distribution (Log Scale)", fontsize=14)
        plt.xlabel("Weight Value", fontsize=12)
        plt.ylabel("Frequency (Log Scale)", fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "global_weight_distribution_log.png"), dpi=300)
        plt.close()
    
    # Plot Whole Model Weight Distribution (Zero-Centered)
    if include_zero_centered:
        plt.figure(figsize=(12, 8))
        # Focus on the center region (±3 std around mean)
        mean, std = np.mean(all_samples), np.std(all_samples)
        center_samples = all_samples[np.abs(all_samples - mean) < 3 * std]
        hist, bin_edges = np.histogram(center_samples, bins=n_bins)
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', alpha=0.7)
        plt.title("Weight Distribution (Central Region ±3σ)", fontsize=14)
        plt.xlabel("Weight Value", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "global_weight_distribution_center.png"), dpi=300)
        plt.close()
    
    # Plot Distributions by Layer Type (One Plot per Layer Type)
    for layer_type, samples in samples_by_layer_type.items():
        if len(samples) > 100:  # Only plot if we have enough samples
            # Normal scale
            plt.figure(figsize=(12, 8))
            hist, bin_edges = get_weight_histogram(samples, n_bins)
            plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', alpha=0.7)
            plt.title(f"{layer_type} Weight Distribution", fontsize=14)
            plt.xlabel("Weight Value", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{layer_type}_weight_distribution.png"), dpi=300)
            plt.close()
            
            # Log Scale 
            if include_log_scale:
                plt.figure(figsize=(12, 8))
                hist, bin_edges = get_weight_histogram(samples, n_bins)
                plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', alpha=0.7)
                plt.title(f"{layer_type} Weight Distribution (Log Scale)", fontsize=14)
                plt.xlabel("Weight Value", fontsize=12)
                plt.ylabel("Frequency (Log Scale)", fontsize=12)
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{layer_type}_weight_distribution_log.png"), dpi=300)
                plt.close()
            
            # Zero-Centered
            if include_zero_centered and len(samples) > 1000:
                plt.figure(figsize=(12, 8))
                mean, std = np.mean(samples), np.std(samples)
                center_samples = samples[np.abs(samples - mean) < 3 * std]
                if len(center_samples) > 100:
                    hist, bin_edges = np.histogram(center_samples, bins=n_bins)
                    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', alpha=0.7)
                    plt.title(f"{layer_type} Weight Distribution (Central Region ±3σ)", fontsize=14)
                    plt.xlabel("Weight Value", fontsize=12)
                    plt.ylabel("Frequency", fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{layer_type}_weight_distribution_center.png"), dpi=300)
                    plt.close()
    
    # Create a Comparison Plot with KDE for All Layer Types
    plt.figure(figsize=(14, 10))
    
    # Use Kernel Density Estimation for Smoother Comparison
    from scipy.stats import gaussian_kde
    
    # Select Layer Types with Sufficient Samples (Skip "Other" if We Have Enough Categories)
    selected_types = []
    for layer_type, samples in samples_by_layer_type.items():
        if len(samples) > 1000 and (layer_type != "other" or len(selected_types) < 2):
            selected_types.append(layer_type)
    
    # If We Have Too Many Layer Types, Select the Most Important Ones
    if len(selected_types) > 6:
        important_types = ["attention", "self_attention", "ffn", "normalization", "embedding", "head"]
        selected_types = [t for t in selected_types if t in important_types][:6]
    
    # Create KDE Plots
    for layer_type in selected_types:
        samples = samples_by_layer_type[layer_type]
        
        # For Computational Efficiency, Subsample if We Have Too Many Points
        if len(samples) > 10000:
            indices = np.random.choice(len(samples), 10000, replace=False)
            kde_samples = samples[indices]
        else:
            kde_samples = samples
        
        # Focus on Central Region (±5σ from Global Mean)
        global_mean, global_std = np.mean(all_samples), np.std(all_samples)
        kde_samples = kde_samples[np.abs(kde_samples - global_mean) < 5 * global_std]
        
        if len(kde_samples) > 100:
            # Create KDE
            kde = gaussian_kde(kde_samples)
            
            # Plot KDE on a Consistent X Range
            x = np.linspace(global_mean - 3 * global_std, global_mean + 3 * global_std, 1000)
            plt.plot(x, kde(x), label=layer_type, linewidth=2)
    
    plt.title("Weight Distribution Comparison by Layer Type", fontsize=16)
    plt.xlabel("Weight Value", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_type_weight_comparison.png"), dpi=300)
    plt.close()


def analyze_model(
    model_path: str,
    max_tensors: int = 100,
    max_elements_per_tensor: int = 100000,
    block_size: int = 8,
    seed: int = 42,
    output_dir: Optional[str] = None,
):
    """Analyze Weight Distributions and Sparsity Patterns Across a Model."""
    rng = random.Random(seed)
    
    # Load the Model
    print(f"Loading model from {model_path}...")
    tensor_dict = safe_load_tensor(model_path)
    
    # Get All Tensor Names
    tensor_names = list(tensor_dict.keys())
    print(f"Model contains {len(tensor_names)} tensors")
    
    # Sample Tensors if There Are Too Many
    if len(tensor_names) > max_tensors:
        sampled_names = rng.sample(tensor_names, max_tensors)
    else:
        sampled_names = tensor_names
    
    # Create Data Structures for Analysis
    tensor_shapes = {}
    tensor_dtypes = {}
    tensor_stats = {}
    
    # Group Tensors by Category
    tensor_by_layer_type = defaultdict(list)
    tensor_by_position = defaultdict(list)
    tensor_by_param_type = defaultdict(list)
    
    # Create Histograms for All Weights Combined and by Category
    all_samples = []
    samples_by_layer_type = defaultdict(list)
    samples_by_position = defaultdict(list)
    samples_by_param_type = defaultdict(list)
    
    # Sparsity Metrics
    sparsity_by_layer_type = defaultdict(list)
    sparsity_by_position = defaultdict(list)
    sparsity_by_param_type = defaultdict(list)
    
    # Process Each Tensor
    print(f"Analyzing {len(sampled_names)} tensors...")
    for name in tqdm(sampled_names):
        tensor = tensor_dict[name]
        
        # Record Shape and Dtype
        tensor_shapes[name] = tensor.shape
        tensor_dtypes[name] = str(tensor.dtype)
        
        # Categorize Tensor
        layer_type, position, param_type = categorize_tensor(name)
        tensor_by_layer_type[layer_type].append(name)
        tensor_by_position[position].append(name)
        tensor_by_param_type[param_type].append(name)
        
        # Sample Tensor Values
        tensor_samples = sample_tensor(tensor, max_elements_per_tensor, rng)
        
        # Add to Collections
        all_samples.extend(tensor_samples.tolist())
        samples_by_layer_type[layer_type].extend(tensor_samples.tolist())
        samples_by_position[position].extend(tensor_samples.tolist())
        samples_by_param_type[param_type].extend(tensor_samples.tolist())
        
        # Analyze Sparsity
        sparsity_metrics = analyze_sparsity(tensor, block_size)
        tensor_stats[name] = {
            "shape": tensor.shape,
            "size": tensor.size,
            "min": float(np.min(tensor_samples)),
            "max": float(np.max(tensor_samples)),
            "mean": float(np.mean(tensor_samples)),
            "std": float(np.std(tensor_samples)),
            "median": float(np.median(tensor_samples)),
            "layer_type": layer_type,
            "position": position,
            "param_type": param_type,
            **sparsity_metrics
        }
        
        # Add sparsity metrics to collections
        sparsity_by_layer_type[layer_type].append(sparsity_metrics)
        sparsity_by_position[position].append(sparsity_metrics)
        sparsity_by_param_type[param_type].append(sparsity_metrics)
    
    # Convert Samples to Numpy Arrays for Histograms
    all_samples = np.array(all_samples)
    samples_by_layer_type = {k: np.array(v) for k, v in samples_by_layer_type.items()}
    samples_by_position = {k: np.array(v) for k, v in samples_by_position.items()}
    samples_by_param_type = {k: np.array(v) for k, v in samples_by_param_type.items()}
    
    # Calculate Global Statistics
    global_stats = {
        "total_tensors": len(tensor_names),
        "analyzed_tensors": len(sampled_names),
        "total_parameters": sum(tensor_dict[name].size for name in tensor_names),
        "min_weight": float(np.min(all_samples)) if len(all_samples) > 0 else 0,
        "max_weight": float(np.max(all_samples)) if len(all_samples) > 0 else 0,
        "mean_weight": float(np.mean(all_samples)) if len(all_samples) > 0 else 0,
        "std_weight": float(np.std(all_samples)) if len(all_samples) > 0 else 0,
        "median_weight": float(np.median(all_samples)) if len(all_samples) > 0 else 0,
    }
    
    # Calculate Average Sparsity Metrics by Category
    avg_sparsity_by_layer_type = {}
    for layer_type, metrics_list in sparsity_by_layer_type.items():
        avg_sparsity_by_layer_type[layer_type] = {
            "count": len(metrics_list),
            "element_sparsity": np.mean([m["element_sparsity"] for m in metrics_list]),
            "small_element_sparsity": np.mean([m["small_element_sparsity"] for m in metrics_list]),
            "block_sparsity": np.mean([m["block_sparsity"] for m in metrics_list]),
            "near_zero_block_sparsity": np.mean([m["near_zero_block_sparsity"] for m in metrics_list]),
        }
    
    avg_sparsity_by_position = {}
    for position, metrics_list in sparsity_by_position.items():
        avg_sparsity_by_position[position] = {
            "count": len(metrics_list),
            "element_sparsity": np.mean([m["element_sparsity"] for m in metrics_list]),
            "small_element_sparsity": np.mean([m["small_element_sparsity"] for m in metrics_list]),
            "block_sparsity": np.mean([m["block_sparsity"] for m in metrics_list]),
            "near_zero_block_sparsity": np.mean([m["near_zero_block_sparsity"] for m in metrics_list]),
        }
    
    avg_sparsity_by_param_type = {}
    for param_type, metrics_list in sparsity_by_param_type.items():
        avg_sparsity_by_param_type[param_type] = {
            "count": len(metrics_list),
            "element_sparsity": np.mean([m["element_sparsity"] for m in metrics_list]),
            "small_element_sparsity": np.mean([m["small_element_sparsity"] for m in metrics_list]),
            "block_sparsity": np.mean([m["block_sparsity"] for m in metrics_list]),
            "near_zero_block_sparsity": np.mean([m["near_zero_block_sparsity"] for m in metrics_list]),
        }
    
    # Print Summary
    print("\n=== Model Weight Distribution Analysis ===")
    print(f"Model: {model_path}")
    print(f"Total tensors: {global_stats['total_tensors']}")
    print(f"Total parameters: {global_stats['total_parameters']:,}")
    print(f"Weight range: [{global_stats['min_weight']:.6f}, {global_stats['max_weight']:.6f}]")
    print(f"Weight mean: {global_stats['mean_weight']:.6f}")
    print(f"Weight std: {global_stats['std_weight']:.6f}")
    
    print("\n=== Sparsity Analysis ===")
    print("By Layer Type:")
    for layer_type, stats in avg_sparsity_by_layer_type.items():
        print(f"  {layer_type} (count: {stats['count']})")
        print(f"    Element sparsity: {stats['element_sparsity']*100:.2f}%")
        print(f"    Small element sparsity (|x|<1e-6): {stats['small_element_sparsity']*100:.2f}%")
        print(f"    {block_size}x{block_size} Block sparsity: {stats['block_sparsity']*100:.2f}%")
        print(f"    Near-zero block sparsity: {stats['near_zero_block_sparsity']*100:.2f}%")
    
    # Generate Plots if Output Directory is Provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate Detailed Weight Distribution Plots
        plot_weight_distributions(all_samples, samples_by_layer_type, output_dir)
        
        # Plot Sparsity by Category
        categories = [
            ("Layer Type", avg_sparsity_by_layer_type),
            ("Position", avg_sparsity_by_position),
            ("Parameter Type", avg_sparsity_by_param_type)
        ]
        
        for title, data in categories:
            plt.figure(figsize=(12, 6))
            
            # Element Sparsity
            names = list(data.keys())
            values = [data[n]["element_sparsity"] * 100 for n in names]
            small_values = [data[n]["small_element_sparsity"] * 100 for n in names]
            
            x = np.arange(len(names))
            width = 0.35
            
            plt.bar(x - width/2, values, width, label='Exact Zero')
            plt.bar(x + width/2, small_values, width, label='Near Zero (|x|<1e-6)')
            
            plt.xlabel(title)
            plt.ylabel('Sparsity %')
            plt.title(f'Element Sparsity by {title}')
            plt.xticks(x, names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"element_sparsity_by_{title.lower().replace(' ', '_')}.png"))
            plt.close()
            
            # Block sparsity
            plt.figure(figsize=(12, 6))
            
            block_values = [data[n]["block_sparsity"] * 100 for n in names]
            near_zero_block_values = [data[n]["near_zero_block_sparsity"] * 100 for n in names]
            
            plt.bar(x - width/2, block_values, width, label=f'Exact Zero {block_size}x{block_size} Blocks')
            plt.bar(x + width/2, near_zero_block_values, width, label=f'Near Zero {block_size}x{block_size} Blocks')
            
            plt.xlabel(title)
            plt.ylabel('Sparsity %')
            plt.title(f'Block Sparsity by {title}')
            plt.xticks(x, names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"block_sparsity_by_{title.lower().replace(' ', '_')}.png"))
            plt.close()
    
    # Return Analysis Results
    return {
        "global_stats": global_stats,
        "tensor_stats": tensor_stats,
        "sparsity_by_layer_type": avg_sparsity_by_layer_type,
        "sparsity_by_position": avg_sparsity_by_position,
        "sparsity_by_param_type": avg_sparsity_by_param_type,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Weight Distributions and Sparsity Across Model"
    )
    parser.add_argument("model_path", type=str, help="Path to .safetensors file")
    parser.add_argument(
        "--max-tensors", type=int, default=100, 
        help="Maximum number of tensors to sample for analysis"
    )
    parser.add_argument(
        "--max-elements", type=int, default=100000,
        help="Maximum number of elements to sample per tensor"
    )
    parser.add_argument(
        "--block-size", type=int, default=8,
        help="Block size for block sparsity analysis"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save output plots (if not provided, no plots are generated)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling"
    )
    
    args = parser.parse_args()
    
    try:
        analysis = analyze_model(
            args.model_path,
            max_tensors=args.max_tensors,
            max_elements_per_tensor=args.max_elements,
            block_size=args.block_size,
            output_dir=args.output_dir,
            seed=args.seed
        )
        print("Analysis complete!")
        if args.output_dir:
            print(f"Plots saved to {args.output_dir}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 