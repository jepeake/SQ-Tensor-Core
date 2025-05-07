import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from typing import Dict, List, Tuple
import time
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from perf_model import SIMDEngine
from perf_model.preprocessing.preprocess_weights import preprocess_weights

class SuppressOutput:
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)
        return self

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def generate_random_matrix(size: int, bits: int=8, sparsity: float=0.0):

    value_range = 2 ** (bits - 1) - 1
    matrix = np.random.randint(-value_range, value_range, (size, size))
    
    if sparsity > 0:
        mask = np.random.choice([0, 1], size=(size, size), p=[sparsity, 1-sparsity])
        matrix = matrix * mask
        
    return matrix

def update_config(num_pes: int, matrix_size: int, tile_size: int) -> bool:

    config_data = {
        "num_pes": num_pes,
        "matrix_size": matrix_size,
        "tile_size": tile_size
    }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    config_file_locations = [
        os.path.join(parent_dir, "src", "core", "perf_model_config.json"),
        os.path.join(parent_dir, "models", "perf_model", "src", "core", "perf_model_config.json")
    ]
    
    for path in config_file_locations:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(config_data, f, indent=4)
            return True
        except Exception as e:
            print(f"Warning: Could not write to config path {path}: {e}")
    
    return False

def run_matrix_multiplication(matrix_size: int, tile_size: int, num_bits: int, num_pes: int) -> Dict:

    with SuppressOutput():

        if not update_config(num_pes, matrix_size, tile_size):
            pass  
        
        weights = generate_random_matrix(matrix_size, bits=num_bits, sparsity=0.5)
        activations = generate_random_matrix(matrix_size, bits=16)
        
        _ = preprocess_weights(weights, num_bits, tile_size)
        weight_file = "weight_bits.bin"
        
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"Weight file {weight_file} not found after preprocessing")
        
        engine = SIMDEngine(weight_file)

        activations_flat = activations.flatten().astype(np.int16).tolist()

    start_time = time.time()
    with SuppressOutput():
        result = engine.compute(activations_flat)
    end_time = time.time()
    
    with SuppressOutput():
        clock_frequency_hz = 1e9  # 1 GHz
        performance_metrics = engine.get_performance_metrics(clock_frequency_hz)
        
        stats = engine.get_stats()
    
    wall_time_ms = (end_time - start_time) * 1000
    
    return {
        "matrix_size": matrix_size,
        "tile_size": tile_size,
        "num_bits": num_bits,
        "num_pes": num_pes,
        "arithmetic_intensity": performance_metrics.arithmetic_intensity,
        "throughput_ops": performance_metrics.throughput_ops,
        "memory_bandwidth_bytes_per_sec": performance_metrics.memory_bandwidth_bytes_per_sec,
        "system_latency_ns": performance_metrics.system_latency_ns,
        "total_parallel_cycles": stats.total_parallel_cycles,
        "wall_time_ms": wall_time_ms
    }

def plot_roofline(results: List[Dict], use_empirical_peaks: bool = True):

    arithmetic_intensities = [r["arithmetic_intensity"] for r in results]
    achieved_performances = [r["throughput_ops"] for r in results]
    matrix_sizes = [r["matrix_size"] for r in results]
    
    peak_flops = max(achieved_performances)
    peak_bandwidth = max([r["memory_bandwidth_bytes_per_sec"] for r in results])
    ridge_point = peak_flops / peak_bandwidth
    
    scatter = plt.scatter(arithmetic_intensities, achieved_performances, s=80, alpha=0.7, 
              c=matrix_sizes, cmap='viridis')
    
    for i, result in enumerate(results):
        plt.annotate(f"{result['matrix_size']}×{result['matrix_size']}", 
                     (arithmetic_intensities[i], achieved_performances[i]),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label("Matrix Size")
    
    min_ai = min(min(arithmetic_intensities), ridge_point / 10)
    max_ai = max(max(arithmetic_intensities), ridge_point * 10)
    
    x_roof = np.logspace(np.log10(min_ai), np.log10(max_ai), 1000)
    y_roof_memory = peak_bandwidth * x_roof
    y_roof = np.minimum(np.full_like(x_roof, peak_flops), y_roof_memory)
    
    plt.plot(x_roof, y_roof, 'r-', linewidth=2, label='Roofline')
    plt.axvline(x=ridge_point, color='k', linestyle='--', alpha=0.3, label='Ridge Point')
    plt.axhline(y=peak_flops, color='r', linestyle='--', alpha=0.3, label='Peak Compute')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Arithmetic Intensity (FLOPS/Byte)')
    plt.ylabel('Performance (FLOPS)')
    plt.title('Roofline Model for SQ-TC Hardware Architecture')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    text_x = max_ai * 0.5
    text_y = peak_flops / 2
    plt.text(text_x, text_y, 
             f"Peak Compute: {peak_flops/1e12:.2f} TFLOPS\n"
             f"Peak Bandwidth: {peak_bandwidth/1e9:.2f} GB/s\n"
             f"Ridge Point: {ridge_point:.2f} FLOPS/Byte",
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Add energy efficiency plot
    plt.figure(figsize=(10, 8))
    energy_efficiency = [r["throughput_ops"] / (r["total_energy_pj"] * 1e-12) for r in results]
    
    plt.bar(range(len(matrix_sizes)), energy_efficiency)
    plt.xlabel('Matrix Size')
    plt.ylabel('Energy Efficiency (FLOPS/W)')
    plt.title('Energy Efficiency vs Matrix Size')
    plt.xticks(range(len(matrix_sizes)), [f"{size}×{size}" for size in matrix_sizes])
    plt.grid(True, alpha=0.3)
    
    for i, eff in enumerate(energy_efficiency):
        plt.text(i, eff + max(energy_efficiency)*0.02, f"{eff/1e9:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('sqtc_energy_efficiency.png', dpi=300)
    
    plt.tight_layout()
    plt.savefig('sqtc_roofline.png', dpi=300)
    plt.show()

def main():

    clock_frequency_hz = 1e9  # 1 GHz
    num_pes = 4096  
    num_bits = 4    
    tile_size = 4   
    
    parser = argparse.ArgumentParser(description='Generate a Roofline Model for SQ-TC.')
    parser.add_argument('--pes', type=int, default=num_pes, help='Number of Processing Elements')
    parser.add_argument('--bits', type=int, default=num_bits, help='Number of Bits for Weights')
    parser.add_argument('--tile', type=int, default=tile_size, help='Tile Size')
    parser.add_argument('--clock', type=float, default=clock_frequency_hz, help='Clock Frequency in Hz')
    parser.add_argument('--sizes', type=str, default="32,64,128,256,512,1024", 
                       help='Comma-separated list of Matrix Sizes to Test')
    parser.add_argument('--verbose', action='store_true', help='Enable Verbose Output')
    args = parser.parse_args()
    
    num_pes = args.pes
    num_bits = args.bits
    tile_size = args.tile
    clock_frequency_hz = args.clock
    matrix_sizes = [int(s) for s in args.sizes.split(',')]
    verbose = args.verbose
    
    print("\nSQ-TC Hardware Roofline Analysis")
    print("=" * 50)
    print(f"Hardware Configuration: {num_pes} PEs, {tile_size}×{tile_size} tiles, {num_bits}-bit weights, {clock_frequency_hz/1e9:.2f} GHz")
    
    print("\nRunning tests...")
    
    results = []
    for matrix_size in matrix_sizes:
        print(f"  Processing {matrix_size}×{matrix_size} matrix... ", end="", flush=True)
        
        try:

            result = run_matrix_multiplication(
                matrix_size=matrix_size,
                tile_size=tile_size,
                num_bits=num_bits,
                num_pes=num_pes
            )
            
            results.append(result)
            print("Done.")
            
            if verbose:
                print(f"    Arithmetic Intensity: {result['arithmetic_intensity']:.2f} FLOPS/Byte")
                print(f"    Performance: {result['throughput_ops']/1e9:.2f} GFLOPS")
                print(f"    Memory Bandwidth: {result['memory_bandwidth_bytes_per_sec']/1e9:.2f} GB/s")
                print(f"    Latency: {result['system_latency_ns']/1e3:.2f} μs")
                print(f"    Wall Clock Time: {result['wall_time_ms']:.2f} ms")
        except Exception as e:
            print(f"Error: {e}")
    
    if not results:
        print("No successful results to plot.")
        return
    
    peak_flops = max([r["throughput_ops"] for r in results])
    peak_bandwidth = max([r["memory_bandwidth_bytes_per_sec"] for r in results])
    ridge_point = peak_flops / peak_bandwidth
    
    print("\nEmpirical Peak Values:")
    print(f"Peak Compute: {peak_flops/1e12:.2f} TFLOPS")
    print(f"Peak Bandwidth: {peak_bandwidth/1e9:.2f} GB/s")
    print(f"Ridge Point: {ridge_point:.2f} FLOPS/Byte")
    
    print("\nGenerating Roofline Plot...")
    plot_roofline(results)
    print("Roofline Analysis Complete. Plot Saved to 'sqtc_roofline.png'")

if __name__ == "__main__":
    main()