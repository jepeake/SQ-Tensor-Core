#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse
import time
import json
import matplotlib.pyplot as plt


#  ███████████                   █████                                                                                                            
# ░█░░░███░░░█                  ░░███                                                                                                             
# ░   ░███  ░   ██████   █████  ███████                                                                                                           
#     ░███     ███░░███ ███░░  ░░░███░                                                                                                            
#     ░███    ░███████ ░░█████   ░███                                                                                                             
#     ░███    ░███░░░   ░░░░███  ░███ ███                                                                                                         
#     █████   ░░██████  ██████   ░░█████                                                                                                          
#    ░░░░░     ░░░░░░  ░░░░░░     ░░░░░         
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
#   █████████                                 █████                                     ███                     █████     ███                     
#  ███░░░░░███                               ░░███                                     ░░░                     ░░███     ░░░                      
# ░███    ░░░  █████ ████ ████████    ██████  ░███████   ████████   ██████  ████████   ████   █████   ██████   ███████   ████   ██████  ████████  
# ░░█████████ ░░███ ░███ ░░███░░███  ███░░███ ░███░░███ ░░███░░███ ███░░███░░███░░███ ░░███  ███░░   ░░░░░███ ░░░███░   ░░███  ███░░███░░███░░███ 
#  ░░░░░░░░███ ░███ ░███  ░███ ░███ ░███ ░░░  ░███ ░███  ░███ ░░░ ░███ ░███ ░███ ░███  ░███ ░░█████   ███████   ░███     ░███ ░███ ░███ ░███ ░███ 
#  ███    ░███ ░███ ░███  ░███ ░███ ░███  ███ ░███ ░███  ░███     ░███ ░███ ░███ ░███  ░███  ░░░░███ ███░░███   ░███ ███ ░███ ░███ ░███ ░███ ░███ 
# ░░█████████  ░░███████  ████ █████░░██████  ████ █████ █████    ░░██████  ████ █████ █████ ██████ ░░████████  ░░█████  █████░░██████  ████ █████
#  ░░░░░░░░░    ░░░░░███ ░░░░ ░░░░░  ░░░░░░  ░░░░ ░░░░░ ░░░░░      ░░░░░░  ░░░░ ░░░░░ ░░░░░ ░░░░░░   ░░░░░░░░    ░░░░░  ░░░░░  ░░░░░░  ░░░░ ░░░░░ 
#               ███ ░███                                                                                                                          
#              ░░██████                                                                                                                           
#               ░░░░░░       
#                                                                                                                      
#  ██████   ██████              █████                                                                                                             
# ░░██████ ██████              ░░███                                                                                                              
#  ░███░█████░███   ██████   ███████   ██████   █████                                                                                             
#  ░███░░███ ░███  ███░░███ ███░░███  ███░░███ ███░░                                                                                              
#  ░███ ░░░  ░███ ░███ ░███░███ ░███ ░███████ ░░█████                                                                                             
#  ░███      ░███ ░███ ░███░███ ░███ ░███░░░   ░░░░███                                                                                            
#  █████     █████░░██████ ░░████████░░██████  ██████                                                                                             
# ░░░░░     ░░░░░  ░░░░░░   ░░░░░░░░  ░░░░░░  ░░░░░░                                                                                              


# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import perf_model
from preprocessing.preprocess_weights import preprocess_weights


def test_synchronisation_modes(matrix_size=32, tile_size=4, num_bits=8, sparsity_levels=None, verbose=True):
    """
    Test different synchronisation modes and compare performance
    """
    if sparsity_levels is None:
        sparsity_levels = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    # Names of synchronisation modes for plotting and reporting
    sync_mode_names = {
        0: "Global Stalling",
        1: "Global Barrier per GEMM",
        2: "Global Barrier per Batch",
        3: "Async Local FIFOs",
        4: "Async Shared Buffer"
    }
    
    # Store results
    results = {}
    
    # Configure batch size for mode 2
    batch_size = 8
    
    # Test each synchronisation mode with different sparsity levels
    for sync_mode in range(5):  # 0-4 for different modes
        print(f"\n========== Testing Synchronisation Mode {sync_mode}: {sync_mode_names[sync_mode]} ==========")
        
        results[sync_mode] = {
            'name': sync_mode_names[sync_mode],
            'sparsity': [],
            'latency': [],
            'throughput': [],
            'utilization': [],
            'cycle_count': []
        }
        
        for sparsity in sparsity_levels:
            print(f"\nWeight Sparsity = {sparsity:.2f}")
            
            # Create a temporary weight file with the desired sparsity
            weight_file = generate_test_weights(matrix_size, tile_size, num_bits, sparsity)
            
            # Update the config file
            update_config_file(sync_mode=sync_mode, batch_size=batch_size, 
                               weight_sparsity=sparsity, activation_sparsity=sparsity)
            
            # Create test activations with the same sparsity
            activations = generate_test_activations(matrix_size, sparsity)
            
            # Run the test
            start_time = time.time()
            engine = perf_model.SIMDEngine(weight_file)
            result_tile = engine.compute(activations.flatten().tolist(), 0)
            end_time = time.time()
            
            # Get performance metrics
            metrics = engine.get_performance_metrics(1.0e9)  # Assume 1 GHz clock
            
            # Calculate PE utilization (actual ops / theoretical max ops)
            actual_ops = metrics.ops_per_cycle
            max_ops = engine.get_num_pes() * 2  # Assuming 2 ops per PE per cycle (MAC)
            utilization = actual_ops / max_ops if max_ops > 0 else 0
            
            # Store results
            results[sync_mode]['sparsity'].append(sparsity)
            results[sync_mode]['latency'].append(metrics.system_latency_ns)
            results[sync_mode]['throughput'].append(metrics.throughput_ops)
            results[sync_mode]['utilization'].append(utilization)
            results[sync_mode]['cycle_count'].append(engine.get_total_cycles())
            
            # Print result summary
            print(f"Execution Time: {end_time - start_time:.4f} seconds")
            print(f"Total Cycles: {engine.get_total_cycles()}")
            print(f"System Latency: {metrics.system_latency_ns:.2f} ns")
            print(f"Throughput: {metrics.throughput_ops / 1e9:.2f} GOPS")
            print(f"PE Utilisation: {utilization * 100:.2f}%")
            
            # Print mode-specific stats
            if sync_mode == 0:  # Global Stalling
                print(f"Total Stall Cycles: {engine.get_stats().global_stalls}")
            elif sync_mode == 1:  # Global Barrier per GEMM
                print(f"Global Barriers: {engine.get_global_barriers()}")
            elif sync_mode == 2:  # Global Barrier per Batch
                print(f"Batch Size: {batch_size}")
                print(f"Batch Barriers: {engine.get_stats().batch_barriers}")
            elif sync_mode == 3:  # Async Local FIFOs
                print(f"FIFO Depth: {engine.get_stats().fifo_depth}")
                total_fifo_waits = sum(pe.fifo_wait_cycles for pe in engine.get_stats().pe_stats)
                print(f"Total FIFO Wait Cycles: {total_fifo_waits}")
            elif sync_mode == 4:  # Async Shared Buffer
                print(f"Output Buffer Size: {engine.get_stats().output_buffer_size}")
            
            # Maximum skew between PEs
            print(f"Maximum Skew Between PEs: {engine.get_max_skew_cycles()} cycles")
            
            # Clean up temporary files
            if os.path.exists(weight_file):
                os.remove(weight_file)
    
    # Plot results
    plot_results(results, sparsity_levels)
    
    return results


def generate_test_weights(matrix_size, tile_size, num_bits, sparsity):
    """
    Generate test weights with specified sparsity and preprocess into a temporary file
    """

    # Create a temporary output path
    temp_output_dir = os.path.dirname(__file__)
    temp_weight_file = os.path.join(temp_output_dir, f"temp_weights_{sparsity:.2f}.bin")
    
    # Save current directory to restore later
    original_dir = os.getcwd()
    
    try:
        # Change to the temp directory
        os.chdir(temp_output_dir)
        
        # Generate random weights
        np.random.seed(42)  # For Reproducibility
        weights = np.random.randint(-128, 127, size=(matrix_size, matrix_size)).astype(np.int8)
        
        # Apply Sparsity
        if sparsity > 0:
            sparsity_mask = np.random.choice(
                [0, 1], 
                size=weights.shape, 
                p=[sparsity, 1-sparsity]
            )
            weights = weights * sparsity_mask
        
        # Preprocess Weights Using the Preprocessing Module
        # Note: this function writes to a file named "weight_bits.bin" in the current directory
        preprocess_weights(weights, num_bits=num_bits, tile_size=tile_size)
        
        # Rename the output file to our desired name if it exists
        if os.path.exists("weight_bits.bin"):
            if os.path.exists(temp_weight_file):
                os.remove(temp_weight_file)
            os.rename("weight_bits.bin", os.path.basename(temp_weight_file))
        
        return temp_weight_file
    
    finally:
        # Restore original directory
        os.chdir(original_dir)


def generate_test_activations(matrix_size, sparsity):
    """
    Generate test activations with specified sparsity
    """
    np.random.seed(43)  # Different Seed from Weights
    activations = np.random.randint(-128, 127, size=(matrix_size, matrix_size)).astype(np.int16)
    
    # Apply Sparsity
    if sparsity > 0:
        sparsity_mask = np.random.choice(
            [0, 1], 
            size=activations.shape, 
            p=[sparsity, 1-sparsity]
        )
        activations = activations * sparsity_mask
    
    return activations


def update_config_file(sync_mode, batch_size, weight_sparsity, activation_sparsity):
    """
    Update the configuration file with the specified parameters
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "src", "core", "perf_model_config.json")
    
    # Read Existing Config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Update Parameters
    config["synchronisation_mode"] = sync_mode
    config["batch_size"] = batch_size
    config["expected_weight_sparsity"] = weight_sparsity
    config["expected_activation_sparsity"] = activation_sparsity
    
    # Write Updated Config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

def plot_results(results, sparsity_levels):
    """
    Plot the Performance Results Across Different Synchronisation Modes
    """
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Latency vs Sparsity
    plt.subplot(2, 2, 1)
    for mode in results:
        plt.plot(results[mode]['sparsity'], results[mode]['latency'], 
                 marker='o', label=results[mode]['name'])
    plt.xlabel('Sparsity')
    plt.ylabel('Latency (ns)')
    plt.title('Latency vs Sparsity')
    plt.grid(True)
    plt.legend()
    
    # Subplot 2: Throughput vs Sparsity
    plt.subplot(2, 2, 2)
    for mode in results:
        plt.plot(results[mode]['sparsity'], [t/1e9 for t in results[mode]['throughput']], 
                 marker='o', label=results[mode]['name'])
    plt.xlabel('Sparsity')
    plt.ylabel('Throughput (GOPS)')
    plt.title('Throughput vs Sparsity')
    plt.grid(True)
    plt.legend()
    
    # Subplot 3: PE Utilisation vs Sparsity
    plt.subplot(2, 2, 3)
    for mode in results:
        plt.plot(results[mode]['sparsity'], [u*100 for u in results[mode]['utilization']], 
                 marker='o', label=results[mode]['name'])
    plt.xlabel('Sparsity')
    plt.ylabel('PE Utilization (%)')
    plt.title('PE Utilization vs Sparsity')
    plt.grid(True)
    plt.legend()
    
    # Subplot 4: Cycle Count vs Sparsity
    plt.subplot(2, 2, 4)
    for mode in results:
        plt.plot(results[mode]['sparsity'], results[mode]['cycle_count'], 
                 marker='o', label=results[mode]['name'])
    plt.xlabel('Sparsity')
    plt.ylabel('Total Cycles')
    plt.title('Cycle Count vs Sparsity')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the Plot
    plot_path = os.path.join(os.path.dirname(__file__), "synchronisation_comparison.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Also Create a Bar Chart for a Single Sparsity Level (0.5)
    plt.figure(figsize=(12, 8))
    sparsity_idx = sparsity_levels.index(0.5) if 0.5 in sparsity_levels else 0
    
    # Prepare Data
    mode_names = [results[mode]['name'] for mode in results]
    latencies = [results[mode]['latency'][sparsity_idx] for mode in results]
    throughputs = [results[mode]['throughput'][sparsity_idx]/1e9 for mode in results]
    utilizations = [results[mode]['utilization'][sparsity_idx]*1e2 for mode in results]
    
    # Bar Positions
    x = np.arange(len(mode_names))
    width = 0.25
    
    # Plot
    plt.bar(x - width, latencies, width, label='Latency (ns)')
    plt.bar(x, throughputs, width, label='Throughput (GOPS)')
    plt.bar(x + width, utilizations, width, label='Utilization (%)')
    
    plt.xlabel('Synchronisation Mode')
    plt.ylabel('Value')
    plt.title(f'Performance Comparison at Sparsity={sparsity_levels[sparsity_idx]}')
    plt.xticks(x, mode_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the Bar Chart
    bar_path = os.path.join(os.path.dirname(__file__), "synchronisation_bars.png")
    plt.savefig(bar_path)
    print(f"Bar Chart Saved to {bar_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Different Synchronisation Modes")
    parser.add_argument("--matrix_size", type=int, default=32, help="Matrix Size")
    parser.add_argument("--tile_size", type=int, default=4, help="Tile Size")
    parser.add_argument("--num_bits", type=int, default=8, help="Number of Bits")
    parser.add_argument("--sparsity", type=float, nargs='+', help="List of Sparsity Levels to Test")
    parser.add_argument("--verbose", action="store_true", help="Print More Details")
    
    args = parser.parse_args()
    
    sparsity_levels = args.sparsity if args.sparsity else [0.0, 0.3, 0.5, 0.7, 0.9]
    
    test_synchronisation_modes(
        matrix_size=args.matrix_size,
        tile_size=args.tile_size,
        num_bits=args.num_bits,
        sparsity_levels=sparsity_levels,
        verbose=args.verbose
    ) 