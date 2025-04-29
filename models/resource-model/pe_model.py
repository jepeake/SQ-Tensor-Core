import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
import matplotlib.pyplot as plt

@dataclass
class HardwareCosts:
    # Energy costs in pJ from Table 3.1
    ENERGY = {
        'add_8bit': 0.03,
        'add_32bit': 0.1,
        'mul_8bit': 0.2,
        'mul_32bit': 3.1,
        'transmission_gate': 0.0012,
        'shift_8bit': 0.01, 
        'compare_4bit': 0.005,
        'mux_8bit': 0.01,
        
        'bit_mul': 0.01,      
        'bit_add': 0.005,     
        'bit_acc': 0.015,     
        'bit_mux': 0.002      
    }
    
    # Area costs in μm² from Table 3.2
    AREA = {
        'add_8bit': 36,
        'add_32bit': 137,
        'mul_8bit': 282,
        'mul_32bit': 3495,
        'transmission_gate': 1.4, 
        'shift_8bit': 15,  
        'compare_4bit': 8,
        'mux_8bit': 12,

        'bit_mul': 5,        
        'bit_add': 8,        
        'bit_acc': 25,        
        'bit_mux': 3         
    }

def calculate_bit_serial_mac_costs(weight_bits: int = 4, activation_bits: int = 8) -> Dict:
    # - Single Bit Multiplier
    # - Shift Operation
    # - Adder for Accumulation
    # - Processes Serial over Weight Bits
    
    components = {
        'bit_multiplier': {
            'count': 1,  # Single Bit Multiplier
            'energy': HardwareCosts.ENERGY['bit_mul'],
            'area': HardwareCosts.AREA['bit_mul']
        },
        'shifter': {
            'count': 1,
            'energy': HardwareCosts.ENERGY['shift_8bit'],
            'area': HardwareCosts.AREA['shift_8bit']
        },
        'accumulator': {
            'count': 1,
            'energy': HardwareCosts.ENERGY['bit_acc'],
            'area': HardwareCosts.AREA['bit_acc']
        }
    }
    
    # Account for Weight_Bits Cycles of Operation
    # Energy Scales with Cycles, Area Remains the Same
    cycle_factor = weight_bits
    
    total_area = sum(comp['area'] for comp in components.values())
    total_energy = sum(comp['energy'] for comp in components.values()) * cycle_factor
    
    return {
        'config': f"W{weight_bits}A{activation_bits}",
        'components': components,
        'total_energy': total_energy,
        'total_area': total_area
    }

def calculate_bit_slice_mac_costs(weight_bits: int = 4, activation_bits: int = 8) -> Dict:
    # - Separate Multipliers for High/Low Bits
    # - Shift Operation
    # - Adder for Accumulation
    # - Operates in 2 Cycles in This Case (High/Low Slices)
    
    num_slices = 2  # High/Low Bit Slices
    
    components = {
        'bit_multiplier': {
            'count': 1,  # One Multiplier Reused for Each Slice
            'energy': HardwareCosts.ENERGY['bit_mul'] * activation_bits,  # Operates on Activation_Bits/2 Bits per Slice
            'area': HardwareCosts.AREA['bit_mul'] * activation_bits
        },
        'shifter': {
            'count': 1,
            'energy': HardwareCosts.ENERGY['shift_8bit'],
            'area': HardwareCosts.AREA['shift_8bit']
        },
        'accumulator': {
            'count': 1,
            'energy': HardwareCosts.ENERGY['bit_acc'],
            'area': HardwareCosts.AREA['bit_acc']
        }
    }
    
    # Account for the 2 Cycles of Operation (High/Low Slices)
    # Energy Scales with Cycles, Area Remains the Same
    cycle_factor = num_slices
    
    total_area = sum(comp['area'] for comp in components.values())
    total_energy = sum(comp['energy'] for comp in components.values()) * cycle_factor
    
    return {
        'config': f"W{weight_bits}A{activation_bits}",
        'components': components,
        'total_energy': total_energy,
        'total_area': total_area
    }

def calculate_bit_interleaved_mac_costs(weight_bits: int = 4, activation_bits: int = 8) -> Dict:
    # - Multiple Bit-Multipliers with Shared Weights
    # - Adder Tree
    # - Multiplexer for Bit Selection
    
    components = {
        'bit_multipliers': {
            'count': 8,  
            'energy': 8 * HardwareCosts.ENERGY['bit_mul'],  # Bit-Level Multipliers
            'area': 8 * HardwareCosts.AREA['bit_mul']
        },
        'adder_tree': {
            'count': 1,
            'energy': HardwareCosts.ENERGY['add_8bit'],  # Still Need a Decent Adder for All Results
            'area': HardwareCosts.AREA['add_8bit']
        },
        'mux': {
            'count': 1,
            'energy': HardwareCosts.ENERGY['bit_mux'] * weight_bits * 2,  # MUX for Weight Bits and Activations
            'area': HardwareCosts.AREA['bit_mux'] * weight_bits * 2
        }
    }
    
    # No Need for Cycle Factor as This Operates in Parallel
    
    total_energy = sum(comp['energy'] for comp in components.values())
    total_area = sum(comp['area'] for comp in components.values())
    
    return {
        'config': f"W{weight_bits}A{activation_bits}",
        'components': components,
        'total_energy': total_energy,
        'total_area': total_area
    }

def calculate_bit_column_mac_costs(weight_bits: int = 4, activation_bits: int = 8) -> Dict:
    # - Parallel Bit Multipliers for Each Weight Bit
    # - Simple Adder Tree
    # - Shift for Accumulation
    
    components = {
        'bit_multipliers': {
            'count': weight_bits,  # One per Weight Bit Column
            'energy': weight_bits * HardwareCosts.ENERGY['bit_mul'],
            'area': weight_bits * HardwareCosts.AREA['bit_mul']
        },
        'adders': {
            'count': weight_bits - 1 + 1,  # Tree + Accumulator
            'energy': weight_bits * HardwareCosts.ENERGY['bit_add'],
            'area': weight_bits * HardwareCosts.AREA['bit_add']
        },
        'shifter': {
            'count': 1,
            'energy': HardwareCosts.ENERGY['shift_8bit'],
            'area': HardwareCosts.AREA['shift_8bit']
        }
    }
    
    # Account for Cycles to Process All Activation Bits
    cycle_factor = activation_bits
    
    total_area = sum(comp['area'] for comp in components.values())
    total_energy = sum(comp['energy'] for comp in components.values()) * cycle_factor
    
    return {
        'config': f"W{weight_bits}A{activation_bits}",
        'components': components,
        'total_energy': total_energy,
        'total_area': total_area
    }

def calculate_pe_costs(tile_dim: int = 4, weight_bits: int = 4, activation_bits: int = 8) -> Dict:
    elements_per_tile = tile_dim * tile_dim
    base_adders = tile_dim * weight_bits - 1
    
    sparsity = 0.80

    active_ratio = 1 - sparsity
    compressed_adders = int(base_adders * active_ratio)  # Reduce Adders Based on Sparsity
    
    sq_tc_components = {
        'transmission_gates': {
            'count': elements_per_tile * weight_bits,
            'energy': elements_per_tile * weight_bits * HardwareCosts.ENERGY['transmission_gate'] * active_ratio,
            'area': elements_per_tile * weight_bits * HardwareCosts.AREA['transmission_gate']
        },
        'adder_tree': {
            'count': compressed_adders,
            'energy': compressed_adders * HardwareCosts.ENERGY['add_8bit'],
            'area': compressed_adders * HardwareCosts.AREA['add_8bit']
        }
    }
    
    mac_components = {
        'multipliers': {
            'count': elements_per_tile,
            'energy': elements_per_tile * HardwareCosts.ENERGY['mul_8bit'],
            'area': elements_per_tile * HardwareCosts.AREA['mul_8bit']
        },
        'adder_tree': {
            'count': (elements_per_tile - 1),
            'energy': (elements_per_tile - 1) * HardwareCosts.ENERGY['add_8bit'],
            'area': (elements_per_tile - 1) * HardwareCosts.AREA['add_8bit']
        }
    }
    
    bit_serial_costs = calculate_bit_serial_mac_costs(weight_bits, activation_bits)
    bit_slice_costs = calculate_bit_slice_mac_costs(weight_bits, activation_bits)
    bit_interleaved_costs = calculate_bit_interleaved_mac_costs(weight_bits, activation_bits)
    bit_column_costs = calculate_bit_column_mac_costs(weight_bits, activation_bits)
    
    sq_tc_total_energy = sum(comp['energy'] for comp in sq_tc_components.values())
    sq_tc_total_area = sum(comp['area'] for comp in sq_tc_components.values())
    
    mac_total_energy = sum(comp['energy'] for comp in mac_components.values())
    mac_total_area = sum(comp['area'] for comp in mac_components.values())
    
    # For Bit-Serial Architectures, We Need One PE per Tile Element
    # But Each PE is Much Simpler than a Full MAC
    return {
        'tile_config': f"{tile_dim}x{tile_dim} W{weight_bits}A{activation_bits}",
        'sq_tc': {
            'components': sq_tc_components,
            'total_energy': sq_tc_total_energy,
            'total_area': sq_tc_total_area
        },
        'mac': {
            'components': mac_components,
            'total_energy': mac_total_energy,
            'total_area': mac_total_area
        },
        'bit_serial': {
            'components': bit_serial_costs['components'],
            'total_energy': bit_serial_costs['total_energy'] * elements_per_tile,
            'total_area': bit_serial_costs['total_area'] * elements_per_tile
        },
        'bit_slice': {
            'components': bit_slice_costs['components'],
            'total_energy': bit_slice_costs['total_energy'] * elements_per_tile,
            'total_area': bit_slice_costs['total_area'] * elements_per_tile
        },
        'bit_interleaved': {
            'components': bit_interleaved_costs['components'],
            'total_energy': bit_interleaved_costs['total_energy'],  # Don't Scale by Elements_Per_Tile as It's More Shared
            'total_area': bit_interleaved_costs['total_area']  # More Shared Hardware, Don't Scale by Full Tile Elements
        },
        'bit_column': {
            'components': bit_column_costs['components'],
            'total_energy': bit_column_costs['total_energy'],  # Already Accounts for Column-Parallel Operation
            'total_area': bit_column_costs['total_area']  # Already Sized for the Bit-Level Parallelism
        },
        'comparison': {
            'sq_tc_mac_energy_ratio': sq_tc_total_energy / mac_total_energy,
            'sq_tc_mac_area_ratio': sq_tc_total_area / mac_total_area,
            'bit_serial_mac_energy_ratio': (bit_serial_costs['total_energy'] * elements_per_tile) / mac_total_energy,
            'bit_serial_mac_area_ratio': (bit_serial_costs['total_area'] * elements_per_tile) / mac_total_area,
            'bit_slice_mac_energy_ratio': (bit_slice_costs['total_energy'] * elements_per_tile) / mac_total_energy,
            'bit_slice_mac_area_ratio': (bit_slice_costs['total_area'] * elements_per_tile) / mac_total_area,
            'bit_interleaved_mac_energy_ratio': bit_interleaved_costs['total_energy'] / mac_total_energy,
            'bit_interleaved_mac_area_ratio': bit_interleaved_costs['total_area'] / mac_total_area,
            'bit_column_mac_energy_ratio': bit_column_costs['total_energy'] / mac_total_energy,
            'bit_column_mac_area_ratio': bit_column_costs['total_area'] / mac_total_area
        }
    }

def analyse_matrix_scaling(matrix_sizes: list, tile_dim: int = 4):
    results = []
    for size in matrix_sizes:
        num_tiles = ((size + tile_dim - 1) // tile_dim) ** 2  
        pe_costs = calculate_pe_costs(tile_dim=tile_dim)
        
        results.append({
            'matrix_size': size,
            'num_tiles': num_tiles,
            'total_sq_tc_energy': pe_costs['sq_tc']['total_energy'] * num_tiles,
            'total_mac_energy': pe_costs['mac']['total_energy'] * num_tiles,
            'total_bit_serial_energy': pe_costs['bit_serial']['total_energy'] * num_tiles,
            'total_bit_slice_energy': pe_costs['bit_slice']['total_energy'] * num_tiles,
            'total_bit_interleaved_energy': pe_costs['bit_interleaved']['total_energy'] * num_tiles,
            'total_bit_column_energy': pe_costs['bit_column']['total_energy'] * num_tiles,
            'total_sq_tc_area': pe_costs['sq_tc']['total_area'] * num_tiles,
            'total_mac_area': pe_costs['mac']['total_area'] * num_tiles,
            'total_bit_serial_area': pe_costs['bit_serial']['total_area'] * num_tiles,
            'total_bit_slice_area': pe_costs['bit_slice']['total_area'] * num_tiles,
            'total_bit_interleaved_area': pe_costs['bit_interleaved']['total_area'] * num_tiles,
            'total_bit_column_area': pe_costs['bit_column']['total_area'] * num_tiles
        })
    
    return results

def plot_scaling_analysis(matrix_sizes: list, results: list):
    plt.style.use('seaborn-v0_8-paper')
    
    def create_plot(fig_name, use_log_scale=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        
        colors = {
            'sq_tc': '#808080',          
            'mac': '#8B7CC8',            
            'bit_serial': '#6F9EE3',     
            'bit_slice': '#B784D9',      
            'bit_interleaved': '#E091C8', 
            'bit_column': '#4B6BCC'      
        }
        
        line_styles = {
            'sq_tc': '-',
            'mac': '--',
            'bit_serial': '-.',
            'bit_slice': ':',
            'bit_interleaved': (0, (3, 1, 1, 1)),  
            'bit_column': (0, (5, 1))             
        }
        
        markers = {
            'sq_tc': 'o',
            'mac': 's',
            'bit_serial': '^',
            'bit_slice': 'D',
            'bit_interleaved': 'v',
            'bit_column': 'p'
        }
        
        # Energy Plot
        ax1.plot(matrix_sizes, [r['total_sq_tc_energy'] for r in results], 
                 color=colors['sq_tc'], linestyle=line_styles['sq_tc'], marker=markers['sq_tc'], 
                 markersize=5, label='SQ-TC')
        ax1.plot(matrix_sizes, [r['total_mac_energy'] for r in results], 
                 color=colors['mac'], linestyle=line_styles['mac'], marker=markers['mac'], 
                 markersize=5, label='MAC')
        ax1.plot(matrix_sizes, [r['total_bit_serial_energy'] for r in results], 
                 color=colors['bit_serial'], linestyle=line_styles['bit_serial'], marker=markers['bit_serial'], 
                 markersize=5, label='Bit-Serial MAC')
        ax1.plot(matrix_sizes, [r['total_bit_slice_energy'] for r in results], 
                 color=colors['bit_slice'], linestyle=line_styles['bit_slice'], marker=markers['bit_slice'], 
                 markersize=5, label='Bit-Slice MAC')
        ax1.plot(matrix_sizes, [r['total_bit_interleaved_energy'] for r in results], 
                 color=colors['bit_interleaved'], linestyle=line_styles['bit_interleaved'], marker=markers['bit_interleaved'], 
                 markersize=5, label='Bit-Interleaved MAC')
        ax1.plot(matrix_sizes, [r['total_bit_column_energy'] for r in results], 
                 color=colors['bit_column'], linestyle=line_styles['bit_column'], marker=markers['bit_column'], 
                 markersize=5, label='Bit-Column MAC')
        
        # Format Energy Plot
        ax1.set_title('Total Energy vs Matrix Size', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Matrix Dimension', fontsize=12)
        ax1.set_ylabel('Total Energy (pJ)', fontsize=12)
        if use_log_scale:
            ax1.set_xscale('log', base=2)
            ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # Area Plot
        ax2.plot(matrix_sizes, [r['total_sq_tc_area'] for r in results], 
                 color=colors['sq_tc'], linestyle=line_styles['sq_tc'], marker=markers['sq_tc'], 
                 markersize=5, label='SQ-TC')
        ax2.plot(matrix_sizes, [r['total_mac_area'] for r in results], 
                 color=colors['mac'], linestyle=line_styles['mac'], marker=markers['mac'], 
                 markersize=5, label='MAC')
        ax2.plot(matrix_sizes, [r['total_bit_serial_area'] for r in results], 
                 color=colors['bit_serial'], linestyle=line_styles['bit_serial'], marker=markers['bit_serial'], 
                 markersize=5, label='Bit-Serial MAC')
        ax2.plot(matrix_sizes, [r['total_bit_slice_area'] for r in results], 
                 color=colors['bit_slice'], linestyle=line_styles['bit_slice'], marker=markers['bit_slice'], 
                 markersize=5, label='Bit-Slice MAC')
        ax2.plot(matrix_sizes, [r['total_bit_interleaved_area'] for r in results], 
                 color=colors['bit_interleaved'], linestyle=line_styles['bit_interleaved'], marker=markers['bit_interleaved'], 
                 markersize=5, label='Bit-Interleaved MAC')
        ax2.plot(matrix_sizes, [r['total_bit_column_area'] for r in results], 
                 color=colors['bit_column'], linestyle=line_styles['bit_column'], marker=markers['bit_column'], 
                 markersize=5, label='Bit-Column MAC')
        
        # Format Area Plot
        ax2.set_title('Total Area vs Matrix Size', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Matrix Dimension', fontsize=12)
        ax2.set_ylabel('Total Area (μm²)', fontsize=12)
        if use_log_scale:
            ax2.set_xscale('log', base=2)
            ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        # Create a Single Legend for the Figure
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                  ncol=3, fontsize=10, frameon=True, title="Architecture Types")
        
        # Adjust Layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the Figure with High Quality
        plt.savefig(f'{fig_name}.png', format='png', dpi=300, bbox_inches='tight')
    
    # Create and save Logarithmic Scale Plot
    create_plot('mac_architecture_comparison_log', use_log_scale=True)
    
    # Create and save Linear Scale Plot
    create_plot('mac_architecture_comparison_linear', use_log_scale=False)
    
    # Show Only the Last Plot (The Linear One)
    plt.show()

def analyse_tile_scaling(tile_sizes: list, weight_bits: int = 4, activation_bits: int = 8):
    """Analyze how SQ-TC energy and area scale with different tile sizes"""
    results = []
    for tile_dim in tile_sizes:
        pe_costs = calculate_pe_costs(tile_dim=tile_dim, weight_bits=weight_bits, activation_bits=activation_bits)
        
        results.append({
            'tile_dim': tile_dim,
            'sq_tc_energy': pe_costs['sq_tc']['total_energy'],
            'sq_tc_area': pe_costs['sq_tc']['total_area'],
            'mac_energy': pe_costs['mac']['total_energy'],
            'mac_area': pe_costs['mac']['total_area']
        })
    
    return results

def analyse_weight_bits_scaling(weight_bits_list: list, tile_dim: int = 4, activation_bits: int = 8):
    """Analyze how SQ-TC energy and area scale with different weight bit-widths"""
    results = []
    for w_bits in weight_bits_list:
        pe_costs = calculate_pe_costs(tile_dim=tile_dim, weight_bits=w_bits, activation_bits=activation_bits)
        
        results.append({
            'weight_bits': w_bits,
            'sq_tc_energy': pe_costs['sq_tc']['total_energy'],
            'sq_tc_area': pe_costs['sq_tc']['total_area'],
            'mac_energy': pe_costs['mac']['total_energy'],
            'mac_area': pe_costs['mac']['total_area']
        })
    
    return results

def analyse_combined_scaling(tile_sizes: list, weight_bits_list: list, activation_bits: int = 8):
    """Analyze SQ-TC energy and area for combinations of tile size and weight bits"""
    combined_results = {}
    
    for tile_dim in tile_sizes:
        combined_results[tile_dim] = {}
        for w_bits in weight_bits_list:
            pe_costs = calculate_pe_costs(tile_dim=tile_dim, weight_bits=w_bits, activation_bits=activation_bits)
            combined_results[tile_dim][w_bits] = {
                'sq_tc_energy': pe_costs['sq_tc']['total_energy'],
                'sq_tc_area': pe_costs['sq_tc']['total_area'],
                'mac_energy': pe_costs['mac']['total_energy'],
                'mac_area': pe_costs['mac']['total_area']
            }
    
    return combined_results

def plot_tile_size_scaling(tile_sizes: list, results: list):
    """Plot SQ-TC energy and area versus tile size"""
    plt.style.use('seaborn-v0_8-paper')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Energy Plot
    ax1.plot(tile_sizes, [r['sq_tc_energy'] for r in results], 
             color='#808080', marker='o', markersize=5, label='SQ-TC')
    
    # Format Energy Plot
    ax1.set_title('SQ-TC Energy vs Tile Size', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Tile Dimension (NxN)', fontsize=12)
    ax1.set_ylabel('Energy (pJ)', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Area Plot
    ax2.plot(tile_sizes, [r['sq_tc_area'] for r in results], 
             color='#808080', marker='o', markersize=5, label='SQ-TC')
    
    # Format Area Plot
    ax2.set_title('SQ-TC Area vs Tile Size', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Tile Dimension (NxN)', fontsize=12)
    ax2.set_ylabel('Area (μm²)', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust Layout
    plt.tight_layout()
    
    # Save the Figure with High Quality
    plt.savefig('sq_tc_vs_tile_size.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_weight_bits_scaling(weight_bits_list: list, results: list):
    """Plot SQ-TC energy and area versus weight bits"""
    plt.style.use('seaborn-v0_8-paper')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Energy Plot
    ax1.plot(weight_bits_list, [r['sq_tc_energy'] for r in results], 
             color='#808080', marker='o', markersize=5, label='SQ-TC')
    
    # Format Energy Plot
    ax1.set_title('SQ-TC Energy vs Weight Bits', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Weight Bit Width', fontsize=12)
    ax1.set_ylabel('Energy (pJ)', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Area Plot
    ax2.plot(weight_bits_list, [r['sq_tc_area'] for r in results], 
             color='#808080', marker='o', markersize=5, label='SQ-TC')
    
    # Format Area Plot
    ax2.set_title('SQ-TC Area vs Weight Bits', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Weight Bit Width', fontsize=12)
    ax2.set_ylabel('Area (μm²)', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust Layout
    plt.tight_layout()
    
    # Save the Figure with High Quality
    plt.savefig('sq_tc_vs_weight_bits.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_combined_heatmap(tile_sizes: list, weight_bits_list: list, combined_results: dict):
    """Plot heatmaps of SQ-TC energy and area for combinations of tile size and weight bits"""
    plt.style.use('seaborn-v0_8-paper')
    
    # Create energy and area matrices for the heatmap
    energy_matrix = np.zeros((len(tile_sizes), len(weight_bits_list)))
    area_matrix = np.zeros((len(tile_sizes), len(weight_bits_list)))
    
    # Fill the matrices
    for i, tile_dim in enumerate(tile_sizes):
        for j, w_bits in enumerate(weight_bits_list):
            energy_matrix[i, j] = combined_results[tile_dim][w_bits]['sq_tc_energy']
            area_matrix[i, j] = combined_results[tile_dim][w_bits]['sq_tc_area']
    
    # Create a figure with two subplots for energy and area
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Plot energy heatmap
    im1 = ax1.imshow(energy_matrix, cmap='Blues', aspect='auto')
    ax1.set_title('SQ-TC Energy (pJ)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Weight Bit Width', fontsize=12)
    ax1.set_ylabel('Tile Dimension (NxN)', fontsize=12)
    ax1.set_xticks(range(len(weight_bits_list)))
    ax1.set_yticks(range(len(tile_sizes)))
    ax1.set_xticklabels(weight_bits_list)
    ax1.set_yticklabels(tile_sizes)
    
    # Add colorbar
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label('Energy (pJ)', rotation=270, labelpad=15)
    
    # Plot area heatmap
    im2 = ax2.imshow(area_matrix, cmap='Purples', aspect='auto')
    ax2.set_title('SQ-TC Area (μm²)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Weight Bit Width', fontsize=12)
    ax2.set_ylabel('Tile Dimension (NxN)', fontsize=12)
    ax2.set_xticks(range(len(weight_bits_list)))
    ax2.set_yticks(range(len(tile_sizes)))
    ax2.set_xticklabels(weight_bits_list)
    ax2.set_yticklabels(tile_sizes)
    
    # Add colorbar
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label('Area (μm²)', rotation=270, labelpad=15)
    
    # Add value annotations
    for i in range(len(tile_sizes)):
        for j in range(len(weight_bits_list)):
            ax1.text(j, i, f"{energy_matrix[i, j]:.1f}", ha="center", va="center", color="black" if energy_matrix[i, j] < np.max(energy_matrix) / 1.5 else "white")
            ax2.text(j, i, f"{area_matrix[i, j]:.1f}", ha="center", va="center", color="black" if area_matrix[i, j] < np.max(area_matrix) / 1.5 else "white")
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('sq_tc_combined_heatmap.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Analysis for a Single PE:")
    results = calculate_pe_costs(tile_dim=4, weight_bits=4, activation_bits=8)

    print("\nAnalysing Scaling with Matrix Size...")
    matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    scaling_results = analyse_matrix_scaling(matrix_sizes)
    
    plot_scaling_analysis(matrix_sizes, scaling_results)
    
    print("\nAnalysing SQ-TC vs Tile Size...")
    tile_sizes = [2, 4, 8, 16, 32]
    tile_scaling_results = analyse_tile_scaling(tile_sizes)
    plot_tile_size_scaling(tile_sizes, tile_scaling_results)
    
    print("\nAnalysing SQ-TC vs Weight Bits...")
    weight_bits_list = [1, 2, 4, 8, 16]
    weight_bits_results = analyse_weight_bits_scaling(weight_bits_list)
    plot_weight_bits_scaling(weight_bits_list, weight_bits_results)
    
    print("\nAnalysing Combined Scaling (Tile Size vs Weight Bits)...")
    combined_results = analyse_combined_scaling(tile_sizes, weight_bits_list)
    plot_combined_heatmap(tile_sizes, weight_bits_list, combined_results)