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
        'transmission_gate': 0.001,
        'shift_8bit': 0.01, 
        'compare_4bit': 0.005  
    }
    
    # Area costs in μm² from Table 3.2
    AREA = {
        'add_8bit': 36,
        'add_32bit': 137,
        'mul_8bit': 282,
        'mul_32bit': 3495,
        'transmission_gate': 1, 
        'shift_8bit': 15,  
        'compare_4bit': 8  
    }

def calculate_mixpe_costs(weight_bits: int = 4, activation_bits: int = 8) -> Dict:
    # - 4 Shift Operations 
    # - 4 Adders
    # - Comparison Logic for Each Weight Bit
    components = {
        'shifts': {
            'count': weight_bits,
            'energy': weight_bits * HardwareCosts.ENERGY['shift_8bit'],
            'area': weight_bits * HardwareCosts.AREA['shift_8bit']
        },
        'adder_tree': {
            'count': weight_bits,  
            'energy': (weight_bits) * HardwareCosts.ENERGY['add_8bit'],
            'area': (weight_bits) * HardwareCosts.AREA['add_8bit']
        },
        'bit_comparators': {
            'count': weight_bits,
            'energy': weight_bits * HardwareCosts.ENERGY['compare_4bit'],
            'area': weight_bits * HardwareCosts.AREA['compare_4bit']
        }
    }
    
    total_energy = sum(comp['energy'] for comp in components.values())
    total_area = sum(comp['area'] for comp in components.values())
    
    return {
        'config': f"W{weight_bits}A{activation_bits}",
        'components': components,
        'total_energy': total_energy,
        'total_area': total_area
    }

def calculate_pe_costs(tile_dim: int = 4, weight_bits: int = 4, activation_bits: int = 8) -> Dict:

    elements_per_tile = tile_dim * tile_dim
    base_adders = tile_dim * weight_bits - 1
    compressed_adders = base_adders // 4
    
    spmpgemm_components = {
        'transmission_gates': {
            'count': elements_per_tile * weight_bits,
            'energy': elements_per_tile * weight_bits * HardwareCosts.ENERGY['transmission_gate'],
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
    
    mixpe_costs = calculate_mixpe_costs(weight_bits, activation_bits)
    
    spmpgemm_total_energy = sum(comp['energy'] for comp in spmpgemm_components.values())
    spmpgemm_total_area = sum(comp['area'] for comp in spmpgemm_components.values())
    
    mac_total_energy = sum(comp['energy'] for comp in mac_components.values())
    mac_total_area = sum(comp['area'] for comp in mac_components.values())
    
    return {
        'tile_config': f"{tile_dim}x{tile_dim} W{weight_bits}A{activation_bits}",
        'spmpgemm': {
            'components': spmpgemm_components,
            'total_energy': spmpgemm_total_energy,
            'total_area': spmpgemm_total_area
        },
        'mac': {
            'components': mac_components,
            'total_energy': mac_total_energy,
            'total_area': mac_total_area
        },
        'mixpe': {
            'components': mixpe_costs['components'],
            'total_energy': mixpe_costs['total_energy'],
            'total_area': mixpe_costs['total_area']
        },
        'comparison': {
            'spmpgemm_mac_energy_ratio': spmpgemm_total_energy / mac_total_energy,
            'spmpgemm_mac_area_ratio': spmpgemm_total_area / mac_total_area,
            'mixpe_mac_energy_ratio': mixpe_costs['total_energy'] / mac_total_energy,
            'mixpe_mac_area_ratio': mixpe_costs['total_area'] / mac_total_area
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
            'total_spmpgemm_energy': pe_costs['spmpgemm']['total_energy'] * num_tiles,
            'total_mac_energy': pe_costs['mac']['total_energy'] * num_tiles,
            'total_mixpe_energy': pe_costs['mixpe']['total_energy'] * num_tiles,
            'total_spmpgemm_area': pe_costs['spmpgemm']['total_area'] * num_tiles,
            'total_mac_area': pe_costs['mac']['total_area'] * num_tiles,
            'total_mixpe_area': pe_costs['mixpe']['total_area'] * num_tiles
        })
    
    return results

def plot_scaling_analysis(matrix_sizes: list, results: list):
    plt.figure(figsize=(20, 10))
    
    # Energy
    plt.subplot(2, 2, 1)
    plt.plot(matrix_sizes, [r['total_spmpgemm_energy'] for r in results], 'b-o', label='SpMpGEMM')
    # plt.plot(matrix_sizes, [r['total_mac_energy'] for r in results], 'r-o', label='MAC')
    plt.plot(matrix_sizes, [r['total_mixpe_energy'] for r in results], 'g-o', label='MixPE')
    plt.title('Total Energy vs Matrix Size')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Total Energy (pJ)')
    plt.grid(True)
    plt.legend()
    
    # Area 
    plt.subplot(2, 2, 2)
    plt.plot(matrix_sizes, [r['total_spmpgemm_area'] for r in results], 'b-o', label='SpMpGEMM')
    # plt.plot(matrix_sizes, [r['total_mac_area'] for r in results], 'r-o', label='MAC')
    plt.plot(matrix_sizes, [r['total_mixpe_area'] for r in results], 'g-o', label='MixPE')
    plt.title('Total Area vs Matrix Size')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Total Area (μm²)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Analysis for a single PE:")
    results = calculate_pe_costs(tile_dim=4, weight_bits=4, activation_bits=8)

    print("\nAnalyzing scaling with matrix size...")
    matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    scaling_results = analyse_matrix_scaling(matrix_sizes)
    
    plot_scaling_analysis(matrix_sizes, scaling_results)