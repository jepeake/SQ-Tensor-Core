import numpy as np
from spmp_gemm import SIMDEngine
from preprocessing.preprocess_weights import preprocess_weights

def test_mpGEMM():

    # Test with Random Weights and Activations
    N = 8
    weights = np.random.randint(0, 15, size=(N, N), dtype=np.int8)          # UINT4
    activations = np.random.randint(-128, 127, size=(N, N), dtype=np.int32) # INT8
    
    print("\nWeight Matrix:")
    print(weights)
    print("\nActivation Matrix:")
    print(activations)
    
    preprocess_weights(weights, num_bits=4, tile_size=4)
    
    engine = SIMDEngine("weight_bits.bin")
    
    activations_flat = activations.flatten().tolist()
    activation_threshold = 0  
    
    result = engine.compute(activations_flat, activation_threshold)
    
    result_array = np.array(result.data).reshape(N, N)
    print("\nResult Matrix:")
    print(result_array)

    expected_result = np.matmul(activations, weights)    
    print("\nExpected Result (Software):")
    print(expected_result)

    # Print Stats
    print("\nHardware Statistics:")
    print("-------------------")
    stats = engine.get_stats()
    for i, pe_stat in enumerate(stats.pe_stats):
        print(f"\nProcessing Element {i}:")
        print(f"  Total Cycles: {pe_stat.total_cycles}")
        print(f"  Per-cycle Operations:")
        print(f"    - Masking: {pe_stat.masking_operations}")
        print(f"    - Shifting: {pe_stat.shifting_operations}")
        print(f"    - Addition: {pe_stat.addition_operations}")
        print(f"  Total Operations:")
        print(f"    - Mask Ops: {pe_stat.total_mask_ops}")
        print(f"    - Shifts: {pe_stat.total_shifts}")
        print(f"    - Additions: {pe_stat.total_additions}")

    assert np.allclose(result_array, expected_result, atol=1e-2)

if __name__ == "__main__":
    test_mpGEMM() 