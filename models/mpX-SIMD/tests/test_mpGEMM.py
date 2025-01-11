import numpy as np
from mpx_simd import SIMDEngine
from preprocessing.preprocess_weights import preprocess_weights

def test_mpGEMM():
    # Set dimensions
    N = 7
    
    # Generate random weights and activations
    weights = np.random.randint(0, 4, size=(N, N), dtype=np.int8)
    activations = np.random.randint(0, 4, size=(N, N), dtype=np.int8)
    
    # Print original matrices
    print("\nWeight Matrix:")
    print(weights)
    print("\nActivation Matrix:")
    print(activations)
    
    # Preprocess weights
    preprocess_weights(weights, num_bits=4, tile_size=4)
    
    engine = SIMDEngine("weight_bits.bin")
    
    # Flatten activations for engine input
    activations_flat = activations.flatten().tolist()
    
    result = engine.compute(activations_flat)
    
    # Reshape and print result
    result_array = np.array(result.data).reshape(N, N)
    print("\nResult Matrix:")
    print(result_array)

    expected_result = np.matmul(activations, weights)    
    print("\nExpected Result (Software):")
    print(expected_result)

if __name__ == "__main__":
    test_mpGEMM() 