import numpy as np
from spmp_gemm import SIMDEngine
from preprocessing.preprocess_weights import preprocess_weights

def test_mpGEMM():

    # Test with Random Weights and Activations
    N = 6
    weights = np.random.randint(0, 4, size=(N, N), dtype=np.int8)
    activations = np.random.randint(0, 4, size=(N, N), dtype=np.int8)
    
    print("\nWeight Matrix:")
    print(weights)
    print("\nActivation Matrix:")
    print(activations)
    
    preprocess_weights(weights, num_bits=2, tile_size=3)
    
    engine = SIMDEngine("weight_bits.bin")
    
    activations_flat = activations.flatten().tolist()
    
    result = engine.compute(activations_flat)
    
    result_array = np.array(result.data).reshape(N, N)
    print("\nResult Matrix:")
    print(result_array)

    expected_result = np.matmul(activations, weights)    
    print("\nExpected Result (Software):")
    print(expected_result)

if __name__ == "__main__":
    test_mpGEMM() 