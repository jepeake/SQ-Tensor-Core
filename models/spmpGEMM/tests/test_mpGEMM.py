import numpy as np
from spmp_gemm import SIMDEngine
from preprocessing.preprocess_weights import preprocess_weights

def test_mpGEMM():

    # Test with Random Weights and Activations
    N = 128
    weights = np.random.randint(0, 15, size=(N, N), dtype=np.int8)          # UINT4
    activations = np.random.randint(-128, 127, size=(N, N), dtype=np.int32) # INT8
    
    print("\nWeight Matrix:")
    print(weights)
    print("\nActivation Matrix:")
    print(activations)
    
    preprocess_weights(weights, num_bits=4, tile_size=4)
    
    engine = SIMDEngine("weight_bits.bin")
    
    activations_flat = activations.flatten().tolist()
    
    result = engine.compute(activations_flat)
    
    result_array = np.array(result.data).reshape(N, N)
    print("\nResult Matrix:")
    print(result_array)

    expected_result = np.matmul(activations, weights)    
    print("\nExpected Result (Software):")
    print(expected_result)

    assert np.allclose(result_array, expected_result, atol=1e-2)

if __name__ == "__main__":
    test_mpGEMM() 