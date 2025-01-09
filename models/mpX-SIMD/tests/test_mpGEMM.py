import numpy as np
from mpx_simd import SIMDEngine
from preprocessing.preprocess_weights import preprocess_weights

def test_mpGEMM():

    N = 4
    weights = np.random.randint(0, 15, size=(N, N), dtype=np.int8)
    preprocess_weights(weights, num_bits=4, tile_size=4)
    
    engine = SIMDEngine("weight_bits.bin")
    
    activations = [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    ]
    
    result = engine.compute(activations)
    
    result_array = np.array(result.data).reshape(result.rows, result.cols)
    print("\nResult as NumPy Array:")
    print(result_array)

if __name__ == "__main__":
    test_mpGEMM() 