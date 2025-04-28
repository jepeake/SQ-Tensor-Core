import numpy as np
import pytest
from test_matmul import run_matmul_test

test_params = [
    (2, 2, 4),      # 2   = 2^1 (using tile_size=2)
    (4, 2, 4),      # 4   = 2^2
    (8, 4, 4),      # 8   = 2^3
    (16, 4, 4),     # 16  = 2^4
    (32, 8, 4),     # 32  = 2^5
    (64, 8, 4),     # 64  = 2^6
    (128, 8, 4),    # 128 = 2^7
    (256, 16, 4),   # 256 = 2^8
    (512, 16, 4),   # 512 = 2^9
    (1024, 16, 4),  # 1024 = 2^10
    # (2048, 32, 4),  # 2048 = 2^11
    # (4096, 32, 4),  # 4096 = 2^12
    # (8192, 64, 4)   # 8192 = 2^13
]

@pytest.mark.parametrize("matrix_size, tile_size, num_bits", test_params)
def test_multiple_matmuls(matrix_size, tile_size, num_bits):
    result, reference, stats = run_matmul_test(matrix_size, tile_size, num_bits, verbose=False)
    assert np.array_equal(result, reference), (
        f"Mismatch for matrix_size={matrix_size}, tile_size={tile_size}, num_bits={num_bits}"
    )

if __name__ == "__main__":
    total_tests = len(test_params)
    passed_tests = 0

    print("\nTest Suite Results")
    print("=" * 60)

    for matrix_size, tile_size, num_bits in test_params:
        try:
            result, reference, stats = run_matmul_test(matrix_size, tile_size, num_bits, verbose=False)
            if np.array_equal(result, reference):
                status = "✅"
                passed_tests += 1
            else:
                status = "❌"
            resource_summary = (
                f"Cycles: {getattr(stats, 'total_parallel_cycles', 'N/A')}, "
                f"Mask Ops: {getattr(stats, 'total_parallel_mask_ops', 'N/A')}, "
                f"Shift Ops: {getattr(stats, 'total_parallel_shifts', 'N/A')}, "
                f"Addition Ops: {getattr(stats, 'total_parallel_additions', 'N/A')}"
            )
            print(f"Test for matrix_size={matrix_size}, tile_size={tile_size}, num_bits={num_bits}: {status}")
            print(f"Resource Usage: {resource_summary}")
            print("-" * 60)
        except Exception as e:
            print(f"Test for matrix_size={matrix_size}, tile_size={tile_size}, num_bits={num_bits}: ❌")
            print(f"Error: {str(e)}")
            print("-" * 60)

    print(f"Summary: {passed_tests}/{total_tests} tests passed.") 