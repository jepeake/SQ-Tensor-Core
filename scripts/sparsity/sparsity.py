import argparse
import random
from typing import Dict, List, Tuple, Literal
from dataclasses import dataclass
import heapq

import numpy as np
from safetensors.numpy import load_file
from tqdm import tqdm

try:
    import torch
    from safetensors.torch import load_file as torch_load_file
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def sample_tensor(tensor: np.ndarray, max_elements: int = 100_000, rng: random.Random = random) -> np.ndarray:
    flat = tensor.ravel()
    n = flat.shape[0]
    if n <= max_elements:
        return flat
    indices = rng.sample(range(n), k=max_elements)
    return flat[indices]


def block_sparsity(tensor: np.ndarray, block_size: int = 4) -> float:
    """Compute Block-Wise Sparsity (Fraction of Blocks that are 0s)"""
    if tensor.ndim < 2:
        return float(np.allclose(tensor, 0.0))

    h, w = tensor.shape[-2:]

    # Pad Tensor so h & w are multiples of block_size
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    if pad_h or pad_w:
        pad_width = [(0, 0)] * tensor.ndim
        pad_width[-2] = (0, pad_h)
        pad_width[-1] = (0, pad_w)
        tensor = np.pad(tensor, pad_width, mode="constant")
        h += pad_h
        w += pad_w

    new_shape = (*tensor.shape[:-2], h // block_size, block_size, w // block_size, block_size)
    # Move Block Dimensions Together: (..., n_blocks_h, n_blocks_w, block_size, block_size)
    tensor_reshaped = tensor.reshape(new_shape)
    # Bring Block Dimensions to Front for Easier Reduction
    # Combine Leading Dimensions except Blocks
    blocks = tensor_reshaped.swapaxes(-3, -2).reshape(-1, block_size, block_size)
    # Check Each Block Individually
    is_zero_block = np.array([np.all(np.isclose(block, 0.0, atol=1e-8)) for block in blocks])
    return is_zero_block.mean().item()


def word_sparsity(sample: np.ndarray, threshold: float = 0.0) -> float:
    """Fraction of Zero-Valued Elements in the Sample"""
    if threshold > 0.0:
        # Count Elements Whose Absolute Value is Below the Threshold
        return np.count_nonzero(np.abs(sample) < threshold) / sample.size
    else:
        # Count Exact Zeros Only
        return np.count_nonzero(sample == 0) / sample.size


def bit_sparsity(sample: np.ndarray) -> float:
    """Fraction of Zero Bits in the IEEE-754 Binary Representation of the Sample"""
    # Convert to Numpy Array of Bytes
    byte_view = sample.view(np.uint8)
    bits = np.unpackbits(byte_view, bitorder="big")
    return np.count_nonzero(bits == 0) / bits.size


def calculate_bit_block_sparsity(sample: np.ndarray, block_size: int = 4) -> float:
    """Fraction of Bit Blocks that are All Zeros"""
    # For 1D samples, reshape to 2D if possible
    if sample.ndim == 1:
        size = sample.size
        side_length = int(np.sqrt(size))
        if side_length * side_length == size:  # Perfect square
            sample = sample.reshape(side_length, side_length)
        else:
            # Make a square-ish 2D array
            new_side = int(np.ceil(np.sqrt(size)))
            padded = np.zeros((new_side * new_side,), dtype=sample.dtype)
            padded[:size] = sample
            sample = padded.reshape(new_side, new_side)
    
    if sample.ndim != 2:
        # For higher dimensions, just use the first 2 dims or flatten and reshape
        if sample.ndim > 2:
            # Take first two dimensions if they're big enough
            if sample.shape[0] >= block_size and sample.shape[1] >= block_size:
                sample = sample[0:sample.shape[0]-(sample.shape[0] % block_size), 
                               0:sample.shape[1]-(sample.shape[1] % block_size)]
            else:
                # Flatten and reshape to 2D
                size = sample.size
                new_side = int(np.ceil(np.sqrt(size)))
                padded = np.zeros((new_side * new_side,), dtype=sample.dtype)
                padded[:size] = sample.ravel()
                sample = padded.reshape(new_side, new_side)
     
    # Make sure the sample dimensions are multiples of block_size
    h, w = sample.shape
    h_adjusted = h - (h % block_size) if h % block_size != 0 else h
    w_adjusted = w - (w % block_size) if w % block_size != 0 else w
    
    if h_adjusted == 0 or w_adjusted == 0:
        return 0.0  # Not enough data for even one block
    
    sample = sample[:h_adjusted, :w_adjusted]
    
    # Convert each value to its binary representation
    # For floating point values this includes sign, exponent, and mantissa bits
    byte_view = sample.ravel().view(np.uint8)
    
    # We'll analyze bit block sparsity per bit position across the tensor
    # This better captures the true bit-level block sparsity
    
    # Determine number of bits per element based on dtype
    if sample.dtype in [np.float32, np.int32]:
        bits_per_element = 32
    elif sample.dtype in [np.float64, np.int64]:
        bits_per_element = 64
    elif sample.dtype in [np.float16, np.int16]:
        bits_per_element = 16
    elif sample.dtype in [np.int8, np.uint8]:
        bits_per_element = 8
    else:
        # For other types, use 32 bits as a default
        bits_per_element = 32
    
    total_blocks = 0
    zero_blocks = 0
    
    # For each bit position...
    for bit_pos in range(bits_per_element):
        # Create a bit plane array
        bit_plane = np.zeros((h_adjusted, w_adjusted), dtype=np.uint8)
        
        # Extract the bit at position bit_pos for each element
        flat_idx = 0
        for i in range(h_adjusted):
            for j in range(w_adjusted):
                # Calculate which byte and bit within that byte
                element_idx = i * w_adjusted + j
                byte_idx = (element_idx * bits_per_element + bit_pos) // 8
                bit_idx = (element_idx * bits_per_element + bit_pos) % 8
                
                # Check if we have enough bytes
                if byte_idx < byte_view.size:
                    # Extract the bit
                    bit_value = (byte_view[byte_idx] >> (7 - bit_idx)) & 1
                    bit_plane[i, j] = bit_value
        
        # Now analyze blocks in this bit plane
        for i in range(0, h_adjusted, block_size):
            for j in range(0, w_adjusted, block_size):
                total_blocks += 1
                block = bit_plane[i:i+block_size, j:j+block_size]
                if np.all(block == 0):
                    zero_blocks += 1
    
    # Return fraction of zero blocks
    return zero_blocks / max(1, total_blocks)


def compute_weighted_bit_blocks(tensor: np.ndarray, bit_block_size: int = 4) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """Convert Tensor to Bit Blocks and Compute Weighted Norms for Each Block
    
    Returns:
        - bit_blocks: The Reshaped Bit Blocks (n_blocks, block_size*block_size)
        - block_norms: The L1-Norm of Each Bit Block Weighted by Bit Position
        - block_shape: The Shape Information for Reconstructing the Tensor
    """
    # Get Original Shape for Reconstruction
    orig_shape = tensor.shape
    
    # Get flat binary representation
    byte_view = tensor.ravel().view(np.uint8)
    bits = np.unpackbits(byte_view, bitorder="big")
    
    # Calculate How Many Complete Blocks We Can Form
    n_bits = bits.size
    n_complete_blocks = n_bits // (bit_block_size * bit_block_size)
    
    if n_complete_blocks == 0:
        # Not Enough Bits for Even One Block
        return bits.reshape(1, -1), np.array([0]), (1,)
    
    # Use Only Bits That Form Complete Blocks
    n_bits_to_use = n_complete_blocks * (bit_block_size * bit_block_size)
    bits_to_use = bits[:n_bits_to_use]
    
    # Reshape into Bit Blocks
    bit_blocks = bits_to_use.reshape(n_complete_blocks, bit_block_size * bit_block_size)
    
    # Create Position Weights (Magnitude Based on Bit Position)
    # IEEE 754 Bit Positions: Sign Bit, Exponent Bits, Mantissa Bits
    # We'll Use a Simpler Approach: Weight by Position in the Block
    # Higher Weight for Higher-Order Bits (Left-Most Bits in the Block)
    weights = 2.0 ** np.arange(bit_block_size * bit_block_size - 1, -1, -1)
    
    # Compute Weighted L1 Norm for Each Block
    # Convert Bits to Float for Weighting
    block_norms = np.sum(bit_blocks.astype(np.float32) * weights, axis=1)
    
    # Block Shape Info for Reconstruction
    block_shape = (n_complete_blocks, bit_block_size * bit_block_size)
    
    return bit_blocks, block_norms, block_shape


def apply_bit_block_pruning(tensor: np.ndarray, target_sparsity: float, bit_block_size: int = 4) -> np.ndarray:
    """Apply Bit-Level Block Pruning to Achieve Target Sparsity Level
    
    This Sets the Lowest-Scoring Bit Blocks to All Zeros, Where Score is Weighted by Bit Position.
    """
    if target_sparsity <= 0.0:
        return tensor.copy()
    
    # Only handles 2D tensors correctly for now
    if tensor.ndim != 2:
        # Make a best-effort for non-2D tensors, but results may not be optimal
        orig_shape = tensor.shape
        if tensor.ndim == 1:
            # For 1D, reshape to square-ish 2D
            size = tensor.size
            new_side = int(np.ceil(np.sqrt(size)))
            padded = np.zeros((new_side * new_side,), dtype=tensor.dtype)
            padded[:size] = tensor.ravel()
            tensor_2d = padded.reshape(new_side, new_side)
            
            # Apply pruning to 2D version
            pruned_2d = apply_bit_block_pruning(tensor_2d, target_sparsity, bit_block_size)
            
            # Extract relevant portion and reshape back
            return pruned_2d.ravel()[:size].reshape(orig_shape)
        else:
            # For higher dimensions, just flatten to 2D and then reshape back
            tensor_2d = tensor.reshape(tensor.shape[0], -1)
            pruned_2d = apply_bit_block_pruning(tensor_2d, target_sparsity, bit_block_size)
            return pruned_2d.reshape(orig_shape)
    
    # Make dimensions multiples of bit_block_size
    h, w = tensor.shape
    h_adjusted = h - (h % bit_block_size) if h % bit_block_size != 0 else h
    w_adjusted = w - (w % bit_block_size) if w % bit_block_size != 0 else w
    
    # If tensor is too small, return unchanged
    if h_adjusted == 0 or w_adjusted == 0:
        return tensor.copy()
    
    # Create a working copy and handle padding if needed
    pruned = tensor.copy()
    
    # Determine bits per element
    if tensor.dtype in [np.float32, np.int32]:
        bits_per_element = 32
    elif tensor.dtype in [np.float64, np.int64]:
        bits_per_element = 64
    elif tensor.dtype in [np.float16, np.int16]:
        bits_per_element = 16
    elif tensor.dtype in [np.int8, np.uint8]:
        bits_per_element = 8
    else:
        # For other types, use 32 bits as a default
        bits_per_element = 32
    
    # Extract valid region
    valid_region = tensor[:h_adjusted, :w_adjusted]
    
    # Flatten the region to analyze bits
    flat_view = valid_region.ravel()
    byte_view = flat_view.view(np.uint8)
    
    # Create a bit importance map (higher value = more important)
    # Weight by bit position, giving more importance to sign and higher-order bits
    n_elements = valid_region.size
    bit_importance = np.zeros((h_adjusted, w_adjusted, bits_per_element), dtype=np.float32)
    
    # Fill bit importance based on position and value
    for bit_pos in range(bits_per_element):
        # Higher weight for more significant bits
        bit_weight = 2.0 ** (bits_per_element - 1 - bit_pos)
        
        # Create a bit plane for this position
        bit_plane = np.zeros((h_adjusted, w_adjusted), dtype=np.uint8)
        
        # Extract bits
        for i in range(h_adjusted):
            for j in range(w_adjusted):
                element_idx = i * w_adjusted + j
                byte_idx = (element_idx * bits_per_element + bit_pos) // 8
                bit_idx = (element_idx * bits_per_element + bit_pos) % 8
                
                if byte_idx < byte_view.size:
                    bit_value = (byte_view[byte_idx] >> (7 - bit_idx)) & 1
                    bit_plane[i, j] = bit_value
                    
                    # If bit is set, record its importance
                    if bit_value:
                        bit_importance[i, j, bit_pos] = bit_weight
    
    # Compute block importance by summing bit importances within each block
    n_blocks_h = h_adjusted // bit_block_size
    n_blocks_w = w_adjusted // bit_block_size
    block_importance = np.zeros((n_blocks_h, n_blocks_w), dtype=np.float32)
    
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            i_start, i_end = i * bit_block_size, (i + 1) * bit_block_size
            j_start, j_end = j * bit_block_size, (j + 1) * bit_block_size
            
            # Sum importance of all bits in this block
            block_importance[i, j] = np.sum(bit_importance[i_start:i_end, j_start:j_end, :])
    
    # Determine threshold for pruning
    block_importances_flat = block_importance.ravel()
    n_blocks = block_importances_flat.size
    n_to_prune = int(n_blocks * target_sparsity)
    
    if n_to_prune > 0:
        # Find threshold
        threshold = np.partition(block_importances_flat, n_to_prune)[n_to_prune]
        
        # Create a mask for pruning (True = keep, False = prune)
        keep_mask = block_importance > threshold
        
        # Create a full-sized mask for the tensor
        full_mask = np.zeros((h_adjusted, w_adjusted), dtype=bool)
        
        # Fill the mask
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                i_start, i_end = i * bit_block_size, (i + 1) * bit_block_size
                j_start, j_end = j * bit_block_size, (j + 1) * bit_block_size
                
                # If we're keeping this block, set its mask region to True
                if keep_mask[i, j]:
                    full_mask[i_start:i_end, j_start:j_end] = True
        
        # Apply the mask
        pruned[:h_adjusted, :w_adjusted] = np.where(full_mask, valid_region, 0)
    
    return pruned


@dataclass
class TensorStats:
    name: str
    shape: Tuple[int, ...]
    word_sparsity: float
    bit_sparsity: float
    block_sparsity: float
    bit_block_sparsity: float


def load_safetensors(file_path: str) -> Dict[str, np.ndarray]:
    try:
        # Try Loading with Numpy Backend First (Faster for Simple Types)
        return load_file(file_path)
    except TypeError as e:
        if "bfloat16" in str(e) and TORCH_AVAILABLE:
            print("Detected bfloat16 tensors, loading with PyTorch backend...")
            # Load with PyTorch Backend and Convert to Numpy
            torch_dict = torch_load_file(file_path)
            # Convert to Numpy - Handling bfloat16 by Converting to float32 First
            numpy_dict = {}
            for k, v in torch_dict.items():
                # Convert bfloat16 to float32 Before Conversion to Numpy
                if v.dtype == torch.bfloat16:
                    numpy_dict[k] = v.to(torch.float32).cpu().numpy()
                else:
                    numpy_dict[k] = v.cpu().numpy()
            return numpy_dict
        elif "bfloat16" in str(e):
            raise ImportError(
                "Model contains bfloat16 tensors which require PyTorch. "
                "Please install PyTorch: pip install torch"
            ) from e
        else:
            raise


def compute_block_norms(tensor: np.ndarray, block_size: int = 4, norm_type: Literal["max", "l2"] = "l2") -> np.ndarray:
    """Compute Norms for Each Block in the Tensor
    
    Args:
        tensor: Input Tensor
        block_size: Size of Square Blocks
        norm_type: Type of Norm to Compute ('max' or 'l2')
        
    Returns:
        Array of Norms for Each Block with Shape [num_blocks]
    """
    if tensor.ndim < 2:
        # For 1-D Tensors - Treat the Whole Tensor as a Single Block
        if norm_type == "max":
            return np.array([np.max(np.abs(tensor))])
        else:  # l2
            return np.array([np.sqrt(np.sum(tensor**2))])

    # Get Last Two Dimensions
    h, w = tensor.shape[-2:]
    
    # Pad Tensor so h and w are Multiples of block_size
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    if pad_h or pad_w:
        pad_width = [(0, 0)] * tensor.ndim
        pad_width[-2] = (0, pad_h)
        pad_width[-1] = (0, pad_w)
        tensor = np.pad(tensor, pad_width, mode="constant")
        h += pad_h
        w += pad_w
    
    # Reshape to Blocks
    new_shape = (*tensor.shape[:-2], h // block_size, block_size, w // block_size, block_size)
    tensor_reshaped = tensor.reshape(new_shape)
    
    # Combine Leading Dimensions and Block Indices
    blocks = tensor_reshaped.swapaxes(-3, -2).reshape(-1, block_size, block_size)
    
    # Compute Norms
    if norm_type == "max":
        norms = np.max(np.abs(blocks), axis=(1, 2))
    else:  # l2
        norms = np.sqrt(np.sum(blocks**2, axis=(1, 2)))
        
    return norms


def apply_block_pruning(
    tensor: np.ndarray, 
    target_sparsity: float,
    block_size: int = 4, 
    norm_type: Literal["max", "l2"] = "l2"
) -> np.ndarray:
    """Apply Block-Wise Pruning to Achieve Target Sparsity Level
    
    Args:
        tensor: Input Tensor
        target_sparsity: Target Block Sparsity (0.0-1.0)
        block_size: Size of Square Blocks
        norm_type: Type of Norm to Use for Block Scoring ('max' or 'l2')
        
    Returns:
        Pruned Tensor with Same Shape as Input
    """
    if tensor.ndim < 2 or target_sparsity <= 0.0:
        return tensor.copy()
        
    # Get Original Shape
    orig_shape = tensor.shape
    h, w = orig_shape[-2:]
    
    # Pad Tensor so h and w are Multiples of block_size
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    if pad_h or pad_w:
        pad_width = [(0, 0)] * tensor.ndim
        pad_width[-2] = (0, pad_h)
        pad_width[-1] = (0, pad_w)
        tensor = np.pad(tensor, pad_width, mode="constant")
    
    padded_shape = tensor.shape
    h_padded, w_padded = padded_shape[-2:]
    
    # Reshape to Blocks
    new_shape = (*tensor.shape[:-2], h_padded // block_size, block_size, w_padded // block_size, block_size)
    tensor_reshaped = tensor.reshape(new_shape)
    
    # Get Block Indices and Norms
    n_blocks_h = h_padded // block_size
    n_blocks_w = w_padded // block_size
    
    # Compute Norms for Each Block
    norms = compute_block_norms(tensor, block_size, norm_type)
    
    # Determine Threshold for Pruning
    n_blocks = norms.size
    n_to_prune = int(n_blocks * target_sparsity)
    
    if n_to_prune > 0:
        # Find Threshold (kth Smallest Norm)
        threshold = np.partition(norms, n_to_prune)[n_to_prune - 1]
        
        # Create Mask for Blocks to Keep (True = keep, False = prune)
        keep_mask = norms > threshold
        
        # Reshape Tensor and Apply Mask
        pruned_tensor = tensor.copy()
        batch_dims = tensor.ndim - 2  # Number of Dimensions Before the Last 2
        
        # Handle Masks with Reshaping to Apply to Original Tensor
        reshaped_mask = keep_mask.reshape(*([1] * batch_dims), n_blocks_h, 1, n_blocks_w, 1)
        
        # Broadcast Mask to Full Block Size
        block_mask = np.broadcast_to(
            reshaped_mask, 
            (*([1] * batch_dims), n_blocks_h, block_size, n_blocks_w, block_size)
        )
        
        # Reshape Mask to Match Tensor Shape
        block_mask = block_mask.reshape(padded_shape)
        
        # Apply Mask (Zero Out Pruned Blocks)
        pruned_tensor = pruned_tensor * block_mask
        
        # Remove Padding if Needed
        if pad_h or pad_w:
            slices = [slice(None)] * tensor.ndim
            slices[-2] = slice(0, h)
            slices[-1] = slice(0, w)
            pruned_tensor = pruned_tensor[tuple(slices)]
        
        return pruned_tensor
    else:
        # No Pruning Needed
        return tensor.copy()


@dataclass
class SimulatedPruningStats:
    word_block_sparsity_target: float
    bit_block_sparsity_target: float
    word_norm_type: str
    word_sparsity_before: float
    word_sparsity_after_word_pruning: float
    word_sparsity_after_bit_pruning: float
    block_sparsity_before: float
    block_sparsity_after_word_pruning: float 
    block_sparsity_after_bit_pruning: float
    bit_sparsity_before: float
    bit_sparsity_after_word_pruning: float
    bit_sparsity_after_bit_pruning: float
    bit_block_sparsity_before: float
    bit_block_sparsity_after_word_pruning: float
    bit_block_sparsity_after_bit_pruning: float


def analyse_pruning(
    tensor_dict: Dict[str, np.ndarray],
    tensor_names: List[str], 
    word_block_sparsity: float,
    bit_block_sparsity: float,
    block_size: int = 4,
    bit_block_size: int = 4,
    word_norm_type: Literal["max", "l2"] = "l2",
    max_elements_per_tensor: int = 100_000,
    rng: random.Random = random
) -> SimulatedPruningStats:
    """Analyse the Effect of Block Pruning on Sparsity Metrics"""
    word_sparsity_before = 0.0
    word_sparsity_after_word = 0.0
    word_sparsity_after_bit = 0.0
    block_sparsity_before = 0.0
    block_sparsity_after_word = 0.0
    block_sparsity_after_bit = 0.0
    bit_sparsity_before = 0.0
    bit_sparsity_after_word = 0.0
    bit_sparsity_after_bit = 0.0
    bit_block_sparsity_before = 0.0
    bit_block_sparsity_after_word = 0.0
    bit_block_sparsity_after_bit = 0.0
    
    count = 0
    
    desc = f"Simulating Pruning (Word:{word_block_sparsity*100:.0f}%, Bit:{bit_block_sparsity*100:.0f}%)"
    for name in tqdm(tensor_names, desc=desc):
        if count >= 16:  # Limit Analysis to Not Be Too Slow
            break
            
        tensor = tensor_dict[name]
        if tensor.ndim < 2:
            continue  # Skip 1D Tensors
            
        count += 1
            
        # Sample Tensor for More Efficient Computation
        sample_before = sample_tensor(tensor, max_elements=max_elements_per_tensor, rng=rng)
        
        # Step 1: Apply Word-Level Block Pruning
        tensor_after_word_pruning = apply_block_pruning(
            tensor.astype(np.float32), 
            target_sparsity=word_block_sparsity,
            block_size=block_size,
            norm_type=word_norm_type
        )
        
        # Sample After Word Pruning
        sample_after_word = sample_tensor(tensor_after_word_pruning, max_elements=max_elements_per_tensor, rng=rng)
        
        # Step 2: Apply Bit-Level Block Pruning on Top of Word Pruning
        tensor_after_bit_pruning = apply_bit_block_pruning(
            tensor_after_word_pruning,
            target_sparsity=bit_block_sparsity,
            bit_block_size=bit_block_size
        )
        
        # Sample After Bit Pruning
        sample_after_bit = sample_tensor(tensor_after_bit_pruning, max_elements=max_elements_per_tensor, rng=rng)
        
        # Compute Metrics: Original
        ws_before = word_sparsity(sample_before)
        bs_before = block_sparsity(tensor.astype(np.float32), block_size=block_size)
        bit_s_before = bit_sparsity(sample_before)
        bit_block_s_before = calculate_bit_block_sparsity(sample_before, block_size=bit_block_size)
        
        # Compute Metrics: After Word Pruning
        ws_after_word = word_sparsity(sample_after_word)
        bs_after_word = block_sparsity(tensor_after_word_pruning, block_size=block_size)
        bit_s_after_word = bit_sparsity(sample_after_word)
        bit_block_s_after_word = calculate_bit_block_sparsity(sample_after_word, block_size=bit_block_size)
        
        # Compute Metrics: After Bit Pruning
        ws_after_bit = word_sparsity(sample_after_bit)
        bs_after_bit = block_sparsity(tensor_after_bit_pruning, block_size=block_size)
        bit_s_after_bit = bit_sparsity(sample_after_bit)
        bit_block_s_after_bit = calculate_bit_block_sparsity(sample_after_bit, block_size=bit_block_size)
        
        # Accumulate Results
        word_sparsity_before += ws_before
        word_sparsity_after_word += ws_after_word
        word_sparsity_after_bit += ws_after_bit
        
        block_sparsity_before += bs_before
        block_sparsity_after_word += bs_after_word
        block_sparsity_after_bit += bs_after_bit
        
        bit_sparsity_before += bit_s_before
        bit_sparsity_after_word += bit_s_after_word
        bit_sparsity_after_bit += bit_s_after_bit
        
        bit_block_sparsity_before += bit_block_s_before
        bit_block_sparsity_after_word += bit_block_s_after_word
        bit_block_sparsity_after_bit += bit_block_s_after_bit
    
    # Average Results
    n = max(1, count)
    return SimulatedPruningStats(
        word_block_sparsity_target=word_block_sparsity,
        bit_block_sparsity_target=bit_block_sparsity,
        word_norm_type=word_norm_type,
        word_sparsity_before=word_sparsity_before / n,
        word_sparsity_after_word_pruning=word_sparsity_after_word / n,
        word_sparsity_after_bit_pruning=word_sparsity_after_bit / n,
        block_sparsity_before=block_sparsity_before / n,
        block_sparsity_after_word_pruning=block_sparsity_after_word / n,
        block_sparsity_after_bit_pruning=block_sparsity_after_bit / n,
        bit_sparsity_before=bit_sparsity_before / n,
        bit_sparsity_after_word_pruning=bit_sparsity_after_word / n,
        bit_sparsity_after_bit_pruning=bit_sparsity_after_bit / n,
        bit_block_sparsity_before=bit_block_sparsity_before / n,
        bit_block_sparsity_after_word_pruning=bit_block_sparsity_after_word / n,
        bit_block_sparsity_after_bit_pruning=bit_block_sparsity_after_bit / n
    )


def analyse_safetensors(
    file_path: str,
    max_tensors: int = 64,
    max_elements_per_tensor: int = 100_000,
    block_size: int = 4,
    bit_block_size: int = 4,
    magnitude_threshold: float = 0.0,
    word_block_sparsity: float = 0.0,
    bit_block_sparsity: float = 0.0,
    word_norm_type: Literal["max", "l2"] = "l2",
    seed: int = 42,
) -> Tuple[List[TensorStats], SimulatedPruningStats]:
    rng = random.Random(seed)

    # Replace Direct load_file Call with Our Custom Function
    tensor_dict: Dict[str, np.ndarray] = load_safetensors(file_path)
    tensor_names = list(tensor_dict.keys())

    # Randomly Choose a Subset of Tensors
    if len(tensor_names) > max_tensors:
        tensor_names = rng.sample(tensor_names, k=max_tensors)

    stats: List[TensorStats] = []

    for name in tqdm(tensor_names, desc="Analysing Tensors"):
        tensor = tensor_dict[name]
        # Subsample Elements for Word/Bit Sparsity
        sample = sample_tensor(tensor, max_elements=max_elements_per_tensor, rng=rng)
        ws = word_sparsity(sample, threshold=magnitude_threshold)
        bs = bit_sparsity(sample)
        blk_s = block_sparsity(tensor.astype(np.float32), block_size=block_size)
        bit_blk_s = calculate_bit_block_sparsity(sample, block_size=bit_block_size)
        stats.append(TensorStats(name, tensor.shape, ws, bs, blk_s, bit_blk_s))
    
    # Analyse Block Pruning if Either Sparsity Target > 0
    pruning_stats = None
    target_word_sparsity = word_block_sparsity
    target_bit_sparsity = bit_block_sparsity
    if target_word_sparsity > 0 or target_bit_sparsity > 0:
        pruning_stats = analyse_pruning(
            tensor_dict,
            tensor_names,
            word_block_sparsity=target_word_sparsity,
            bit_block_sparsity=target_bit_sparsity,
            block_size=block_size,
            bit_block_size=bit_block_size,
            word_norm_type=word_norm_type,
            max_elements_per_tensor=max_elements_per_tensor,
            rng=rng
        )

    return stats, pruning_stats


def aggregate_stats(stats: List[TensorStats]) -> Dict[str, float]:
    agg = {
        "word_sparsity": np.mean([s.word_sparsity for s in stats]),
        "bit_sparsity": np.mean([s.bit_sparsity for s in stats]),
        "block_sparsity": np.mean([s.block_sparsity for s in stats]),
        "bit_block_sparsity": np.mean([s.bit_block_sparsity for s in stats]),
    }
    return agg


def print_pruning_report(stats: SimulatedPruningStats):
    print("\nSimulated Pruning Results:")
    print(f"  Word Block Sparsity Target: {stats.word_block_sparsity_target*100:.1f}%")
    print(f"  Bit Block Sparsity Target:  {stats.bit_block_sparsity_target*100:.1f}%")
    print(f"  Word Block Norm Type: {stats.word_norm_type}")
    
    print("\n  Original Metrics:")
    print(f"    Word Sparsity:       {stats.word_sparsity_before*100:.2f}%")
    print(f"    Block Sparsity:      {stats.block_sparsity_before*100:.2f}%")
    print(f"    Bit Sparsity:        {stats.bit_sparsity_before*100:.2f}%")
    print(f"    Bit Block Sparsity:  {stats.bit_block_sparsity_before*100:.2f}%")
    
    print("\n  After Word-Level Block Pruning:")
    print(f"    Word Sparsity:       {stats.word_sparsity_after_word_pruning*100:.2f}%")
    print(f"    Block Sparsity:      {stats.block_sparsity_after_word_pruning*100:.2f}%")
    print(f"    Bit Sparsity:        {stats.bit_sparsity_after_word_pruning*100:.2f}%")
    print(f"    Bit Block Sparsity:  {stats.bit_block_sparsity_after_word_pruning*100:.2f}%")
    
    print("\n  After Bit-Level Block Pruning:")
    print(f"    Word Sparsity:       {stats.word_sparsity_after_bit_pruning*100:.2f}%")
    print(f"    Block Sparsity:      {stats.block_sparsity_after_bit_pruning*100:.2f}%")
    print(f"    Bit Sparsity:        {stats.bit_sparsity_after_bit_pruning*100:.2f}%")
    print(f"    Bit Block Sparsity:  {stats.bit_block_sparsity_after_bit_pruning*100:.2f}%")


def print_report(stats: List[TensorStats], pruning_stats: SimulatedPruningStats = None):
    print("\nPer-Tensor Sparsity (Sampled):")
    print(f"{'Tensor':40s}  {'Shape':15s}  {'Word%':6s}  {'Bit%':6s}  {'Block%':6s}  {'BitBlk%':7s}")
    for s in stats:
        print(
            f"{s.name:40s}  {str(tuple(s.shape)):15s}  " 
            f"{s.word_sparsity*100:5.1f}%  {s.bit_sparsity*100:5.1f}%  " 
            f"{s.block_sparsity*100:5.1f}%  {s.bit_block_sparsity*100:6.1f}%"
        )

    agg = aggregate_stats(stats)
    print("\nAggregated Sparsity over Sampled Tensors:")
    for k, v in agg.items():
        print(f"  {k}: {v*100:.2f}%")
        
    if pruning_stats:
        print_pruning_report(pruning_stats)


def main():
    parser = argparse.ArgumentParser(description="analyse Sparsity of a .safetensors Weight File.")
    parser.add_argument("model_path", type=str, help="Path to the .safetensors file")
    parser.add_argument("--max-tensors", type=int, default=64, help="Maximum Number of Tensors to Sample")
    parser.add_argument(
        "--max-elements", type=int, default=100_000, help="Maximum Number of Elements to Sample per Tensor"
    )
    parser.add_argument("--block-size", type=int, default=4, help="Block Size for Block Sparsity (NxN Blocks)")
    parser.add_argument("--bit-block-size", type=int, default=4, help="Block Size for Bit Block Sparsity")
    parser.add_argument(
        "--threshold", type=float, default=0.0, 
        help="Magnitude Threshold: Values with abs(x) < threshold are Counted as Zero"
    )
    parser.add_argument(
        "--word-block-sparsity", type=float, default=0.0,
        help="Target Word Block Sparsity to Simulate with Block Pruning (0.0-1.0)"
    )
    parser.add_argument(
        "--bit-block-sparsity", type=float, default=0.0,
        help="Target Bit Block Sparsity to Simulate with Bit Block Pruning (0.0-1.0)"
    )
    parser.add_argument(
        "--norm-type", type=str, choices=["max", "l2"], default="l2",
        help="Norm to Use for Block Importance in Pruning"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random Seed for Sampling")

    args = parser.parse_args()

    stats, pruning_stats = analyse_safetensors(
        args.model_path,
        max_tensors=args.max_tensors,
        max_elements_per_tensor=args.max_elements,
        block_size=args.block_size,
        bit_block_size=args.bit_block_size,
        magnitude_threshold=args.threshold,
        word_block_sparsity=args.word_block_sparsity,
        bit_block_sparsity=args.bit_block_sparsity,
        word_norm_type=args.norm_type,
        seed=args.seed,
    )
    print_report(stats, pruning_stats)


if __name__ == "__main__":
    main() 