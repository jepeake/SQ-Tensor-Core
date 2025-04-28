import random, math

PARAMS = {
    'MATRIX_SIZE': 50,          
    'ZERO_PROBABILITY': 0.5,     
    'RANDOM_SEED': 42,           

    # Simulated Annealing
    'MAX_ITERATIONS': 500000,     
    'START_TEMP': 100.0,       
    'COOLING_RATE': 0.99,        
    'MIN_TEMP': 1e-3,
    
    'TILE_SIZE': 2,          
}

# Generate a Random Binary Matrix
def generate_random_matrix(M, N, p_zero=0.3, seed=None):
    if seed is not None: 
        random.seed(seed)
    return [[0 if random.random() < p_zero else 1 for _ in range(N)] for _ in range(M)]

# Count 2x2 Tiles of 0s
def count_zero_tiles(matrix, row_order=None):
    tile_size = PARAMS['TILE_SIZE']
    M = len(matrix)
    N = len(matrix[0]) if matrix else 0
    if M < tile_size or N < tile_size: 
        return 0
    if row_order is None:
        row_order = range(M)
    return sum(
        1 
        for i in range(0, M - tile_size + 1, tile_size) 
        for j in range(0, N - tile_size + 1, tile_size) 
        if all(matrix[row_order[i + x]][j + y] == 0 for x in range(tile_size) for y in range(tile_size))
    )

# Build Sets of Starting Positions for Runs of 2 Zeroes in Each Row
def find_starting_positions(matrix):
    if not matrix: 
        return []
    tile_size = PARAMS['TILE_SIZE']
    return [
        set(
            j for j in range(len(matrix[0]) - tile_size + 1)
            if all(matrix[r][j + k] == 0 for k in range(tile_size))
        )
        for r in range(len(matrix))
    ]

# Find Overlapping Runs of 2 Zeroes: Now only consider 2 consecutive rows
def find_overlapping_runs(row_win, order, i):
    return len(row_win[order[i]] & row_win[order[i+1]])

# Calculates Total Number of Potential 2x2 Zero Tiles for a Given Row Ordering
def total_zero_tiles(row_win, order):
    tile_size = PARAMS['TILE_SIZE']
    return sum(find_overlapping_runs(row_win, order, idx) for idx in range(len(order) - (tile_size - 1)))

# Generate a New Ordering by Random Swap or Reverse (modified affected range)
def random_move(order):
    M = len(order)
    new_order = order[:]
    tile_size = PARAMS['TILE_SIZE']
    if random.choice(['swap','reverse']) == 'swap':
        i, j = sorted([random.randrange(M), random.randrange(M)])
        new_order[i], new_order[j] = new_order[j], new_order[i]
        affected = [x for x in range(i - (tile_size - 1), j + 1) if 0 <= x <= M - tile_size]
    else:
        start, end = sorted([random.randrange(M), random.randrange(M)])
        if start < end: 
            new_order[start:end+1] = reversed(new_order[start:end+1])
        affected = [x for x in range(start - (tile_size - 1), end + 1) if 0 <= x <= M - tile_size]
    return new_order, list(set(affected))

# Calculate Change in No. of 2x2 Zero Tiles for Affected Indices Only
def delta_zero_tiles(row_win, old_order, new_order, old_val, affected):
    return sum(
        find_overlapping_runs(row_win, new_order, idx) - find_overlapping_runs(row_win, old_order, idx) 
        for idx in affected
    )

# Main Simulated Annealing Optimisation adapted for 2x2 blocks
def simulated_annealing(matrix, init_order, max_iters=None, start_temp=None, cooling_rate=None):
    tile_size = PARAMS['TILE_SIZE']
    if len(matrix) < tile_size: 
        return init_order
    max_iters = max_iters or PARAMS['MAX_ITERATIONS']
    start_temp = start_temp or PARAMS['START_TEMP']
    cooling_rate = cooling_rate or PARAMS['COOLING_RATE']
    
    row_win = find_starting_positions(matrix)
    current_order, current_val = init_order[:], total_zero_tiles(row_win, init_order)
    best_order, best_val = current_order[:], current_val
    
    T = start_temp
    no_improve = 0
    
    for _ in range(max_iters):
        new_ord, affected = random_move(current_order)
        diff = delta_zero_tiles(row_win, current_order, new_ord, current_val, affected)
        
        if (new_val := current_val + diff) > current_val or random.random() < math.exp(diff / T):
            current_order, current_val = new_ord, new_val
            if new_val > best_val:
                best_val, best_order = new_val, current_order[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        # Adaptive Cooling - slow down if not improving
        cooling_factor = cooling_rate * (1.0 + (no_improve / 1000))
        T = max(T * min(cooling_factor, 0.999), PARAMS['MIN_TEMP'])
        
        # Reset temperature if stuck for too long
        if no_improve > 5000:
            T = start_temp * 0.5
            no_improve = 0
            
    return best_order

# Wrapper Function to Initialise and Run Simulated Annealing
def reorder(matrix, max_sa_iters=None):
    return [] if not matrix else simulated_annealing(matrix, list(range(len(matrix))), max_iters=max_sa_iters)

# Visualise Matrix with Highlighted 2x2 Zero Blocks
def print_matrix(matrix, row_order=None):
    if not matrix: 
        return
    if row_order is None: 
        row_order = range(len(matrix))
    tile_size = PARAMS['TILE_SIZE']
    M, N = len(matrix), len(matrix[0])
    zero_block_cells = set(
        (i + di, j + dj)
        for i in range(0, M - tile_size + 1, tile_size)
        for j in range(0, N - tile_size + 1, tile_size)
        for di in range(tile_size)
        for dj in range(tile_size)
        if all(matrix[row_order[i + x]][j + y] == 0 for x in range(tile_size) for y in range(tile_size))
    )
    for i, r in enumerate(row_order):
        print(''.join(
            '⬛' if x == 1 else '🟥' if (i, j) in zero_block_cells else '⬜'
            for j, x in enumerate(matrix[r])
        ))

def induce_zeros(matrix, row_order=None):
    if not matrix:
        return 0
    if row_order is None:
        row_order = list(range(len(matrix)))
    tile_size = PARAMS['TILE_SIZE']
    M = len(matrix)
    N = len(matrix[0])
    flip_count = 0
    
    # Iterate over Disjoint Tiles
    for i in range(0, M - tile_size + 1, tile_size):
        for j in range(0, N - tile_size + 1, tile_size):
            block_coords = [(row_order[i + x], j + y) for x in range(tile_size) for y in range(tile_size)]
            one_positions = [(r, c) for (r, c) in block_coords if matrix[r][c] == 1]
            # If Tile has a Single 1 - Flip to Zero Tile
            if len(one_positions) == 1:
                r, c = one_positions[0]
                matrix[r][c] = 0
                flip_count += 1
    return flip_count

def test_pipeline():
    matrix = generate_random_matrix(
        PARAMS['MATRIX_SIZE'], 
        PARAMS['MATRIX_SIZE'], 
        PARAMS['ZERO_PROBABILITY'], 
        PARAMS['RANDOM_SEED']
    )
    print("Original Matrix:")
    print_matrix(matrix)
    print("\n" + "="*50 + "\n")
    
    before_val = count_zero_tiles(matrix)
    best_order = reorder(matrix)

    print("Reordered Matrix:")
    print_matrix(matrix, best_order)
    val_after_reorder = count_zero_tiles(matrix, best_order)
    
    print("\n" + "="*50 + "\n")
    ones_before = sum(cell == 1 for row in matrix for cell in row)
    flipped = induce_zeros(matrix, best_order)
    percentage_flipped = (flipped / ones_before * 100) if ones_before else 0.0
    
    print("Matrix with Induced Zero Tiles:")
    print_matrix(matrix, best_order)
    val_after_seq = count_zero_tiles(matrix, best_order)
    
    print("\n" + "="*50 + "\n")
    print(
        f"2x2 Blocks BEFORE:            {before_val}\n"
        f"2x2 Blocks AFTER Reordering:  {val_after_reorder}\n"
        f"2x2 Blocks AFTER Flipping:  {val_after_seq}\n"
        f"Total Improvement:            +{val_after_seq - before_val}\n"
        f"Bits Set to Zero:            {flipped} "
        f"({percentage_flipped:.2f}% of Ones)"
    )

if __name__ == "__main__":
    test_pipeline()
