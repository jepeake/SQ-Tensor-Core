import random, math

PARAMS = {
    'MATRIX_SIZE': 200,          
    'ZERO_PROBABILITY': 0.5,     
    'RANDOM_SEED': 42,           
    
    # Simulated Annealing
    'MAX_ITERATIONS': 500000,     
    'START_TEMP': 100.0,       
    'COOLING_RATE': 0.99,        
    'MIN_TEMP': 1e-3,          
}

# Generate a Random Binary Matrix
def generate_random_matrix(M, N, p_zero=0.3, seed=None):
    if seed is not None: random.seed(seed)
    return [[0 if random.random() < p_zero else 1 for _ in range(N)] for _ in range(M)]

# Count 4x4 Tiles of 0s
def count_zero_tiles(matrix, row_order=None):
    M, N = len(matrix), len(matrix[0]) if matrix else 0
    if M < 4 or N < 4: return 0
    if row_order is None: row_order = range(M)
    return sum(1 for i in range(0, M-3, 4) for j in range(0, N-3, 4) 
              if all(matrix[row_order[i+x]][j+y]==0 
                    for x in range(4) for y in range(4)))

# Build Sets of Starting Positions for Runs of 4 Zeroes
def find_starting_positions(matrix):
    if not matrix: return []
    return [set(j for j in range(len(matrix[0])-3) if all(matrix[r][j+k]==0 for k in range(4))) 
            for r in range(len(matrix))]

# Find Overlapping Runs of 4 Zeroes
def find_overlapping_runs(row_4win, order, i):
    return len(row_4win[order[i]] & row_4win[order[i+1]] & row_4win[order[i+2]] & row_4win[order[i+3]])

# Calculates Total Number of Potential 4x4 Zero Tiles for a Given Row Ordering
def total_zero_tiles(row_4win, order):
    return sum(find_overlapping_runs(row_4win,order,i) for i in range(len(order)-3))

# Generate a New Ordering by Random Swap or Reverse
def random_move(order):
    M = len(order)
    new_order = order[:]
    if random.choice(['swap','reverse'])=='swap':
        i,j = sorted([random.randrange(M), random.randrange(M)])
        new_order[i],new_order[j] = new_order[j],new_order[i]
        affected = [x for x in range(i-3,j+1) if 0<=x<=M-4]
    else:
        start,end = sorted([random.randrange(M), random.randrange(M)])
        if start<end: new_order[start:end+1] = reversed(new_order[start:end+1])
        affected = [x for x in range(start-3,end+1) if 0<=x<=M-4]
    return new_order, list(set(affected))

# Calculate Change in No. of 4x4 Zero Tiles for Affected Indices Only
def delta_zero_tiles(row_4win, old_order, new_order, old_val, affected):
    return sum(find_overlapping_runs(row_4win,new_order,idx) - find_overlapping_runs(row_4win,old_order,idx) 
              for idx in affected)

# Main Simulated Annealing Optimisation
def simulated_annealing(matrix, init_order, max_iters=None, start_temp=None, cooling_rate=None):
    if len(matrix)<4: return init_order
    max_iters = max_iters or PARAMS['MAX_ITERATIONS']
    start_temp = start_temp or PARAMS['START_TEMP']
    cooling_rate = cooling_rate or PARAMS['COOLING_RATE']
    
    row_4win = find_starting_positions(matrix)
    current_order, current_val = init_order[:], total_zero_tiles(row_4win, init_order)
    best_order, best_val = current_order[:], current_val
    
    T = start_temp
    no_improve = 0
    
    for iter in range(max_iters):
        new_ord, affected = random_move(current_order)
        diff = delta_zero_tiles(row_4win, current_order, new_ord, current_val, affected)
        
        # Accept Move Based on Improvement or Probability
        if (new_val := current_val + diff) > current_val or random.random() < math.exp(diff/T):
            current_order, current_val = new_ord, new_val
            if new_val > best_val:
                best_val, best_order = new_val, current_order[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        # Adaptive Cooling - Slow Down Cooling if We're Not Improving
        cooling_factor = cooling_rate * (1.0 + (no_improve / 1000))  # Slow Cooling if Stuck
        T = max(T * min(cooling_factor, 0.999), PARAMS['MIN_TEMP'])
        
        # Reset Temperature if Stuck for Too Long
        if no_improve > 5000:
            T = start_temp * 0.5
            no_improve = 0
            
    return best_order

# Wrapper Function to Initialise and Run Simulated Annealing
def reorder(matrix, max_sa_iters=None):
    return [] if not matrix else simulated_annealing(matrix, list(range(len(matrix))), max_iters=max_sa_iters)

# Visualize Matrix with Highlighted 4x4 Zero Blocks
def print_matrix(matrix, row_order=None):
    if not matrix: return
    if row_order is None: row_order = range(len(matrix))
    M,N = len(matrix),len(matrix[0])
    zero_block_cells = set((i+di,j+dj) for i in range(0,M-3,4) for j in range(0,N-3,4) 
                          for di in range(4) for dj in range(4) 
                          if all(matrix[row_order[i+x]][j+y]==0 for x in range(4) for y in range(4)))
    for i,r in enumerate(row_order):
        print(''.join('⬛' if x==1 else '🟥' if (i,j) in zero_block_cells else '⬜' 
                     for j,x in enumerate(matrix[r])))

# Test the Optimisation Pipeline
def test_pipeline():
    matrix = generate_random_matrix(
        PARAMS['MATRIX_SIZE'], 
        PARAMS['MATRIX_SIZE'], 
        PARAMS['ZERO_PROBABILITY'], 
        PARAMS['RANDOM_SEED']
    )
    print("Original matrix:"); print_matrix(matrix); print("\n" + "="*50 + "\n")
    before_val = count_zero_tiles(matrix)
    best_order = reorder(matrix)
    print("Reordered matrix:"); print_matrix(matrix, best_order); print("\n" + "="*50 + "\n")
    after_val = count_zero_tiles(matrix, best_order)
    print(f"4x4 blocks BEFORE: {before_val}\n4x4 blocks AFTER:  {after_val}\nImprovement: +{after_val - before_val}")

if __name__=="__main__": test_pipeline()
