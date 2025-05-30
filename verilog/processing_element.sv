

//  ███████████                                                        ███                     
// ░░███░░░░░███                                                      ░░░                      
//  ░███    ░███ ████████   ██████   ██████   ██████   █████   █████  ████  ████████    ███████
//  ░██████████ ░░███░░███ ███░░███ ███░░███ ███░░███ ███░░   ███░░  ░░███ ░░███░░███  ███░░███
//  ░███░░░░░░   ░███ ░░░ ░███ ░███░███ ░░░ ░███████ ░░█████ ░░█████  ░███  ░███ ░███ ░███ ░███
//  ░███         ░███     ░███ ░███░███  ███░███░░░   ░░░░███ ░░░░███ ░███  ░███ ░███ ░███ ░███
//  █████        █████    ░░██████ ░░██████ ░░██████  ██████  ██████  █████ ████ █████░░███████
// ░░░░░        ░░░░░      ░░░░░░   ░░░░░░   ░░░░░░  ░░░░░░  ░░░░░░  ░░░░░ ░░░░ ░░░░░  ░░░░░███
//                                                                                     ███ ░███
//                                                                                    ░░██████ 
//                                                                                     ░░░░░░  
//  ██████████ ████                                                █████                       
// ░░███░░░░░█░░███                                               ░░███                        
//  ░███  █ ░  ░███   ██████  █████████████    ██████  ████████   ███████                      
//  ░██████    ░███  ███░░███░░███░░███░░███  ███░░███░░███░░███ ░░░███░                       
//  ░███░░█    ░███ ░███████  ░███ ░███ ░███ ░███████  ░███ ░███   ░███                        
//  ░███ ░   █ ░███ ░███░░░   ░███ ░███ ░███ ░███░░░   ░███ ░███   ░███ ███                    
//  ██████████ █████░░██████  █████░███ █████░░██████  ████ █████  ░░█████                     
// ░░░░░░░░░░ ░░░░░  ░░░░░░  ░░░░░ ░░░ ░░░░░  ░░░░░░  ░░░░ ░░░░░    ░░░░░     
           
                 
// Architecture:
// -------------
// - Parallel Bit-Plane Processing
// - Single-Cycle Gating and Shifting
// - Fully Parallel Output Summation (One Adder Per Output Element)
// - Sparsity Detection (Skips Zero-Weight Contributions)

// ==================
// Processing Element
// ==================
module processing_element #(
    parameter int TILE_SIZE        = 2,             // Size of Input Tiles (NxN)
    parameter int ACT_WIDTH        = 8,             // Width of Activation Values (int8_t)
    parameter int WEIGHT_WIDTH     = 1,             // Width of Weight Values (Bits)
    parameter int NUM_BIT_PLANES   = 4,             // Number of Weight Bit Planes
    parameter int RESULT_WIDTH     = 16,            // Width of Output Values (int16_t)
    
    parameter type ACTIVATION_T    = logic signed [ACT_WIDTH-1:0],
    parameter type WEIGHT_T        = logic [WEIGHT_WIDTH-1:0],
    parameter type RESULT_T        = logic signed [RESULT_WIDTH-1:0]
)(
    // Control Signals
    input  logic                                  clk,
    input  logic                                  rst_n,
    input  logic                                  start,
    output logic                                  done,
    
    // Configuration
    input  ACTIVATION_T                           activation_threshold,
    
    // Input Data
    input  WEIGHT_T                               weight_tiles[NUM_BIT_PLANES][TILE_SIZE][TILE_SIZE],
    input  ACTIVATION_T                           activation_tile[TILE_SIZE][TILE_SIZE],
    
    // Output Data
    output RESULT_T                               result_tile[TILE_SIZE][TILE_SIZE]
);

    // State Definitions
    typedef enum logic [1:0] {
        IDLE,                // Waiting for Start Signal
        COMPUTE,             // Single Cycle Computation (combines all operations)
        DONE_STATE           // Processing Complete
    } pe_state_t;
    
    pe_state_t state;
    
    // Matrices of Gated and Shifted Contributions for Each Bit Plane
    // Format: [bit_plane][i_row][j_col][k_idx]
    RESULT_T contributions[NUM_BIT_PLANES][TILE_SIZE][TILE_SIZE][TILE_SIZE];
    
    // Input Validity Tracking for Sparse Activation Skipping
    logic [NUM_BIT_PLANES-1:0][TILE_SIZE-1:0][TILE_SIZE-1:0][TILE_SIZE-1:0] contrib_valid;
    
    // Counter for Total Cycles
    logic [31:0] total_cycles;
    
    // ==============
    // State Machine
    // ==============
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            total_cycles <= '0;
            
            // Clear Output Registers
            for (int i = 0; i < TILE_SIZE; i++) begin
                for (int j = 0; j < TILE_SIZE; j++) begin
                    result_tile[i][j] <= '0;
                end
            end
        end else begin
            // Count Cycles for Statistics
            if (state != IDLE && !done) begin
                total_cycles <= total_cycles + 1'b1;
            end
            
            // State Machine Transitions
            case (state)
                IDLE: begin
                    if (start) begin
                        // Move to single compute state
                        state <= COMPUTE;
                        done <= 1'b0;  // Clear done flag when starting
                        total_cycles <= '0;
                    end
                end
                
                COMPUTE: begin
                    // Perform all computations in a single cycle, and directly update output registers
                    for (int i = 0; i < TILE_SIZE; i++) begin
                        for (int j = 0; j < TILE_SIZE; j++) begin
                            RESULT_T sum_total;
                            sum_total = '0;
                            
                            // Immediately compute the results based on inputs
                            for (int k = 0; k < TILE_SIZE; k++) begin
                                for (int b = 0; b < NUM_BIT_PLANES; b++) begin
                                    ACTIVATION_T act_value = activation_tile[i][k];
                                    WEIGHT_T weight_value = weight_tiles[b][k][j];
                                    
                                    logic is_sparse = ($signed(act_value) <= activation_threshold) && 
                                                      ($signed(act_value) >= -activation_threshold);
                                    
                                    if (weight_value == 1'b1 && !is_sparse) begin
                                        RESULT_T shifted_value;
                                        shifted_value = $signed({{(RESULT_WIDTH-ACT_WIDTH){act_value[ACT_WIDTH-1]}}, act_value}) << b;
                                        sum_total = $signed(sum_total) + $signed(shifted_value);
                                    end
                                end
                            end
                            
                            // Directly update output register (bypass final_results)
                            result_tile[i][j] <= sum_total;
                        end
                    end
                    
                    // Move directly to done state
                    state <= DONE_STATE;
                end
                
                DONE_STATE: begin
                    done <= 1'b1;
                    // Stay in DONE_STATE until next start signal
                    if (start) begin
                        state <= COMPUTE;
                        done <= 1'b0;
                        total_cycles <= '0;
                    end
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule

// ======================
// Top Processing Element
// ======================
module top_processing_element (
    input  logic                                  clk,
    input  logic                                  rst_n,
    input  logic                                  start,
    output logic                                  done,
    
    input  logic signed [7:0]                     activation_threshold,
    
    input  logic [0:0]                            weight_tiles[4][4][4],
    input  logic signed [7:0]                     activation_tile[4][4],
    
    output logic signed [31:0]                    result_tile[4][4]
);

    processing_element #(
        .TILE_SIZE(4),           // 4x4 tiles
        .ACT_WIDTH(8),           // 8-bit activations
        .WEIGHT_WIDTH(1),        // 1-bit weights (per bit plane)
        .NUM_BIT_PLANES(4),      // 4 bit planes (for 4-bit weights)
        .RESULT_WIDTH(32)        // 32-bit results
    ) pe_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .activation_threshold(activation_threshold),
        .weight_tiles(weight_tiles),
        .activation_tile(activation_tile),
        .result_tile(result_tile)
    );

endmodule
