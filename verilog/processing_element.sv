

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
    typedef enum logic [2:0] {
        IDLE,                // Waiting for Start Signal
        GATE_AND_SHIFT,      // Single Cycle for Gating and Shifting Across All Bit Planes
        SUM_ALL,             // Single Cycle to Sum All Contributions for All Output Elements
        STORE_RESULTS,       // Single Cycle to Store Results
        DONE_STATE           // Processing Complete
    } pe_state_t;
    
    pe_state_t state;
    
    // Matrices of Gated and Shifted Contributions for Each Bit Plane
    // Format: [bit_plane][i_row][j_col][k_idx]
    RESULT_T contributions[NUM_BIT_PLANES][TILE_SIZE][TILE_SIZE][TILE_SIZE];
    
    // Input Validity Tracking for Sparse Activation Skipping
    logic [NUM_BIT_PLANES-1:0][TILE_SIZE-1:0][TILE_SIZE-1:0][TILE_SIZE-1:0] contrib_valid;
    
    // Final Output Results
    RESULT_T final_results[TILE_SIZE][TILE_SIZE];
    
    // Phase Trackers
    logic gate_shift_done;
    
    // Counter for Total Cycles
    logic [31:0] total_cycles;
    
    // =========================
    // PHASE 1: Gate and Shift  
    // =========================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gate_shift_done <= 1'b0;
            
            for (int b = 0; b < NUM_BIT_PLANES; b++) begin
                for (int i = 0; i < TILE_SIZE; i++) begin
                    for (int j = 0; j < TILE_SIZE; j++) begin
                        for (int k = 0; k < TILE_SIZE; k++) begin
                            contributions[b][i][j][k] <= '0;
                            contrib_valid[b][i][j][k] <= 1'b0;
                        end
                    end
                end
            end
        end else if (state == GATE_AND_SHIFT) begin

            for (int b = 0; b < NUM_BIT_PLANES; b++) begin
                for (int i = 0; i < TILE_SIZE; i++) begin
                    for (int j = 0; j < TILE_SIZE; j++) begin
                        for (int k = 0; k < TILE_SIZE; k++) begin
                            // Check Activation Sparsity
                            logic is_sparse;
                            ACTIVATION_T act_value;
                            WEIGHT_T weight_value;
                            
                            act_value = activation_tile[i][k];
                            weight_value = weight_tiles[b][k][j];
                            
                            // Check If Activation Should Be Treated as Zero (Sparse)
                            is_sparse = ($signed(act_value) <= activation_threshold) && 
                                        ($signed(act_value) >= -activation_threshold);
                            
                            if (weight_value == 1'b1 && !is_sparse) begin
                                // Valid Contribution: Gate and Shift in One Step
                                RESULT_T shifted_value;
                                shifted_value = $signed({{(RESULT_WIDTH-ACT_WIDTH){act_value[ACT_WIDTH-1]}}, act_value}) << b;
                                
                                contributions[b][i][j][k] <= shifted_value;
                                contrib_valid[b][i][j][k] <= 1'b1;
                                
                                $display("GATE+SHIFT: BP%0d [i=%0d,j=%0d,k=%0d]: act=%0d, shifted=%0d", 
                                         b, i, j, k, act_value, shifted_value);
                            end else begin
                                // No Contribution Due to Weight=0 or Sparse Activation
                                contributions[b][i][j][k] <= '0;
                                contrib_valid[b][i][j][k] <= 1'b0;
                                
                                if (is_sparse) begin
                                    $display("GATE+SHIFT: BP%0d [i=%0d,j=%0d,k=%0d]: sparse activation skipped", 
                                             b, i, j, k);
                                end else if (weight_value == 1'b0) begin
                                    $display("GATE+SHIFT: BP%0d [i=%0d,j=%0d,k=%0d]: weight=0, skipped", 
                                             b, i, j, k);
                                end
                            end
                        end
                    end
                end
            end
            gate_shift_done <= 1'b1;
            $display("GATE+SHIFT: Complete");
        end
    end
    
    // ==================================
    // PHASE 2: Direct Parallel Summation
    // ==================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < TILE_SIZE; i++) begin
                for (int j = 0; j < TILE_SIZE; j++) begin
                    final_results[i][j] <= '0;
                end
            end
        end else if (state == SUM_ALL) begin
            // Fully Parallel Direct Sum for All Output Elements
            for (int i = 0; i < TILE_SIZE; i++) begin
                for (int j = 0; j < TILE_SIZE; j++) begin
                    RESULT_T sum_total;
                    sum_total = '0;
                    
                    // Sum across all k values and bit planes in parallel
                    for (int k = 0; k < TILE_SIZE; k++) begin
                        for (int b = 0; b < NUM_BIT_PLANES; b++) begin
                            if (contrib_valid[b][i][j][k]) begin
                                sum_total = sum_total + contributions[b][i][j][k];
                            end
                        end
                    end
                    
                    final_results[i][j] <= sum_total;
                    $display("DIRECT SUM: result_tile[%0d][%0d] = %0d", i, j, sum_total);
                end
            end
        end
    end

    // ==============
    // State Machine
    // ==============
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            total_cycles <= '0;
            
            // Clear Output Registers1
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
                        state <= GATE_AND_SHIFT;
                        $display("STATE: Starting Processing Element");
                    
                        // Reset Counters and Flags
                        gate_shift_done <= 1'b0;
                        total_cycles <= '0;
                    end
                end
                
                GATE_AND_SHIFT: begin
                    if (gate_shift_done) begin
                        state <= SUM_ALL;
                        $display("STATE: Gate+Shift Complete, Starting Summation");
                    end
                end
                
                SUM_ALL: begin
                    state <= STORE_RESULTS;
                    $display("STATE: Summation Complete");
                end
                
                STORE_RESULTS: begin
                    for (int i = 0; i < TILE_SIZE; i++) begin
                        for (int j = 0; j < TILE_SIZE; j++) begin
                            result_tile[i][j] <= final_results[i][j];
                            $display("STORE: result_tile[%0d][%0d] = %0d", i, j, final_results[i][j]);
                        end
                    end
                    
                    state <= DONE_STATE;
                    $display("STATE: Results Stored, Moving to DONE");
                end
                
                DONE_STATE: begin
                    done <= 1'b1;
                    $display("DONE: Processing Complete in %0d cycles", total_cycles);
                end
                
                default: begin
                    state <= IDLE;
                    $display("STATE: Invalid State Detected, Returning to IDLE");
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
