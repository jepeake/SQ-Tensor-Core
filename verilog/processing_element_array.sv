

//  ███████████  ██████████                               
// ░░███░░░░░███░░███░░░░░█                               
//  ░███    ░███ ░███  █ ░                                
//  ░██████████  ░██████                                  
//  ░███░░░░░░   ░███░░█                                  
//  ░███         ░███ ░   █                               
//  █████        ██████████                               
// ░░░░░        ░░░░░░░░░░                                
                                                       
                                                                                                  
//    █████████                                           
//   ███░░░░░███                                          
//  ░███    ░███  ████████  ████████   ██████   █████ ████
//  ░███████████ ░░███░░███░░███░░███ ░░░░░███ ░░███ ░███ 
//  ░███░░░░░███  ░███ ░░░  ░███ ░░░   ███████  ░███ ░███ 
//  ░███    ░███  ░███      ░███      ███░░███  ░███ ░███ 
//  █████   █████ █████     █████    ░░████████ ░░███████ 
// ░░░░░   ░░░░░ ░░░░░     ░░░░░      ░░░░░░░░   ░░░░░███ 
//                                               ███ ░███ 
//                                              ░░██████  
//                                               ░░░░░░                                             


module processing_element_array #(
    parameter int GRID_HEIGHT      = 2,              // Number of PEs in Grid Height
    parameter int GRID_WIDTH       = 2,              // Number of PEs in Grid Width
    parameter int TILE_SIZE        = 2,              // Size of Input Tiles (NxN)
    parameter int ACT_WIDTH        = 8,              // Width of Activation Values (int8_t)
    parameter int WEIGHT_WIDTH     = 1,              // Width of Weight Values (Bits)
    parameter int NUM_BIT_PLANES   = 4,              // Number of Weight Bit Planes
    parameter int RESULT_WIDTH     = 16,             // Width of Output Values (int16_t)
    
    parameter type ACTIVATION_T    = logic signed [ACT_WIDTH-1:0],
    parameter type WEIGHT_T        = logic [WEIGHT_WIDTH-1:0],
    parameter type RESULT_T        = logic signed [RESULT_WIDTH-1:0]
)(
    // Global Control Signals
    input  logic                                       clk,
    input  logic                                       rst_n,
    
    // Control Signals for Each PE
    input  logic [GRID_HEIGHT-1:0][GRID_WIDTH-1:0]    pe_start,
    output logic [GRID_HEIGHT-1:0][GRID_WIDTH-1:0]    pe_done,
    
    // Broadcast Control
    input  logic                                      broadcast_mode,     // Enable Broadcast Mode
    input  logic [1:0]                                broadcast_row,      // Row to Broadcast Weights to
    input  logic [1:0]                                broadcast_col,      // Column to Broadcast Activations to
    input  logic                                      broadcast_load,     // Load Broadcast Data
    
    // Configuration for all PEs
    input  ACTIVATION_T                                pe_activation_threshold,
    
    // Input Data
    input  logic [NUM_BIT_PLANES-1:0][TILE_SIZE-1:0][TILE_SIZE-1:0][WEIGHT_WIDTH-1:0] broadcast_weight_tile,
    input  logic [TILE_SIZE-1:0][TILE_SIZE-1:0][ACT_WIDTH-1:0]                      broadcast_activation_tile,
    
    // For Compatibility with Top Module
    input  WEIGHT_T    [GRID_HEIGHT-1:0][GRID_WIDTH-1:0][NUM_BIT_PLANES-1:0][TILE_SIZE-1:0][TILE_SIZE-1:0] pe_weight_tiles,
    input  ACTIVATION_T [GRID_HEIGHT-1:0][GRID_WIDTH-1:0][TILE_SIZE-1:0][TILE_SIZE-1:0] pe_activation_tile,
    
    // Output Data
    output RESULT_T    [GRID_HEIGHT-1:0][GRID_WIDTH-1:0][TILE_SIZE-1:0][TILE_SIZE-1:0] pe_result_tile,
    
    output logic                                       all_pes_done
);

    // Store Weights and Activations for Each PE
    WEIGHT_T    stored_weight_tiles[GRID_HEIGHT-1:0][GRID_WIDTH-1:0][NUM_BIT_PLANES-1:0][TILE_SIZE-1:0][TILE_SIZE-1:0];
    ACTIVATION_T stored_activation_tiles[GRID_HEIGHT-1:0][GRID_WIDTH-1:0][TILE_SIZE-1:0][TILE_SIZE-1:0];

    // Handle Broadcast Mechanism to Configure PEs
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset All Storage
            for (int row = 0; row < GRID_HEIGHT; row++) begin
                for (int col = 0; col < GRID_WIDTH; col++) begin
                    for (int bp = 0; bp < NUM_BIT_PLANES; bp++) begin
                        for (int i = 0; i < TILE_SIZE; i++) begin
                            for (int j = 0; j < TILE_SIZE; j++) begin
                                stored_weight_tiles[row][col][bp][i][j] <= '0;
                            end
                        end
                    end
                    
                    for (int i = 0; i < TILE_SIZE; i++) begin
                        for (int j = 0; j < TILE_SIZE; j++) begin
                            stored_activation_tiles[row][col][i][j] <= '0;
                        end
                    end
                end
            end
        end 
        else begin
            // First Copy Data from PE Weight and Activation Tiles into Storage
            for (int row = 0; row < GRID_HEIGHT; row++) begin
                for (int col = 0; col < GRID_WIDTH; col++) begin
                    for (int bp = 0; bp < NUM_BIT_PLANES; bp++) begin
                        for (int i = 0; i < TILE_SIZE; i++) begin
                            for (int j = 0; j < TILE_SIZE; j++) begin
                                stored_weight_tiles[row][col][bp][i][j] <= pe_weight_tiles[row][col][bp][i][j];
                            end
                        end
                    end
                    
                    for (int i = 0; i < TILE_SIZE; i++) begin
                        for (int j = 0; j < TILE_SIZE; j++) begin
                            stored_activation_tiles[row][col][i][j] <= pe_activation_tile[row][col][i][j];
                        end
                    end
                end
            end
            
            // Update Broadcast Tiles if Broadcast Mode and Broadcast Load are Active
            if (broadcast_mode && broadcast_load) begin
                for (int row = 0; row < GRID_HEIGHT; row++) begin
                    for (int col = 0; col < GRID_WIDTH; col++) begin
                        // Broadcast Weight Tile to All PEs in the Specified Row
                        if (32'(row) == 32'($unsigned(broadcast_row))) begin
                            for (int bp = 0; bp < NUM_BIT_PLANES; bp++) begin
                                for (int i = 0; i < TILE_SIZE; i++) begin
                                    for (int j = 0; j < TILE_SIZE; j++) begin
                                        stored_weight_tiles[row][col][bp][i][j] <= broadcast_weight_tile[bp][i][j][0];
                                    end
                                end
                            end
                        end
                        
                        // Broadcast Activation Tile to All PEs in the Specified Column
                        if (32'(col) == 32'($unsigned(broadcast_col))) begin
                            for (int i = 0; i < TILE_SIZE; i++) begin
                                for (int j = 0; j < TILE_SIZE; j++) begin
                                    stored_activation_tiles[row][col][i][j] <= $signed(broadcast_activation_tile[i][j]);
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    // ================================
    // Processing Element Instantiation
    // ================================
    
    // Create a 2D Grid of Processing Elements
    genvar row, col;
    generate
        for (row = 0; row < GRID_HEIGHT; row++) begin : pe_row
            for (col = 0; col < GRID_WIDTH; col++) begin : pe_col
                // Create Local Variables to Hold Arrays for This PE
                WEIGHT_T    pe_local_weights[NUM_BIT_PLANES][TILE_SIZE][TILE_SIZE];
                ACTIVATION_T pe_local_activations[TILE_SIZE][TILE_SIZE];
                RESULT_T     pe_local_results[TILE_SIZE][TILE_SIZE];
                
                // Connect the Arrays
                always_comb begin
                    // Copy Data from the Stored Arrays to Local Arrays
                    for (int b = 0; b < NUM_BIT_PLANES; b++) begin
                        for (int i = 0; i < TILE_SIZE; i++) begin
                            for (int j = 0; j < TILE_SIZE; j++) begin
                                pe_local_weights[b][i][j] = stored_weight_tiles[row][col][b][i][j];
                            end
                        end
                    end
                    
                    for (int i = 0; i < TILE_SIZE; i++) begin
                        for (int j = 0; j < TILE_SIZE; j++) begin
                            pe_local_activations[i][j] = stored_activation_tiles[row][col][i][j];
                            pe_result_tile[row][col][i][j] = pe_local_results[i][j];
                        end
                    end
                end
                
                // Instantiate Each Processing Element with Direct Connections
                processing_element #(
                    .TILE_SIZE(TILE_SIZE),
                    .ACT_WIDTH(ACT_WIDTH),
                    .WEIGHT_WIDTH(WEIGHT_WIDTH),
                    .NUM_BIT_PLANES(NUM_BIT_PLANES),
                    .RESULT_WIDTH(RESULT_WIDTH),
                    .ACTIVATION_T(ACTIVATION_T),
                    .WEIGHT_T(WEIGHT_T),
                    .RESULT_T(RESULT_T)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .start(pe_start[row][col]),
                    .done(pe_done[row][col]),
                    .activation_threshold(pe_activation_threshold),
                    .weight_tiles(pe_local_weights),
                    .activation_tile(pe_local_activations),
                    .result_tile(pe_local_results)
                );
            end
        end
    endgenerate
    
        // Compute all_pes_done Signal - Logical AND of All Individual Done Signals
    always_comb begin
        all_pes_done = 1'b1;
        for (int i = 0; i < GRID_HEIGHT; i++) begin
            for (int j = 0; j < GRID_WIDTH; j++) begin
                all_pes_done = all_pes_done & pe_done[i][j];
            end
        end
    end

endmodule

// ==============================
// Top Processing Element Array
// ==============================

module top_processing_element_array (
    input  logic                                  clk,
    input  logic                                  rst_n,
    
    input  logic [1:0]                            pe_row_select,   // For a 4x4 Grid
    input  logic [1:0]                            pe_col_select,
    
    input  logic                                  global_start,    // Start all PEs at once
    
    input  logic                                  pe_start,
    output logic                                  pe_done,
    
    input  logic signed [7:0]                     pe_activation_threshold,
    
    // New Broadcast Inputs with Buffers for All Rows/Columns
    input  logic                                  broadcast_enable,      // Enable Broadcast Mode
    input  logic                                  broadcast_weights,     // Broadcast Weights to Selected Row
    input  logic                                  broadcast_activations, // Broadcast Activations to Selected Column
    input  logic [1:0]                            broadcast_row_select,  // Select Which Row to Broadcast to
    input  logic [1:0]                            broadcast_col_select,  // Select Which Column to Broadcast to
    input  logic                                  broadcast_all,         // Broadcast to All Rows/Columns at Once
    
    // Buffer for Storing Weights for Each Row (Left-Side Buffer)
    input  logic [0:0]                            row_weight_buffer[4][4][4][4], // [row][bp][i][j]
    
    // Buffer for Storing Activations for Each Column (Top Buffer)
    input  logic signed [7:0]                     col_activation_buffer[4][4][4], // [col][i][j]
    
    // Output Results
    output logic signed [31:0]                    pe_result_tile[4][4],
    output logic [3:0][3:0][3:0][31:0]            output_buffer,         
    
    output logic                                  all_pes_done
);

    // Create Internal Arrays for PE Start/Done Signals
    logic [3:0][3:0] internal_pe_start;
    logic [3:0][3:0] internal_pe_done;
    
    // Use Packed Arrays for PE Module Connections
    logic [3:0][3:0][3:0][3:0][3:0][0:0] packed_weight_tiles;
    logic [3:0][3:0][3:0][3:0][7:0] packed_activation_tile;
    logic [3:0][3:0][3:0][3:0][31:0] packed_result_tile;
    
    // Broadcast Parameters for the PE Array
    logic internal_broadcast_mode;
    logic [1:0] internal_broadcast_row;
    logic [1:0] internal_broadcast_col;
    logic internal_broadcast_load;
    logic [3:0][3:0][3:0][0:0] packed_broadcast_weight;
    logic [3:0][3:0][7:0] packed_broadcast_activation;
    
    // Map the PE Indices and Data to the Packed Arrays
    always_comb begin
        // Default: Set All PE Start Signals to 0
        internal_pe_start = '{default: '{default: 1'b0}};
        
        // Default Initialization for Broadcast Signals to Prevent Latches
        packed_broadcast_weight = '{default: '{default: '{default: '{default: 1'b0}}}};
        packed_broadcast_activation = '{default: '{default: '{default: 8'b0}}};
        output_buffer = '{default: '{default: '{default: '{default: 32'b0}}}};
        
        // Handle Global Start (Start All PEs) or Individual PE Start
        if (global_start) begin
            // Start All PEs When Global Start is Active
            internal_pe_start = '{default: '{default: 1'b1}};
        end else begin
            // Start Only the Selected PE When pe_start is Active
            internal_pe_start[pe_row_select][pe_col_select] = pe_start;
        end
        
        // Select the Done Signal from the Selected PE
        pe_done = internal_pe_done[pe_row_select][pe_col_select];
        
        // Map the Selected PE's Result to the Output
        for (int ti = 0; ti < 4; ti++) begin
            for (int tj = 0; tj < 4; tj++) begin
                pe_result_tile[ti][tj] = packed_result_tile[pe_row_select][pe_col_select][ti][tj];
            end
        end
        
        // Set Broadcast Parameters
        internal_broadcast_mode = broadcast_enable;
        internal_broadcast_row = broadcast_row_select;
        internal_broadcast_col = broadcast_col_select;
        internal_broadcast_load = broadcast_weights || broadcast_activations || broadcast_all;
        
        // Map Buffer to Broadcast Inputs - for broadcast_weights mode
        if (broadcast_weights) begin
            for (int i = 0; i < 4; i++) begin
                for (int j = 0; j < 4; j++) begin
                    for (int bp = 0; bp < 4; bp++) begin
                        packed_broadcast_weight[bp][i][j][0] = row_weight_buffer[broadcast_row_select][bp][i][j];
                    end
                end
            end
        end
        
        // Map Buffer to Broadcast Inputs - for broadcast_activations mode
        if (broadcast_activations) begin
            for (int i = 0; i < 4; i++) begin
                for (int j = 0; j < 4; j++) begin
                    packed_broadcast_activation[i][j] = col_activation_buffer[broadcast_col_select][i][j];
                end
            end
        end
        
        // Map Buffer to Broadcast Inputs - for broadcast_all mode
        if (broadcast_all) begin
            for (int row = 0; row < 4; row++) begin
                for (int i = 0; i < 4; i++) begin
                    for (int j = 0; j < 4; j++) begin
                        for (int bp = 0; bp < 4; bp++) begin
                            // Select Proper Row's Weight for Broadcast
                            if (internal_broadcast_row == row) begin
                                packed_broadcast_weight[bp][i][j][0] = row_weight_buffer[row][bp][i][j];
                            end
                        end
                    end
                end
            end
            
            for (int col = 0; col < 4; col++) begin
                for (int i = 0; i < 4; i++) begin
                    for (int j = 0; j < 4; j++) begin
                        // Select Proper Column's Activation for Broadcast
                        if (internal_broadcast_col == col) begin
                            packed_broadcast_activation[i][j] = col_activation_buffer[col][i][j];
                        end
                    end
                end
            end
        end
        
        // Map Results to Output Buffer
        for (int row = 0; row < 4; row++) begin
            for (int col = 0; col < 4; col++) begin
                for (int i = 0; i < 4; i++) begin
                    for (int j = 0; j < 4; j++) begin
                        output_buffer[row][col][i][j] = packed_result_tile[row][col][i][j];
                    end
                end
            end
        end
    end
    
    // Data Loading Using Only Broadcast Functionality
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize the Packed Arrays to 0
            for (int row = 0; row < 4; row++) begin
                for (int col = 0; col < 4; col++) begin
                    for (int bp = 0; bp < 4; bp++) begin
                        for (int i = 0; i < 4; i++) begin
                            for (int j = 0; j < 4; j++) begin
                                packed_weight_tiles[row][col][bp][i][j][0] <= 1'b0;
                            end
                        end
                    end
                    
                    for (int i = 0; i < 4; i++) begin
                        for (int j = 0; j < 4; j++) begin
                            packed_activation_tile[row][col][i][j] <= 8'b0;
                        end
                    end
                end
            end
        end else begin
            // Only Support Broadcast Mode Configuration
            if (broadcast_enable) begin
                // Handle All-at-Once Broadcast Mode
                if (broadcast_all) begin
                    // Broadcast Weights to All Rows and Activations to All Columns Simultaneously
                    for (int row = 0; row < 4; row++) begin
                        for (int col = 0; col < 4; col++) begin
                            // Load Weights from Each Row's Buffer
                            for (int bp = 0; bp < 4; bp++) begin
                                for (int i = 0; i < 4; i++) begin
                                    for (int j = 0; j < 4; j++) begin
                                        packed_weight_tiles[row][col][bp][i][j][0] <= row_weight_buffer[row][bp][i][j];
                                    end
                                end
                            end
                            
                            // Load Activations from Each Column's Buffer
                            for (int i = 0; i < 4; i++) begin
                                for (int j = 0; j < 4; j++) begin
                                    packed_activation_tile[row][col][i][j] <= col_activation_buffer[col][i][j];
                                end
                            end
                        end
                    end
                end else begin
                    // Broadcast Mode - Handle Row-Wise Weight Broadcasting
                    if (broadcast_weights) begin
                        // Broadcast to All PEs in the Selected Row
                        for (int col = 0; col < 4; col++) begin
                            for (int bp = 0; bp < 4; bp++) begin
                                for (int i = 0; i < 4; i++) begin
                                    for (int j = 0; j < 4; j++) begin
                                        packed_weight_tiles[broadcast_row_select][col][bp][i][j][0] <= row_weight_buffer[broadcast_row_select][bp][i][j];
                                    end
                                end
                            end
                        end
                    end
                    
                    // Handle Column-Wise Activation Broadcasting
                    if (broadcast_activations) begin
                        // Broadcast to All PEs in the Selected Column
                        for (int row = 0; row < 4; row++) begin
                            for (int i = 0; i < 4; i++) begin
                                for (int j = 0; j < 4; j++) begin
                                    packed_activation_tile[row][broadcast_col_select][i][j] <= col_activation_buffer[broadcast_col_select][i][j];
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    // Instantiate the PE Array
    processing_element_array #(
        .GRID_HEIGHT(4),
        .GRID_WIDTH(4),
        .TILE_SIZE(4),
        .ACT_WIDTH(8),
        .WEIGHT_WIDTH(1),
        .NUM_BIT_PLANES(4),
        .RESULT_WIDTH(32)
    ) pe_array_inst (
        .clk(clk),
        .rst_n(rst_n),
        .pe_start(internal_pe_start),
        .pe_done(internal_pe_done),
        
        // Broadcast Parameters
        .broadcast_mode(internal_broadcast_mode),
        .broadcast_row(internal_broadcast_row),
        .broadcast_col(internal_broadcast_col),
        .broadcast_load(internal_broadcast_load),
        .broadcast_weight_tile(packed_broadcast_weight),
        .broadcast_activation_tile(packed_broadcast_activation),
        
        .pe_activation_threshold(pe_activation_threshold),
        .pe_weight_tiles(packed_weight_tiles),
        .pe_activation_tile(packed_activation_tile),
        .pe_result_tile(packed_result_tile),
        .all_pes_done(all_pes_done)
    );

endmodule 
