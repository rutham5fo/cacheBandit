
//--------------------------------------------------------------------------
//
// Random Mask Generator    : to generate Pseudo Random mask for Compressor.
//
//--------------------------------------------------------------------------

module rand_mask_gen #(
    parameter BLOCKS            = 4,                    // Determines TDM channels (cacheBandits)
    parameter BLOCK_W           = 128,                  // Size of a single cacheBandit block
    parameter RAND_TAPS         = 'h60,                 // Set based of LFSR length
    parameter RAND_INIT         = 'h0a                  // Randomly chosen constant
)(
    input wire                  i_clk,
    input wire                  i_rst,

	output wire [BLOCK_W-1:0]   o_mask
);

    localparam MASK_W               = BLOCK_W;
    localparam DATA_W               = $clog2(BLOCK_W);
    localparam LIM_OFFSET           = BLOCKS+1;             // Start filling compressor pipeline when free_ptr hits LIM_OFFSET
    localparam STAGES               = $clog2(BLOCK_W);

    wire    [DATA_W-1:0]            w_rand;
    wire    [DATA_W-1:0]            w_shift;
    wire    [DATA_W-1:0]            w_stg_shift[0:STAGES-1];
    wire    [MASK_W-1:0]            w_stg_mask[0:STAGES-1];

    genvar i;

    assign o_mask = w_stg_mask[STAGES-1];

    lfsr #(
        .DATA_W(DATA_W)
    ) lfsr_i (
        .i_clk(i_clk),
        .i_rst(i_rst),
    	.i_init(RAND_INIT),
    	.i_taps(RAND_TAPS),

    	.o_state(w_rand)
    );

    limiter #(
        .OREG_EN(1),
        .OFFSET(LIM_OFFSET),
        .DATA_W(DATA_W)
    ) limiter_i (
        .i_clk(i_clk),
        .i_rst(i_rst),
    	.i_random(w_rand),

    	.o_shift(w_shift)
    );

    generate
        for (i = 0; i < STAGES; i = i+1) begin          :   gen_mask_shifter
            if (i == 0) begin
                mask_stage #(
                    .OREG_EN(1),
                    .STAGE(i),
                    .DATA_W(DATA_W),
                    .MASK_W(MASK_W)
                ) mask_stg_i (
                    .i_clk(i_clk),
                    .i_rst(i_rst),
                	.i_shift(w_shift),
                    .i_mask({MASK_W{1'b0}}),

                	.o_shift(w_stg_shift[i]),
                    .o_mask(w_stg_mask[i])
                );
            end
            else begin
                mask_stage #(
                    .OREG_EN(1),
                    .STAGE(i),
                    .DATA_W(DATA_W),
                    .MASK_W(MASK_W)
                ) mask_stg_i (
                    .i_clk(i_clk),
                    .i_rst(i_rst),
                	.i_shift(w_stg_shift[i-1]),
                    .i_mask(w_stg_mask[i-1]),

                	.o_shift(w_stg_shift[i]),
                    .o_mask(w_stg_mask[i])
                );
            end
        end
    endgenerate
    
endmodule