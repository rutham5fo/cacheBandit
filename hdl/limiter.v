
//--------------------------------------------------------------------------
//
// Limiter  : Limit PRNG values within range 0 -> 2^DATA_W - OFFSET.
//          : OFFSET is dependent on Clocking frequency / TDM.
//          : Generated value used to shift right a bunch of 1s.
//
//--------------------------------------------------------------------------

module limiter #(
    parameter OREG_EN       = 1,
    parameter OFFSET        = 8,
    parameter DATA_W        = 7
)(
    input wire                  i_clk,
    input wire                  i_rst,
	input wire  [DATA_W-1:0]    i_random,

	output wire [DATA_W-1:0]    o_shift
);

    localparam                  LIMIT_VAL       = 2**DATA_W - OFFSET;
    localparam  [DATA_W-1:0]    MAX_VAL         = LIMIT_VAL[0 +: DATA_W];

    generate
        if (OREG_EN) begin
            reg     [DATA_W-1:0]        r_shift;

            assign o_shift = r_shift;

            always @(posedge i_clk) begin
                r_shift <= (i_rst || i_random >= MAX_VAL) ? 'b0 : i_random;
            end
        end
        else begin
            assign o_shift = (i_random >= MAX_VAL) ? 'b0 : i_random;
        end
    endgenerate
    
endmodule