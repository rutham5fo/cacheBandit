
//--------------------------------------------------------------------------
//
// Linear Feedback Shift Register   : to generate Pseudo Random Numbers.
//                                  : Ranging from 1 -> 2^DATA_W-1
//
//--------------------------------------------------------------------------

module lfsr #(
    parameter DATA_W    = 7
)(
    input wire                  i_clk,
    input wire                  i_rst,
	input wire  [DATA_W-1:0]    i_init,
	input wire  [DATA_W-1:0]    i_taps,

	output wire [DATA_W-1:0]    o_state
);

    wire                        w_lsb;
    wire    [DATA_W-1:0]        w_sdrv;
    wire    [DATA_W-1:0]        w_ndrv;

    reg     [DATA_W-1:0]        r_sreg;

    genvar i;

    assign w_lsb = r_sreg[0];
    assign w_sdrv = r_sreg >> 1;
    assign w_ndrv = w_sdrv ^ i_taps;
    
    assign o_state = r_sreg;

    always @(posedge i_clk) begin
        if (i_rst) r_sreg <= i_init;
        else r_sreg <= (w_lsb) ? w_ndrv : w_sdrv;
    end
    
endmodule