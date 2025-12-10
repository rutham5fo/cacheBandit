
module mapper_node #(
    parameter OREG_EN               = 0,
    parameter TDM                   = 4
)(
    input wire                      i_clk,
    input wire                      i_rst,
    input wire                      i_en,
    input wire                      i_tdm,          // If set, auto rotate scb regs for tdm; else rotate scb regs only when alloc is performed (i_en)
    input wire                      i_rot_scb,      // SCB from current cycle, post rotation
    input wire                      i_d0,
    input wire                      i_d1,

    output wire                     o_d0,
    output wire                     o_d1,
    output wire                     o_rot_scb,      // SCB from current cycle, out for rotation
    output wire                     o_scb           // SCB from previous cycle, stable for decoder
);

    reg     [TDM-1:0]               r_scb;

    wire                            w_masked;
    wire                            w_d0;
    wire                            w_d1;
    wire                            w_scb;
    wire                            wi_scb;             // SCB to be updated in this cycle
    wire                            wo_scb;             // SCB updated in the previous cycle

    assign wi_scb = r_scb[TDM-1];
    assign wo_scb = r_scb[0];
    assign w_masked = ~i_en | (i_d0 | i_d1);
    assign w_scb = (w_masked & wi_scb) | (~w_masked & i_rot_scb);
    assign w_d0 = (w_scb) ? i_d1 : i_d0;
    assign w_d1 = (w_scb) ? i_d0 : i_d1;

    assign o_rot_scb = wi_scb;
    assign o_scb = wo_scb;

    generate
        if (OREG_EN) begin
            reg                             ro_d0;
            reg                             ro_d1;

            assign o_d0 = ro_d0;
            assign o_d1 = ro_d1;

            always @(posedge i_clk) begin
                ro_d0 <= (i_rst) ? 'b0 : w_d0;
                ro_d1 <= (i_rst) ? 'b0 : w_d1;
            end
        end
        else begin
            assign o_d0 = w_d0;
            assign o_d1 = w_d1;
        end
    endgenerate

    // LSB is freshly assigned and ready for decoder / new val.
    // MSB is subject to change / current val.
    // Hence the TDM rotation order is (MSB) {BLK_N-1, BLK_N-2, ..., BLK_0} (LSB)
    always @(posedge i_clk) begin
        r_scb <= (i_rst) ? 'b0 : (i_en || i_tdm) ? {r_scb[1 +: TDM-1], w_scb} : r_scb;
    end

endmodule