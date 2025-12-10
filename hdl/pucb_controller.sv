`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 19.10.2025 14:33:25
// Design Name: 
// Module Name: pucb_controller
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

// NOTE: Try to keep PIPE_DEP = 3 and DEC_DEP = 1; which results in a case where cb_lru_sel
//       can be computed using modulo 2 operations.

module pucb_controller #(
        parameter PIPE_DEP          = 3,
        parameter DEC_DEP           = 1,
        localparam OFFS_W           = $clog2(PIPE_DEP),
        localparam DEC_W            = $clog2(DEC_DEP+1)
    )(
        input wire                      i_clk,
        input wire                      i_rst,
        input wire                      i_cb_consume,
        
        output wire [DEC_W-1:0]         o_cb_lru_sel,
        output wire [OFFS_W-1:0]        o_cb_vtp_offset
    );
    
    wire    [DEC_W-1:0]                 w_nxt_cb_lru_sel;
    wire    [OFFS_W-1:0]                w_nxt_cb_vtp_offset;
    wire                                w_dec_stg_tap;
    wire                                w_update_stg_tap;
    
    reg     [DEC_DEP-1:0]               r_cb_consume_dec_stg;
    reg     [PIPE_DEP-2:0]              r_cb_consume_update_stg;
    reg     [DEC_W-1:0]                 r_cb_lru_sel;
    reg     [OFFS_W-1:0]                r_cb_vtp_offset;
    
    assign w_dec_stg_tap = r_cb_consume_dec_stg[DEC_DEP-1];
    assign w_update_stg_tap = r_cb_consume_update_stg[PIPE_DEP-2];
    // Compute next states
    assign w_nxt_cb_lru_sel = r_cb_lru_sel + i_cb_consume - w_dec_stg_tap;
    assign w_nxt_cb_vtp_offset = r_cb_vtp_offset + r_cb_lru_sel + i_cb_consume - w_dec_stg_tap - w_update_stg_tap;
    
    assign o_cb_lru_sel = r_cb_lru_sel;
    assign o_cb_vtp_offset = w_nxt_cb_vtp_offset;           // This signal is registered by CDC barrier
    
    generate
        if (DEC_DEP == 1) begin
            always @(posedge i_clk) begin
                r_cb_consume_dec_stg <= (i_rst) ? 'b0 : i_cb_consume;
            end
        end
        else begin
            always @(posedge i_clk) begin
                r_cb_consume_dec_stg <= (i_rst) ? 'b0 : {r_cb_consume_dec_stg[0 +: DEC_DEP-1], i_cb_consume};
            end
        end
    endgenerate
    
    always @(posedge i_clk) begin
        r_cb_consume_update_stg <= (i_rst) ? 'b0 : {r_cb_consume_update_stg[0 +: PIPE_DEP-2], i_cb_consume};
        r_cb_lru_sel <= (i_rst) ? 'b0 : w_nxt_cb_lru_sel;
        r_cb_vtp_offset <= (i_rst) ? 'b0 : w_nxt_cb_vtp_offset;
    end
    
endmodule
