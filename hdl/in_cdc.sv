`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 20.10.2025 11:09:28
// Design Name: 
// Module Name: in_cdc
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


// Assuming PU and CB clk are synchronous, but CB freq = 2*PU
/*
module in_cdc #(
        parameter ADDR_W                    = 32,
        parameter DATA_W                    = 32
    )(
        input wire                          i_pu_clk,
        input wire                          i_cb_clk,
        input wire                          i_rst,
        input wire  [ADDR_W-1:0]            i_field_addr,               // From PU clk
        input wire  [DATA_W-1:0]            i_field_data,               // From PU clk
        
        output wire [ADDR_W-1:0]            o_field_addr,               // From CB clk
        output wire [DATA_W-1:0]            o_field_data                // From CB clk
    );
    
    localparam CDC_RATIO                = PU_CLK_PERIOD / CB_CLK_PERIOD;
    localparam CNTR_W                   = $clog2(CDC_RATIO) + 1;
    
    wire                                w_pu_clk;
    wire                                w_gated_cb_clk;
    
    reg     [ADDR_W-1:0]                r_field_addr;
    reg     [DATA_W-1:0]                r_field_data;
    
    assign w_pu_clk = ~i_pu_clk;                    // Get clock out of clock tree and into datapath. Also invert the clock for enable generation
    
    assign o_field_addr = r_field_addr;
    assign o_field_data = r_field_data;
    
    // Use BUFHCE to gate the cb_clk in accordance with pu_clk to cross domains. Since cb and pu clk are synchronous, CE_TYPE = SYNC.
    BUFHCE #(
        .CE_TYPE("SYNC"), // "SYNC" (glitchless switching) or "ASYNC" (immediate switch)
        .INIT_OUT(0)      // Initial output value (0-1)
    ) BUFHCE_inst (
        .O(w_gated_cb_clk),   // 1-bit output: Clock output
        .CE(w_pu_clk), // 1-bit input: Active high enable
        .I(i_cb_clk)    // 1-bit input: Clock input
    );
    
    always @(posedge w_gated_cb_clk) begin
        r_field_addr <= (i_rst) ? 'b0 : i_field_addr;
        r_field_data <= (i_rst) ? 'b0 : i_field_data;
    end
    
endmodule
*/

// Use Multicycle path to constrain clock paths from pu_clk to cb_clk (ending at r_*_icdc regs)
module in_cdc #(
        parameter PU_CLK_PERIOD             = 10,
        parameter CB_CLK_PERIOD             = 5,
        parameter DATA_W                    = 32,
        parameter VTAG_W                    = 10,
        parameter PTAG_W                    = 10,
        parameter OFFS_W                    = 2
    )(
        input wire                          i_clk,
        input wire                          i_rst,
        input wire  [VTAG_W-1:0]            i_field_table_vtag,
        input wire  [DATA_W-1:0]            i_field_table_wr_data,
        input wire                          i_cb_consume,
        input wire  [PTAG_W-1:0]            i_cb_ctag,
        //input wire  [PTAG_W-1:0]            i_cb_ptag,
        input wire  [OFFS_W-1:0]            i_cb_vtp_offset,
        
        output wire [VTAG_W-1:0]            o_field_table_vtag,
        output wire [DATA_W-1:0]            o_field_table_wr_data,
        output wire                         o_cb_consume,
        output wire [PTAG_W-1:0]            o_cb_ctag,
        output wire [PTAG_W-1:0]            o_cb_lru_ptag,
        output wire [OFFS_W-1:0]            o_cb_vtp_offset
    );
    
    localparam CDC_RATIO                = PU_CLK_PERIOD / CB_CLK_PERIOD;
    localparam CNTR_W                   = $clog2(CDC_RATIO) + 1;
    
    wire                                w_en;
    wire    [PTAG_W-1:0]                wo_cb_lru_ptag;
    
    reg     [CNTR_W-1:0]                r_cntr;
    reg     [VTAG_W-1:0]                r_field_table_vtag;
    reg     [DATA_W-1:0]                r_field_table_wr_data;
    reg                                 r_cb_consume;
    reg     [PTAG_W-1:0]                r_cb_ctag;
    //reg     [PTAG_W-1:0]                r_cb_ptag;
    reg     [OFFS_W-1:0]                r_cb_vtp_offset;
    
    assign w_en = (r_cntr == CDC_RATIO-1) ? 1'b1 : 1'b0;
    assign wo_cb_lru_ptag = r_field_table_wr_data[0 +: PTAG_W];
    
    assign o_field_table_vtag = r_field_table_vtag;
    assign o_field_table_wr_data = r_field_table_wr_data;
    assign o_cb_consume = r_cb_consume;
    assign o_cb_ctag = r_cb_ctag;
    assign o_cb_lru_ptag = wo_cb_lru_ptag;
    assign o_cb_vtp_offset = r_cb_vtp_offset;
    
    always @(posedge i_clk) begin
        r_cntr <= (i_rst || w_en) ? 'b0 : r_cntr+1;
    end
    
    initial begin
        r_field_table_vtag = 'b0;
        r_field_table_wr_data = 'b0;
        r_cb_consume = 'b0;
        r_cb_ctag = 'b0;
        //r_cb_ptag = 'b0;
        r_cb_vtp_offset = 'b0;
    end
    
    always @(posedge i_clk) begin
        r_field_table_vtag <= (w_en) ? i_field_table_vtag : r_field_table_vtag;
        r_field_table_wr_data <= (w_en) ? i_field_table_wr_data : r_field_table_wr_data;
        r_cb_consume <= (w_en) ? i_cb_consume : r_cb_consume;
        r_cb_ctag <= (w_en) ? i_cb_ctag : r_cb_ctag;
        //r_cb_ptag <= (w_en) ? i_cb_ptag : r_cb_ptag;
        r_cb_vtp_offset <= (w_en) ? i_cb_vtp_offset : r_cb_vtp_offset;
    end
    
endmodule
