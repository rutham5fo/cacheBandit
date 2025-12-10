`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 22.10.2025 13:18:52
// Design Name: 
// Module Name: pucb_wb_gen
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


module pucb_wb_gen #(
        parameter DEC_DEP           = 1,
        parameter ASSOC             = 4,
        parameter SETS              = 512,
        parameter DATA_W            = 32,
        parameter PTAG_W            = 10,
        parameter VTAG_W            = 10,
        parameter ATAG_W            = 10,
        localparam ASSOC_W          = $clog2(ASSOC),
        localparam SET_W            = $clog2(SETS),
        localparam DEC_W            = $clog2(DEC_DEP+1)
    )(
        //input wire                          i_field_init,
        input wire                          i_cb_miss,
        input wire                          i_ocb_consume,
        input wire                          i_field_wen,
        input wire  [ATAG_W-1:0]            i_field_atag,
        input wire  [SET_W-1:0]             i_field_set,
        input wire  [PTAG_W-1:0]            i_cb_ctag,
        input wire  [ASSOC_W-1:0]           i_cb_cline,
        input wire  [DEC_DEP:0][PTAG_W-1:0] i_cb_lru,
        input wire  [DEC_W-1:0]             i_cb_lru_sel,
        input wire  [VTAG_W-1:0]            i_ofield_table_vtag,
        //input wire  [DATA_W-1:0]            i_ofield_table_wr_data,
        input wire                          i_cb_rev_ptr_null,
        input wire  [VTAG_W-1:0]            i_cb_rev_ptr,
        
        //output wire [PTAG_W-1:0]            o_cb_lru_ptag,
        //output wire [PTAG_W-1:0]            o_ocb_lru_ptag,
        output wire [VTAG_W-1:0]            o_field_table_vtag,
        output wire [DATA_W-1:0]            o_field_table_wr_data,
        output wire                         o_ovalidity_table_null,
        output wire [VTAG_W-1:0]            o_ovalidity_table_vtag,
        output wire                         o_mem_wen,
        output wire [PTAG_W-1:0]            o_mem_addr
    );
    
    wire    [VTAG_W-1:0]                    w_field_table_vtag;
    wire    [DATA_W-1:0]                    w_field_table_wr_data;
    wire    [PTAG_W-1:0]                    w_cb_lru_ptag;
    //wire    [PTAG_W-1:0]                    wo_cb_lru_ptag;
    wire                                    wo_validity_table_null;
    wire    [VTAG_W-1:0]                    wo_validity_table_vtag;
    wire                                    w_mem_wen;
    wire    [PTAG_W-1:0]                    w_mem_addr;
    
    assign w_cb_lru_ptag = i_cb_lru[i_cb_lru_sel];
    //assign wo_cb_lru_ptag = i_ofield_table_wr_data[0 +: PTAG_W];
    
    assign w_field_table_vtag = {i_field_set, i_cb_cline};
    assign w_field_table_wr_data = {i_field_atag, w_cb_lru_ptag};
    
    //assign wo_validity_table_null = (i_ocb_consume) ? i_cb_rev_ptr_null : i_field_init;
    assign wo_validity_table_null = (i_ocb_consume) ? i_cb_rev_ptr_null : 1'b0;
    assign wo_validity_table_vtag = (i_ocb_consume) ? i_cb_rev_ptr : i_ofield_table_vtag;
    
    //assign w_cb_ptag = (wo_cb_consume) ? wo_cb_lru_ptag : wo_cb_ctag;
    
    assign w_mem_wen = ~i_cb_miss & i_field_wen;
    assign w_mem_addr = i_cb_ctag;
    
    //assign o_cb_lru_ptag = w_cb_lru_ptag;
    //assign o_ocb_lru_ptag = wo_cb_lru_ptag;
    assign o_field_table_vtag = w_field_table_vtag;
    assign o_field_table_wr_data = w_field_table_wr_data;
    assign o_ovalidity_table_null = wo_validity_table_null;
    assign o_ovalidity_table_vtag = wo_validity_table_vtag;
    assign o_mem_wen = w_mem_wen;
    assign o_mem_addr = w_mem_addr;
    
endmodule
