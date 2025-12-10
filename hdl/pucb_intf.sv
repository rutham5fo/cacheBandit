`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 16.10.2025 16:03:54
// Design Name: 
// Module Name: pucb_intf
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


//----------------------------------------------------------------------
//  NOTES
//
//  * CB will push the missed tag into MSHR for handling. This buys a single 
//    cycle bubble, which is used by CB to set the newly allocated valid bit.
//
//  * PU hits can be accessed with single cycle latency, but misses must be handled by MSHR bubble.
//
// -----------------------------------------------------------------------



module pucb_intf #(
        parameter PU_CLK_PERIOD     = 10,
        parameter CB_CLK_PERIOD     = 5,
        //parameter BRAM_EN           = 1,
        parameter ASSOC             = 4,
        parameter SETS              = 512,
        parameter ADDR_W            = 32,
        parameter DATA_W            = 32,
        parameter PAGE_SIZE         = 256,
        parameter PIPE_DEP          = 3,                    // Number of stages from pu_addr assertion (including pucb_intf) to CB scb update stage
        parameter DEC_DEP           = 1,                    // Number of CB vtp decoder stages
        localparam TABLE_W          = SETS*ASSOC,
        localparam ASSOC_W          = $clog2(ASSOC),
        localparam SET_W            = $clog2(SETS),
        localparam PTAG_W           = $clog2(PAGE_SIZE),
        localparam VTAG_W           = $clog2(TABLE_W),
        localparam FULL_ATAG_W      = ADDR_W - SET_W,
        localparam VALID_ATAG_W     = DATA_W - PTAG_W,
        localparam ATAG_W           = (FULL_ATAG_W < VALID_ATAG_W) ? FULL_ATAG_W : VALID_ATAG_W,
        localparam OFFS_W           = $clog2(PIPE_DEP),
        localparam DEC_W            = $clog2(DEC_DEP+1)
    )(
        input wire                          i_pu_clk,
        input wire                          i_cb_clk,
        input wire                          i_rst,
        // From PU
        input wire                          i_cb_en,
        input wire                          i_field_mshr,           // True when i_field_daddr is from MSHR | When this is true, CB will only look for a tag match and will ignore Pvld for comparison, and set it when done.
        input wire                          i_field_wen,            // PU data write enable
        input wire  [ADDR_W-1:0]            i_field_addr,           // PU address {atag, set_sel, ASSOC_W ignore}
        // From CB
        input wire  [DEC_DEP:0][PTAG_W-1:0] i_cb_consume_buf,
        
        // To PU
        output wire                         o_field_mshr,           // True when miss occured at o_field_addr and needs to be handled
        output wire [ADDR_W-1:0]            o_field_addr,           // Address that needs to be pushed into mshr
        output wire                         o_field_wen,
        // To CB
        output wire                         o_cb_consume,           // Indicates consumption from CB_LRU, triggering full-rotation of map.
        output wire [PTAG_W-1:0]            o_cb_ptag,              // PTAG at which hit occured (CB performs LRU on this IFF not in deadzone)
        output wire [OFFS_W-1:0]            o_cb_vtp_offset,
        // To Heap
        output wire                         o_mem_wen,
        output wire [PTAG_W-1:0]            o_mem_addr
    );
    
    localparam LRU_NODES                    = ASSOC-1;
    
    //wire                                    w_field_init;
    wire                                    w_field_stall;
    wire                                    w_field_wen;
    wire    [ADDR_W-1:0]                    w_field_addr;
    wire                                    w_field_mshr;
    wire    [ATAG_W-1:0]                    w_field_atag;
    wire    [SET_W-1:0]                     w_field_set;
    
    wire    [VTAG_W-1:0]                    w_field_table_vtag;         // Register with CDC
    wire    [DATA_W-1:0]                    w_field_table_wr_data;      // Register with CDC
    wire    [VTAG_W-1:0]                    wo_field_table_vtag;
    wire    [DATA_W-1:0]                    wo_field_table_wr_data;
    
    wire    [VTAG_W-1:0]                    wo_validity_table_vtag;     // Extracted from Registered CDC
    wire                                    wo_validity_table_null;     // Extracted from Registered CDC
    
    wire    [ASSOC-1:0]                     w_comp_hit;
    wire    [ASSOC-1:0][PTAG_W-1:0]         w_comp_ptag;
    wire    [PTAG_W-1:0]                    w_comp_lru_ptag;
    wire    [ASSOC-1:0]                     w_comp_pvld;
    
    wire    [ASSOC_W-1:0]                   w_cb_lru_cur;
    wire    [LRU_NODES-1:0]                 w_cb_lru_cur_bits;
    wire    [ASSOC_W-1:0]                   w_cb_lru_nxt;
    wire    [LRU_NODES-1:0]                 w_cb_lru_nxt_bits;
    
    wire                                    w_cb_miss;
    wire    [DEC_DEP:0][PTAG_W-1:0]         w_cb_consume_buf;
    wire    [VTAG_W-1:0]                    w_cb_rev_ptr;
    wire                                    w_cb_rev_ptr_null;
    wire                                    w_cb_consume;               // Register with CDC | True when a new block is taken from CB
    wire                                    wo_cb_consume;
    wire    [ASSOC_W-1:0]                   w_cb_cline;                 // Assoc line at which comparator landed
    wire    [PTAG_W-1:0]                    w_cb_ctag;                  // Register with CDC | Associated ptag at cline;
    wire    [PTAG_W-1:0]                    wo_cb_ctag;
    
    wire    [OFFS_W-1:0]                    w_cb_vtp_offset;            // Register with CDC | Driven by controller
    wire    [OFFS_W-1:0]                    wo_cb_vtp_offset;
    wire    [DEC_W-1:0]                     w_cb_consume_sel;               // Driven by controller
    wire    [PTAG_W-1:0]                    wo_cb_lru_ptag;             // Extract from Registered CDC
    
    wire                                    w_mem_wen;
    wire    [PTAG_W-1:0]                    w_mem_addr;
    
    reg                                     r_field_mshr;
    reg     [ADDR_W-1:0]                    r_field_addr;
    reg                                     r_field_wen;
    //reg                                     r_field_init;
    
    //assign w_field_init = r_field_init;
    assign w_field_stall = i_rst | ~i_cb_en;
    assign w_field_wen = i_field_wen;
    assign w_field_addr = i_field_addr;
    assign w_field_mshr = i_field_mshr;
    assign w_field_set = w_field_addr[0 +: SET_W];
    assign w_field_atag = w_field_addr[SET_W +: ATAG_W];
    assign w_cb_consume_buf = i_cb_consume_buf;
    assign w_cb_miss = ~(|(w_comp_pvld & w_comp_hit));
    
    assign o_field_mshr = r_field_mshr;
    assign o_field_addr = r_field_addr;
    assign o_field_wen = r_field_wen;
    
    assign o_cb_consume = wo_cb_consume;
    assign o_cb_ptag = wo_cb_ctag;
    assign o_cb_vtp_offset = wo_cb_vtp_offset;
    
    assign o_mem_wen = w_mem_wen;
    assign o_mem_addr = w_mem_addr;
    
    pucb_comparator #(
        .ASSOC(ASSOC),
        //.ATAG_W(ATAG_W),
        .PTAG_W(PTAG_W)
    ) comparator_i (
        .i_field_ptag(w_comp_ptag),
        .i_field_hit(w_comp_hit),
        .i_field_pvld(w_comp_pvld),
        .i_field_lru(w_cb_lru_cur),
        .i_field_lru_ptag(w_comp_lru_ptag),
        
        .o_cb_line(w_cb_cline),                 // Line pointed by comparator | in-case of a miss, points to lru line, else to hit line
        .o_cb_ptag(w_cb_ctag)                      // Ptag with associated o_tag_line
    );
    
    pucb_lru #(
        //.BRAM_EN(BRAM_EN),
        .ASSOC(ASSOC)
        //.SETS(SETS)
    ) pucb_lru_i (
        //.i_clk(i_pu_clk),
        //.i_rst(i_rst),
        .i_field_lru_cur_bits(w_cb_lru_cur_bits),
        
        .o_field_lru_nxt_bits(w_cb_lru_nxt_bits),
        .o_field_lru_nxt(w_cb_lru_nxt)
    );
    
    // Tables
    field_table #(
        //.BRAM_EN(BRAM_EN),
        .ASSOC(ASSOC),
        .SETS(SETS),
        .DATA_W(DATA_W),
        .ATAG_W(ATAG_W),
        .PTAG_W(PTAG_W)
    ) field_table_i (
        .i_clk(i_pu_clk),
        .i_rst(i_rst),
        .i_field_atag(w_field_atag),
        .i_field_lru(w_cb_lru_cur),
        .i_rd_addr(w_field_set),
        .i_wr_addr(wo_field_table_vtag),
        .i_wr_data(wo_field_table_wr_data),
        .i_wr_en(wo_cb_consume),
        
        //.o_comp_miss(w_cb_miss),
        .o_comp_hit(w_comp_hit),
        .o_comp_lru_ptag(w_comp_lru_ptag),
        .o_comp_ptag(w_comp_ptag)
    );
    
    validity_table #(
        //.BRAM_EN(BRAM_EN),
        //.BLOCK_W(PAGE_SIZE),
        .ASSOC(ASSOC),
        .SETS(SETS)
    ) validity_table_i (
        .i_clk(i_pu_clk),
        .i_rst(i_rst),
        //.i_cb_en(i_cb_en),
        //.i_field_init(w_field_init),
        .i_field_stall(w_field_stall),
        .i_field_mshr(w_field_mshr),
        .i_field_lru(w_cb_lru_cur),
        .i_field_miss(w_cb_miss),
        .i_rd_addr(w_field_set),                  // From current cycle
        .i_wr_addr(wo_validity_table_vtag),                  // From previous cycle
        .i_wr_null(wo_validity_table_null),                  // When true, write is dropped, i.e., no side-effects
        .i_field_consume(wo_cb_consume),            // when consume is true, miss must be true. But, not vice-versa. From previous cycle
        
        .o_field_consume(w_cb_consume),
        .o_field_pvld(w_comp_pvld)                // Output in current cycle
    );
    
    pucb_lru_regfile #(
        //.BRAM_EN(BRAM_EN),
        .ASSOC(ASSOC),
        .SETS(SETS)
    ) lru_regfile_i (
        .i_clk(i_pu_clk),
        .i_rst(i_rst),
        .i_rd_addr(w_field_set),
        .i_wr_addr(w_field_set),
        .i_field_lru_nxt_bits(w_cb_lru_nxt_bits),
        .i_field_lru_nxt(w_cb_lru_nxt),
        .i_field_lru_wen(w_cb_miss),
        .i_field_stall(w_field_stall),

        .o_field_lru_cur_bits(w_cb_lru_cur_bits),
        .o_field_lru_cur(w_cb_lru_cur)
    );
    
    // Must be Read-Before-Write mode, since read and write to same address in the same cycle is possible
    reverse_ptr_table #(
        //.BRAM_EN(BRAM_EN),
        .BLOCK_W(PAGE_SIZE),
        .ASSOC(ASSOC),
        .SETS(SETS)
    ) rev_ptr_table_i (
        .i_clk(i_pu_clk),
        .i_rst(i_rst),
        .i_rd_addr(wo_cb_lru_ptag),                  // From current cycle
        //.i_wr_addr(wo_cb_ctag),                  // From previous cycle
        .i_wr_addr(wo_cb_lru_ptag),                  // From previous cycle
        .i_wr_data(wo_field_table_vtag),                  // From previous cycle
        .i_field_evict(1'b0),              // When true, value at wr_addr is forced to point to null | This is not the delayed signal of o_rev_ptr_null, this must be connected to the field-evict process where lines are forcefully evicted
        .i_field_consume(wo_cb_consume),            // when consume is true, miss must be true. But, not vice-versa. From previous cycle
        
        .o_rev_ptr_null(w_cb_rev_ptr_null),             // Output in current cycle
        .o_rev_ptr(w_cb_rev_ptr)                   // Output in current cycle
    );
    
    in_cdc #(
        .PU_CLK_PERIOD(PU_CLK_PERIOD),
        .CB_CLK_PERIOD(CB_CLK_PERIOD),
        .DATA_W(DATA_W),
        .VTAG_W(VTAG_W),
        .PTAG_W(PTAG_W),
        .OFFS_W(OFFS_W)
    ) in_cdc_i (
        .i_clk(i_cb_clk),
        .i_rst(i_rst),
        .i_field_table_vtag(w_field_table_vtag),
        .i_field_table_wr_data(w_field_table_wr_data),
        .i_cb_consume(w_cb_consume),
        .i_cb_ctag(w_cb_ctag),
        //.i_cb_ptag(w_cb_ptag),
        .i_cb_vtp_offset(w_cb_vtp_offset),
        
        .o_field_table_vtag(wo_field_table_vtag),
        .o_field_table_wr_data(wo_field_table_wr_data),
        .o_cb_consume(wo_cb_consume),
        .o_cb_ctag(wo_cb_ctag),
        .o_cb_lru_ptag(wo_cb_lru_ptag),
        .o_cb_vtp_offset(wo_cb_vtp_offset)
    );
    
    pucb_wb_gen #(
        .DEC_DEP(DEC_DEP),
        .ASSOC(ASSOC),
        .SETS(SETS),
        .DATA_W(DATA_W),
        .PTAG_W(PTAG_W),
        .VTAG_W(VTAG_W),
        .ATAG_W(ATAG_W)
    ) wb_gen_i (
        //.i_field_init(w_field_init),
        .i_cb_miss(w_cb_miss),
        .i_ocb_consume(wo_cb_consume),
        .i_field_wen(w_field_wen),
        .i_field_atag(w_field_atag),
        .i_field_set(w_field_set),
        .i_cb_ctag(w_cb_ctag),
        .i_cb_cline(w_cb_cline),
        .i_cb_lru(w_cb_consume_buf),
        .i_cb_lru_sel(w_cb_consume_sel),
        .i_ofield_table_vtag(wo_field_table_vtag),
        //.i_ofield_table_wr_data(wo_field_table_wr_data),
        .i_cb_rev_ptr_null(w_cb_rev_ptr_null),
        .i_cb_rev_ptr(w_cb_rev_ptr),
        
        //.o_cb_lru_ptag(w_cb_lru_ptag),
        //.o_ocb_lru_ptag(wo_cb_lru_ptag),
        .o_field_table_vtag(w_field_table_vtag),
        .o_field_table_wr_data(w_field_table_wr_data),
        .o_ovalidity_table_null(wo_validity_table_null),
        .o_ovalidity_table_vtag(wo_validity_table_vtag),
        .o_mem_wen(w_mem_wen),
        .o_mem_addr(w_mem_addr)
    );
    
    pucb_controller #(
        .PIPE_DEP(PIPE_DEP),
        .DEC_DEP(DEC_DEP)
    ) controller_i (
        .i_clk(i_pu_clk),
        .i_rst(i_rst),
        .i_cb_consume(w_cb_consume),
        
        .o_cb_lru_sel(w_cb_consume_sel),
        .o_cb_vtp_offset(w_cb_vtp_offset)
    );
    
    initial begin
        r_field_addr = 'b0;
        r_field_wen = 'b0;
    end
    
    always @(posedge i_pu_clk) begin
        r_field_mshr <= (w_field_stall) ? 'b0 : w_cb_miss;
        r_field_addr <= w_field_addr;
        r_field_wen <= w_field_wen;
    end
    
    //always @(posedge i_cb_clk) begin
    //    r_field_init <= (i_rst || !i_cb_en || (r_field_init && !w_cb_consume)) ? 1'b1 : 1'b0; 
    //end
        
endmodule
