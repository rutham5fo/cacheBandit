`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 17.10.2025 14:42:05
// Design Name: 
// Module Name: pucb_comparator
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


module pucb_comparator #(
        parameter ASSOC             = 4,
        //parameter ATAG_W            = 10,
        parameter PTAG_W            = 10,
        localparam ASSOC_W          = $clog2(ASSOC)
    )(
        input wire  [ASSOC-1:0][PTAG_W-1:0] i_field_ptag,
        input wire  [ASSOC-1:0]             i_field_hit,
        input wire  [ASSOC-1:0]             i_field_pvld,
        input wire  [ASSOC_W-1:0]           i_field_lru,
        input wire  [PTAG_W-1:0]            i_field_lru_ptag,
        
        output wire [ASSOC_W-1:0]           o_cb_line,                 // Line pointed by comparator | in-case of a miss, points to lru line, else to hit line
        output wire [PTAG_W-1:0]            o_cb_ptag                      // Ptag with associated o_tag_line
    );
    
    wire    [ASSOC_W-1:0]           w_field_lru;
    wire    [PTAG_W-1:0]            w_field_lru_ptag;
    wire    [ASSOC-1:0]             w_field_hit;
    wire    [ASSOC-1:0]             w_field_pvld;
    wire    [ASSOC-1:0][PTAG_W-1:0] w_field_ptag;
    
    reg     [ASSOC_W-1:0]           w_comp_line;
    reg     [PTAG_W-1:0]            w_comp_ptag;
        
    integer k;
    
    assign w_field_lru = i_field_lru;
    assign w_field_lru_ptag = i_field_lru_ptag;
    assign w_field_hit = i_field_hit;
    assign w_field_pvld = i_field_pvld;
    assign w_field_ptag = i_field_ptag;
    
    assign o_cb_line = w_comp_line;
    assign o_cb_ptag = w_comp_ptag;
    
    // Comparator block
    always @(*) begin
        w_comp_ptag = w_field_lru_ptag;
        w_comp_line = w_field_lru;
        for (k = 0; k < ASSOC; k = k+1) begin   :       comb_comparator
            if (w_field_pvld[k] && w_field_hit[k]) begin
                w_comp_ptag = w_field_ptag[k];
                w_comp_line = k;
            end
        end
    end
    
endmodule
