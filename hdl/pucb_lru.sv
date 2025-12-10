`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 17.10.2025 02:24:24
// Design Name: 
// Module Name: pucb_lru
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


module pucb_lru #(
        //parameter BRAM_EN               = 0,
        parameter ASSOC                 = 4,
        //parameter SETS                  = 512,
        localparam NODES                = ASSOC-1,
        localparam ASSOC_W              = $clog2(ASSOC)
        //localparam SET_W                = $clog2(SETS)
    )(
        //input wire                      i_clk,
        //input wire                      i_rst,
        input wire [NODES-1:0]          i_field_lru_cur_bits,
        
        output wire [NODES-1:0]         o_field_lru_nxt_bits,
        output wire [ASSOC_W-1:0]       o_field_lru_nxt
    );
    
    localparam STAGES                   = ASSOC_W;
    
    // Reduction length func
    function integer read_ptr (input integer lp_cnt);
        integer i, k;
        k = 0;
        for (i = 0; i < lp_cnt; i = i+1) begin
            k = k+(2**k);
        end
        return k;
    endfunction
    
    wire    [NODES-1:0]                 w_nactv;
    wire    [NODES-1:0]                 w_tbits;
    wire    [NODES-1:0]                 w_pactv;
    wire    [NODES-1:0]                 w_field_lru_cur_bits;
    wire    [NODES-1:0]                 w_field_lru_nxt_bits;
    wire    [ASSOC_W-1:0]               w_field_lru_nxt;
        
    genvar i;
    
    assign w_field_lru_cur_bits = i_field_lru_cur_bits;
    
    assign o_field_lru_nxt_bits = w_field_lru_nxt_bits;
    assign o_field_lru_nxt = w_field_lru_nxt;
    
    generate
        for (i = 0; i < STAGES; i = i+1) begin  :   gen_lru_nxt
            localparam READ_START   = read_ptr(i);
            localparam READ_LEN     = 2**i;
            wire    [READ_LEN-1:0]  w_tread;
            assign w_tread = w_tbits[READ_START +: READ_LEN];
            assign w_field_lru_nxt[STAGES-1-i] = |w_tread;
        end
    endgenerate
    
    // Tree generation for nxt_bits computation
    generate
        for (i = 0; i < NODES; i = i+1) begin       :   gen_lru_nxt_bits
            // L_CHILD (odd) = 2*i+1 ; R_CHILD (even) = 2*i+2
            localparam PROOT    = (i%2) ? (i-1)/2 : i/2;
            localparam PSIDE    = (i%2) ? 0 : 1;             // 0 = odd (i%2) = l_child ; 1 = even = r_child
            // Root node
            if (i == 0) begin
                assign w_pactv[i] = 1'b1;
                assign w_field_lru_nxt_bits[i] = w_pactv[i] ^ w_field_lru_cur_bits[i];
            end
            // Other nodes
            else begin
                if (PSIDE) begin
                    assign w_pactv[i] = w_field_lru_cur_bits[PROOT] & w_pactv[PROOT];
                end
                else begin
                    assign w_pactv[i] = ~w_field_lru_cur_bits[PROOT] & w_pactv[PROOT];
                end
                assign w_field_lru_nxt_bits[i] = w_pactv[i] ^ w_field_lru_cur_bits[i];
            end
        end
    endgenerate
    
    // Tree generation for lru_nxt computation
    generate
        for (i = 0; i < NODES; i = i+1) begin       :   gen_lru_nxt_setup
            // L_CHILD (odd) = 2*i+1 ; R_CHILD (even) = 2*i+2
            localparam NROOT    = (i%2) ? (i-1)/2 : i/2;
            localparam NSIDE    = (i%2) ? 0 : 1;             // 0 = odd = l_child ; 1 = even = r_child
            // Root node
            if (i == 0) begin
                assign w_nactv[i] = 1'b1;
                assign w_tbits[i] = ~w_field_lru_cur_bits[i];
            end
            // Other nodes
            else begin
                assign w_tbits[i] = w_nactv[i] & w_field_lru_cur_bits[i];
                // Because the top most root node has its cur_bits flipped, the succeding stage nodes flip left-right
                // Hence the speacial case for nodes 1 and 2 who are children of node 0 (root)
                // All other nodes below stage 1 follow the regular patter determined solely by NSIDE
                if ((NSIDE && i != 2) || i == 1) begin
                    assign w_nactv[i] = w_field_lru_cur_bits[NROOT] & w_nactv[NROOT];
                end
                else begin
                    assign w_nactv[i] = ~w_field_lru_cur_bits[NROOT] & w_nactv[NROOT];
                end
            end
        end
    endgenerate
    
endmodule
