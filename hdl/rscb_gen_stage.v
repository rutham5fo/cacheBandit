
module rscb_gen_stage #(
        parameter BITMAP            = 128,
        parameter STAGE             = 6,
        localparam MAP_W            = 1 << STAGE,
        localparam NODES            = BITMAP >> (STAGE+1)
    )(
        input wire  [NODES*MAP_W-1:0]   i_scb,

        output wire [NODES*MAP_W-1:0]   o_scb
    );
    
    wire    [MAP_W-1:0]             wi_scb[0:NODES-1];
    wire    [MAP_W-1:0]             wo_scb[0:NODES-1];
    
    genvar i;
    
    generate
        for (i = 0; i < NODES; i = i+1) begin           :   gen_rscb_nodes
            assign wi_scb[i] = i_scb[i*MAP_W +: MAP_W];
            assign o_scb[i*MAP_W +: MAP_W] = wo_scb[i];
            
            rscb_gen_node #(
                .STAGE(STAGE)
            ) rscb_gen_node_i (
                .i_scb(wi_scb[i]),
                
                .o_scb(wo_scb[i])
            );
        end
    endgenerate
    
endmodule