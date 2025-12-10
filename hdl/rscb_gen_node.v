
module rscb_gen_node #(
        parameter STAGE                 = 6,
        localparam MAP_W                = 1 << STAGE
    )(
        input wire  [MAP_W-1:0]         i_scb,
        
        output wire [MAP_W-1:0]         o_scb
    );
    
    wire    [MAP_W-1:0]             w_rotc;
    
    genvar i;
    
    generate
        if (STAGE) begin
            assign w_rotc = {i_scb[0 +: MAP_W-1], ~i_scb[MAP-1]};
        end
        else begin
            assign w_rotc = ~i_scb;
        end
    endgenerate
    
    assign o_scb = w_rotc;
    
endmodule