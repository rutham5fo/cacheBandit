
module mapper #(
    parameter BLOCKS                = 4,            // No. of bins of size BLOCK_W the mapper will cycle 
    parameter BLOCK_W               = 128,
    parameter TDM_MASK              = 'haa
)(
    input wire                          i_clk,
    input wire                          i_rst,
    input wire                          i_alloc_done,
    input wire                          i_tdm_en,
    input wire  [BLOCK_W-1:0]           i_mask,

    output wire [STAGES-1:0][NODES-1:0] o_scb
);

    localparam NODES                = BLOCK_W >> 1;
    localparam STAGES               = $clog2(BLOCK_W);

    genvar i;

    wire    [STAGES-1:0]            w_map_stg_en;
    wire    [STAGES-1:0]            w_map_stg_tdm;
    wire    [NODES-1:0]             w_map_stg_scb[0:STAGES-1];
    wire    [BLOCK_W-1:0]           w_map_stg_data[0:STAGES-1];

    generate
        for (i = 0; i < STAGES; i = i+1) begin          :   gen_io
            assign o_scb[i] = w_map_stg_scb[i];
        end
    endgenerate

    generate
        for (i = 0; i < STAGES; i = i+1) begin          :   gen_stages
            if (i == 0) begin
                mapper_stage #(
                    .OREG_EN(TDM_MASK[i]),
                    .TDM(BLOCKS),
                    .BITMAP(BLOCK_W),
                    .STAGE(i),
                    .NODES(NODES),
                    .STAGES(STAGES)
                ) mapper_stg_i (
                    .i_clk(i_clk),
                    .i_rst(i_rst),
                    .i_en(i_alloc_done),
                    .i_tdm(i_tdm_en),
                    .i_data(i_mask),

                    .o_en(w_map_stg_en[i]),
                    .o_tdm(w_map_stg_tdm[i]),
                    .o_scb(w_map_stg_scb[i]),
                    .o_data(w_map_stg_data[i])
                );
            end
            else begin
                mapper_stage #(
                    .OREG_EN(TDM_MASK[i]),
                    .TDM(BLOCKS),
                    .BITMAP(BLOCK_W),
                    .STAGE(i),
                    .NODES(NODES),
                    .STAGES(STAGES)
                ) mapper_stg_i (
                    .i_clk(i_clk),
                    .i_rst(i_rst),
                    .i_en(w_map_stg_en[i-1]),
                    .i_tdm(w_map_stg_tdm[i-1]),
                    .i_data(w_map_stg_data[i-1]),

                    .o_en(w_map_stg_en[i]),
                    .o_tdm(w_map_stg_tdm[i]),
                    .o_scb(w_map_stg_scb[i]),
                    .o_data(w_map_stg_data[i])
                );
            end
        end
    endgenerate

endmodule