
module mapper_stage #(
    parameter OREG_EN       = 0,
    parameter TDM           = 4,
    parameter BITMAP        = 128,
    parameter STAGE         = 7,
    localparam NODES        = BITMAP >> 1,
    localparam STAGES       = $clog2(BITMAP)
)(
    input wire                      i_clk,
    input wire                      i_rst,
    input wire                      i_en,
    input wire                      i_tdm,
    input wire  [BITMAP-1:0]        i_data,

    output wire                     o_en,
    output wire                     o_tdm,
    output wire [NODES-1:0]         o_scb,              // Towards decoder
    output wire [BITMAP-1:0]        o_data
);

    localparam SPLIT            = (STAGE < STAGES-1) ? 2**(STAGE+1) : 2**STAGE;
    
    wire    [BITMAP-1:0]        wi_data;
    wire    [BITMAP-1:0]        wo_data;
    wire    [NODES-1:0]         w_scb;
    wire    [NODES-1:0]         wi_rot_scb;
    wire    [NODES-1:0]         wo_rot_scb;
    
    genvar i;

    assign o_data = wo_data;
    assign o_scb = w_scb;

    // OREG_EN
    generate
        if (OREG_EN) begin
            reg                 r_en;
            reg                 r_tdm;

            assign o_en = r_en;
            assign o_tdm = r_tdm;

            always @(posedge i_clk) begin
                r_en <= i_en;
                r_tdm <= i_tdm;
            end
        end
        else begin
            assign o_en = i_en;
            assign o_tdm = i_tdm;
        end
    endgenerate
    
    // Stage output mapping -> Virtual to physical mapping
    generate      
        if (STAGE_NUM < STAGES-1) begin
            for (i = 0; i < INPUTS; i = i+1) begin
                if ((i/SPLIT)%2 == 0) begin
                    if (i%2 == 0) begin
                        assign wi_data[i] = i_data[i];
                    end
                    else begin
                        assign wi_data[i+SPLIT-1] = i_data[i];
                    end
                end
                else begin
                    if (i%2 == 0) begin
                        assign wi_data[i-SPLIT+1] = i_data[i];
                    end
                    else begin
                        assign wi_data[i] = i_data[i];
                    end
                end
            end
        end
        else begin
            for (i = 0; i < INPUTS; i = i+1) begin
                if (i < SPLIT) begin
                    assign wi_data[2*i] = i_data[i];
                end
                else begin
                    assign wi_data[2*(i-SPLIT)+1] = i_data[i];
                end
            end
        end
    endgenerate
    
    // Node generation
    generate
        for (i = 0; i < NODES; i = i+1) begin
            mapper_node #(
                .OREG_EN(OREG_EN),
                .TDM(TDM)
            ) mapper_node_i (
                .i_clk(i_clk),
                .i_rst(i_rst),
                .i_en(i_en),
                .i_tdm(i_tdm),
                .i_rot_scb(wo_rot_scb[i]),
                .i_d0(wi_data[2*i]),
                .i_d1(wi_data[2*i+1]),
            
                .o_d0(wo_data[2*i]),
                .o_d1(wo_data[2*i+1]),
                .o_rot_scb(wi_rot_scb[i]),
                .o_scb(w_scb[i])
            );
        end
    endgenerate

    // SCB rotator
    rscb_gen_stage #(
        .BITMAP(BITMAP),
        .STAGE(STAGE)
    ) rscb_gen_stage_i (
        .i_scb(wi_rot_scb),

        .o_scb(wo_rot_scb)
    );

endmodule