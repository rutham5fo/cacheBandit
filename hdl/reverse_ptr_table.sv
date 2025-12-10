`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 18.10.2025 02:29:01
// Design Name: 
// Module Name: reverse_ptr_table
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


module reverse_ptr_table #(
        //parameter BRAM_EN           = 0,
        parameter BLOCK_W           = 128,
        parameter ASSOC             = 4,
        parameter SETS              = 512,
        localparam BLOCK_L          = $clog2(BLOCK_W),
        localparam ASSOC_W          = $clog2(ASSOC),
        localparam SET_W            = $clog2(SETS)
    )(
        input wire                      i_clk,
        //input wire                      i_cb_clk,
        input wire                      i_rst,
        input wire  [BLOCK_L-1:0]       i_rd_addr,                  // From current cycle
        input wire  [BLOCK_L-1:0]       i_wr_addr,                  // From previous cycle
        input wire  [SET_W+ASSOC_W-1:0] i_wr_data,                  // From previous cycle
        input wire                      i_field_evict,              // When true, value at wr_addr is forced to point to null | This is not the delayed signal of o_rev_ptr_null, this must be connected to the field-evict process where lines are forcefully evicted
        input wire                      i_field_consume,            // when consume is true, miss must be true. But, not vice-versa. From previous cycle
        
        output wire                     o_rev_ptr_null,             // Output in current cycle
        output wire [SET_W+ASSOC_W-1:0] o_rev_ptr                   // Output in current cycle
    );
    
    //wire                        wo_rev_ptr_null;
    //wire    [SET_W+ASSOC_W-1:0] wo_rev_ptr;
    
    //reg                         r_rev_ptr_null;
    //reg     [SET_W+ASSOC_W-1:0] r_rev_ptr;
    
    //assign o_rev_ptr_null = r_rev_ptr_null;
	//assign o_rev_ptr = r_rev_ptr;
	
	//initial begin
	//   r_rev_ptr_null = 'b0;
	//   r_rev_ptr = 'b0;
	//end
	
	//always @(posedge i_cb_clk) begin
	//   r_rev_ptr_null <= wo_rev_ptr_null;
	//   r_rev_ptr <= wo_rev_ptr;
	//end
    
    //genvar i;
    
`ifndef SIMLIB_VERILATOR
    ///////////////////////////////////////////////////////////////////////
	//  READ_WIDTH | BRAM_SIZE | READ Depth  | RDADDR Width |            //
	// WRITE_WIDTH |           | WRITE Depth | WRADDR Width |  WE Width  //
	// ============|===========|=============|==============|============//
	//    37-72    |  "36Kb"   |      512    |     9-bit    |    8-bit   //
	//    19-36    |  "36Kb"   |     1024    |    10-bit    |    4-bit   //
	//    19-36    |  "18Kb"   |      512    |     9-bit    |    4-bit   //
	//    10-18    |  "36Kb"   |     2048    |    11-bit    |    2-bit   //
	//    10-18    |  "18Kb"   |     1024    |    10-bit    |    2-bit   //
	//     5-9     |  "36Kb"   |     4096    |    12-bit    |    1-bit   //
	//     5-9     |  "18Kb"   |     2048    |    11-bit    |    1-bit   //
	//     3-4     |  "36Kb"   |     8192    |    13-bit    |    1-bit   //
	//     3-4     |  "18Kb"   |     4096    |    12-bit    |    1-bit   //
	//       2     |  "36Kb"   |    16384    |    14-bit    |    1-bit   //
	//       2     |  "18Kb"   |     8192    |    13-bit    |    1-bit   //
	//       1     |  "36Kb"   |    32768    |    15-bit    |    1-bit   //
	//       1     |  "18Kb"   |    16384    |    14-bit    |    1-bit   //
	///////////////////////////////////////////////////////////////////////
		   		
	localparam BRAM_SIZE        = (BLOCK_W > 512) ? "36Kb" : "18Kb";
	localparam WRITE_WIDTH      = 32;
	localparam READ_WIDTH       = 32;
	localparam WRADDR_WIDTH     = (BLOCK_W > 512) ? 10 : 9;
	localparam RDADDR_WIDTH     = (BLOCK_W > 512) ? 10 : 9;
	localparam WEN_WIDTH        = 4;
	localparam WRITE_MODE       = "READ_FIRST";
	       
	localparam BRAM_NULL_CONST  = 2**WRITE_WIDTH >> 1;
	localparam BRAM_INIT_CONST  = 256'h8000000080000000800000008000000080000000800000008000000080000000;
	       
	wire    [RDADDR_WIDTH-1:0]  w_rd_addr;
	wire    [WRADDR_WIDTH-1:0]  w_wr_addr;
	wire    [READ_WIDTH-1:0]    w_rd_data;
	wire                        w_rev_ptr_null;
	wire    [SET_W+ASSOC_W-1:0] w_rev_ptr;
	wire    [WRITE_WIDTH-1:0]   w_wr_data;
	wire    [WEN_WIDTH-1:0]     w_wr_en;
	       
	assign w_rd_addr = i_rd_addr;
	assign w_rev_ptr_null = w_rd_data[READ_WIDTH-1];                            // MSB indicates null ptr
	assign w_rev_ptr = w_rd_data[0 +: SET_W+ASSOC_W];
	assign w_wr_addr = i_wr_addr;
	assign w_wr_en = (i_field_evict || i_field_consume) ? -1 : 0;
	assign w_wr_data = (i_field_evict) ? BRAM_NULL_CONST : i_wr_data;
	       
	assign o_rev_ptr_null = w_rev_ptr_null;
	assign o_rev_ptr = w_rev_ptr;
	       
	BRAM_SDP_MACRO #(
		.BRAM_SIZE(BRAM_SIZE), // Target BRAM, "18Kb" or "36Kb"
		.DEVICE("7SERIES"), // Target device: "7SERIES"
		.WRITE_WIDTH(WRITE_WIDTH),    // Valid values are 1-72 (37-72 only valid when BRAM_SIZE="36Kb")
		.READ_WIDTH(READ_WIDTH),     // Valid values are 1-72 (37-72 only valid when BRAM_SIZE="36Kb")
		.DO_REG(0),         // Optional output register (0 or 1)
		.INIT_FILE ("NONE"),
		.SIM_COLLISION_CHECK ("ALL"), // Collision check enable "ALL", "WARNING_ONLY",
										//   "GENERATE_X_ONLY" or "NONE"
		.SRVAL(72'h000000000000000000), // Set/Reset value for port output
		.INIT(72'h000000000000000000),  // Initial values on output port
		.WRITE_MODE(WRITE_MODE),  // Specify "READ_FIRST" for same clock or synchronous clocks
										//   Specify "WRITE_FIRST for asynchronous clocks on ports
		.INIT_00(BRAM_INIT_CONST),
		.INIT_01(BRAM_INIT_CONST),
		.INIT_02(BRAM_INIT_CONST),
		.INIT_03(BRAM_INIT_CONST),
		.INIT_04(BRAM_INIT_CONST),
		.INIT_05(BRAM_INIT_CONST),
		.INIT_06(BRAM_INIT_CONST),
		.INIT_07(BRAM_INIT_CONST),
		.INIT_08(BRAM_INIT_CONST),
		.INIT_09(BRAM_INIT_CONST),
		.INIT_0A(BRAM_INIT_CONST),
		.INIT_0B(BRAM_INIT_CONST),
		.INIT_0C(BRAM_INIT_CONST),
		.INIT_0D(BRAM_INIT_CONST),
		.INIT_0E(BRAM_INIT_CONST),
		.INIT_0F(BRAM_INIT_CONST),
		.INIT_10(BRAM_INIT_CONST),
		.INIT_11(BRAM_INIT_CONST),
		.INIT_12(BRAM_INIT_CONST),
		.INIT_13(BRAM_INIT_CONST),
		.INIT_14(BRAM_INIT_CONST),
		.INIT_15(BRAM_INIT_CONST),
		.INIT_16(BRAM_INIT_CONST),
		.INIT_17(BRAM_INIT_CONST),
		.INIT_18(BRAM_INIT_CONST),
		.INIT_19(BRAM_INIT_CONST),
		.INIT_1A(BRAM_INIT_CONST),
		.INIT_1B(BRAM_INIT_CONST),
		.INIT_1C(BRAM_INIT_CONST),
		.INIT_1D(BRAM_INIT_CONST),
		.INIT_1E(BRAM_INIT_CONST),
		.INIT_1F(BRAM_INIT_CONST),
		.INIT_20(BRAM_INIT_CONST),
		.INIT_21(BRAM_INIT_CONST),
		.INIT_22(BRAM_INIT_CONST),
		.INIT_23(BRAM_INIT_CONST),
		.INIT_24(BRAM_INIT_CONST),
		.INIT_25(BRAM_INIT_CONST),
		.INIT_26(BRAM_INIT_CONST),
		.INIT_27(BRAM_INIT_CONST),
		.INIT_28(BRAM_INIT_CONST),
		.INIT_29(BRAM_INIT_CONST),
		.INIT_2A(BRAM_INIT_CONST),
		.INIT_2B(BRAM_INIT_CONST),
		.INIT_2C(BRAM_INIT_CONST),
		.INIT_2D(BRAM_INIT_CONST),
		.INIT_2E(BRAM_INIT_CONST),
		.INIT_2F(BRAM_INIT_CONST),
		.INIT_30(BRAM_INIT_CONST),
		.INIT_31(BRAM_INIT_CONST),
		.INIT_32(BRAM_INIT_CONST),
		.INIT_33(BRAM_INIT_CONST),
		.INIT_34(BRAM_INIT_CONST),
		.INIT_35(BRAM_INIT_CONST),
		.INIT_36(BRAM_INIT_CONST),
		.INIT_37(BRAM_INIT_CONST),
		.INIT_38(BRAM_INIT_CONST),
		.INIT_39(BRAM_INIT_CONST),
		.INIT_3A(BRAM_INIT_CONST),
		.INIT_3B(BRAM_INIT_CONST),
		.INIT_3C(BRAM_INIT_CONST),
		.INIT_3D(BRAM_INIT_CONST),
		.INIT_3E(BRAM_INIT_CONST),
		.INIT_3F(BRAM_INIT_CONST),
		
		// The next set of INIT_xx are valid when configured as 36Kb
		.INIT_40(BRAM_INIT_CONST),
		.INIT_41(BRAM_INIT_CONST),
		.INIT_42(BRAM_INIT_CONST),
		.INIT_43(BRAM_INIT_CONST),
		.INIT_44(BRAM_INIT_CONST),
		.INIT_45(BRAM_INIT_CONST),
		.INIT_46(BRAM_INIT_CONST),
		.INIT_47(BRAM_INIT_CONST),
		.INIT_48(BRAM_INIT_CONST),
		.INIT_49(BRAM_INIT_CONST),
		.INIT_4A(BRAM_INIT_CONST),
		.INIT_4B(BRAM_INIT_CONST),
		.INIT_4C(BRAM_INIT_CONST),
		.INIT_4D(BRAM_INIT_CONST),
		.INIT_4E(BRAM_INIT_CONST),
		.INIT_4F(BRAM_INIT_CONST),
		.INIT_50(BRAM_INIT_CONST),
		.INIT_51(BRAM_INIT_CONST),
		.INIT_52(BRAM_INIT_CONST),
		.INIT_53(BRAM_INIT_CONST),
		.INIT_54(BRAM_INIT_CONST),
		.INIT_55(BRAM_INIT_CONST),
		.INIT_56(BRAM_INIT_CONST),
		.INIT_57(BRAM_INIT_CONST),
		.INIT_58(BRAM_INIT_CONST),
		.INIT_59(BRAM_INIT_CONST),
		.INIT_5A(BRAM_INIT_CONST),
		.INIT_5B(BRAM_INIT_CONST),
		.INIT_5C(BRAM_INIT_CONST),
		.INIT_5D(BRAM_INIT_CONST),
		.INIT_5E(BRAM_INIT_CONST),
		.INIT_5F(BRAM_INIT_CONST),
		.INIT_60(BRAM_INIT_CONST),
		.INIT_61(BRAM_INIT_CONST),
		.INIT_62(BRAM_INIT_CONST),
		.INIT_63(BRAM_INIT_CONST),
		.INIT_64(BRAM_INIT_CONST),
		.INIT_65(BRAM_INIT_CONST),
		.INIT_66(BRAM_INIT_CONST),
		.INIT_67(BRAM_INIT_CONST),
		.INIT_68(BRAM_INIT_CONST),
		.INIT_69(BRAM_INIT_CONST),
		.INIT_6A(BRAM_INIT_CONST),
		.INIT_6B(BRAM_INIT_CONST),
		.INIT_6C(BRAM_INIT_CONST),
		.INIT_6D(BRAM_INIT_CONST),
		.INIT_6E(BRAM_INIT_CONST),
		.INIT_6F(BRAM_INIT_CONST),
		.INIT_70(BRAM_INIT_CONST),
		.INIT_71(BRAM_INIT_CONST),
		.INIT_72(BRAM_INIT_CONST),
		.INIT_73(BRAM_INIT_CONST),
		.INIT_74(BRAM_INIT_CONST),
		.INIT_75(BRAM_INIT_CONST),
		.INIT_76(BRAM_INIT_CONST),
		.INIT_77(BRAM_INIT_CONST),
		.INIT_78(BRAM_INIT_CONST),
		.INIT_79(BRAM_INIT_CONST),
		.INIT_7A(BRAM_INIT_CONST),
		.INIT_7B(BRAM_INIT_CONST),
		.INIT_7C(BRAM_INIT_CONST),
		.INIT_7D(BRAM_INIT_CONST),
		.INIT_7E(BRAM_INIT_CONST),
		.INIT_7F(BRAM_INIT_CONST),
		
		// The next set of INITP_xx are for the parity bits
		.INITP_00(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_01(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_02(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_03(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_04(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_05(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_06(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_07(256'h0000000000000000000000000000000000000000000000000000000000000000),
		
		// The next set of INITP_xx are valid when configured as 36Kb
		.INITP_08(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_09(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_0A(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_0B(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_0C(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_0D(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_0E(256'h0000000000000000000000000000000000000000000000000000000000000000),
		.INITP_0F(256'h0000000000000000000000000000000000000000000000000000000000000000)
	) BRAM_SDP_MACRO_inst (
		.DO(w_rd_data),         // Output read data port, width defined by READ_WIDTH parameter
		.DI(w_wr_data),         // Input write data port, width defined by WRITE_WIDTH parameter
		.RDADDR(w_rd_addr), // Input read address, width defined by read port depth
		.RDCLK(i_clk),   // 1-bit input read clock
		.RDEN(1'b1),     // 1-bit input read port enable
		.REGCE(1'b0),   // 1-bit input read output register enable
		.RST(i_rst),       // 1-bit input reset
		.WE(w_wr_en),         // Input write enable, width defined by write port depth
		.WRADDR(w_wr_addr), // Input write address, width defined by write port depth
		.WRCLK(i_clk),   // 1-bit input write clock
		.WREN(1'b1)      // 1-bit input write port enable
	);
`else
    localparam [SET_W+ASSOC_W:0] REG_NULL_CONST   = 1'b1 << SET_W+ASSOC_W;
            
    wire    [BLOCK_L-1:0]       w_rd_addr;
	wire    [BLOCK_L-1:0]       w_wr_addr;
	wire    [SET_W+ASSOC_W:0]   w_rd_data;             // Include null_ptr flag
	wire                        w_rev_ptr_null;
	wire    [SET_W+ASSOC_W-1:0] w_rev_ptr;
	wire    [SET_W+ASSOC_W:0]   w_wr_data;             // Include null_ptr flag
	wire                        w_wr_en_flag;
	wire    [BLOCK_W-1:0]       w_wr_en;
	        
	reg     [SET_W+ASSOC_W:0]   r_ptr_table[0:BLOCK_W-1];
	        
	integer k;
	        
	assign w_rd_addr = i_rd_addr;
	assign w_rd_data = r_ptr_table[w_rd_addr];
	assign w_rev_ptr_null = w_rd_data[SET_W+ASSOC_W];                            // MSB indicates null ptr
	assign w_rev_ptr = w_rd_data[0 +: SET_W+ASSOC_W];
	assign w_wr_addr = i_wr_addr;
	assign w_wr_en_flag = i_field_evict | i_field_consume;
	assign w_wr_en = w_wr_en_flag << w_wr_addr;
	assign w_wr_data = {i_field_evict, i_wr_data};
	        
	assign o_rev_ptr_null = w_rev_ptr_null;
	assign o_rev_ptr = w_rev_ptr;
            
    // Table regfile
    always @(posedge i_clk) begin
        for (k = 0; k < BLOCK_W; k = k+1) begin
            r_ptr_table[k] <= (i_rst) ? (REG_NULL_CONST | k) : (w_wr_en[k]) ? w_wr_data : r_ptr_table[k];
        end
    end
`endif
    	
endmodule
