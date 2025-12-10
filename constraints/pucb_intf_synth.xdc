set "pu_clk_p" 10.0;
set "cb_clk_p" 2.5;
set "CDC_RATIO" [expr int($pu_clk_p / $cb_clk_p)];

# Setup pass for pu_clk = 100 MHz ; fail for 200
# Hold fails for both dut to ignored multi-cycle constraints

create_clock -name pu_clk -period $pu_clk_p [get_ports "i_pu_clk"];
create_clock -name cb_clk -period $cb_clk_p [get_ports "i_cb_clk"];

set_input_delay -clock "pu_clk" -max 2.5 [get_ports [list "i_rst" "i_field_mshr" "i_field_wen" "i_field_addr"]];
set_input_delay -clock "pu_clk" -min 1.5 [get_ports [list "i_rst" "i_field_mshr" "i_field_wen" "i_field_addr"]];
set_input_delay -clock "cb_clk" -max 1.5 [get_ports [list "i_cb_consume_buf*"]];
set_input_delay -clock "cb_clk" -min 1.5 [get_ports [list "i_cb_consume_buf*"]];

set_output_delay -clock "pu_clk" -max 1.0 [get_ports [list "o_field_mshr" "o_field_addr" "o_field_wen" "o_mem_wen" "o_mem_addr"]];
set_output_delay -clock "pu_clk" -min 0.0 [get_ports [list "o_field_mshr" "o_field_addr" "o_field_wen" "o_mem_wen" "o_mem_addr"]];
set_output_delay -clock "cb_clk" -max 1.0 [get_ports [list "o_cb_consume" "o_cb_ptag" "o_cb_vtp_offset"]];
set_output_delay -clock "cb_clk" -min 0.0 [get_ports [list "o_cb_consume" "o_cb_ptag" "o_cb_vtp_offset"]];

#set_max_delay 10 -from [get_clocks "sys_clk"] -to [get_ports [list "o_mem_addr"]] -datapath_only;


#set_input_delay -clock "sys_clk" -max 1.5 [get_ports [list "i_reset" "i_scb" "i_ps_wr_en" "i_ps_wr_addr" "i_ps_wr_data" "i_pl_wr_en" "i_pl_wr_data" "i_ps_rd_addr"]];
#set_input_delay -clock "sys_clk" -min 1.5 [get_ports [list "i_reset" "i_scb" "i_ps_wr_en" "i_ps_wr_addr" "i_ps_wr_data" "i_pl_wr_en" "i_pl_wr_data" "i_ps_rd_addr"]];

#set_output_delay -clock "sys_clk" -max 0.5 [get_ports [list "o_ps_rd_data" "o_pl_rd_data"]];
#set_output_delay -clock "sys_clk" -min 0.5 [get_ports [list "o_ps_rd_data" "o_pl_rd_data"]];

#set_input_delay -clock "sys_clk" -max 1.5 [get_ports [list "i_reset" "i_scb" "i_pl_wr_en" "i_pl_wr_addr" "i_pl_wr_data" "i_pl_rd_addr" "i_mem_rd_data"]];
#set_input_delay -clock "sys_clk" -min 1.5 [get_ports [list "i_reset" "i_scb" "i_pl_wr_en" "i_pl_wr_addr" "i_pl_wr_data" "i_pl_rd_addr" "i_mem_rd_data"]];

#set_output_delay -clock "sys_clk" -max 0.5 [get_ports [list "o_mem_wr_en" "o_mem_wr_addr" "o_mem_wr_data" "o_mem_rd_addr" "o_pl_rd_data"]];
#set_output_delay -clock "sys_clk" -min 0.5 [get_ports [list "o_mem_wr_en" "o_mem_wr_addr" "o_mem_wr_data" "o_mem_rd_addr" "o_pl_rd_data"]];

# Slow-to-Fast CDC
set_multicycle_path $CDC_RATIO -setup -from [get_clocks "pu_clk"] -to [get_clocks "cb_clk"] 
set_multicycle_path [expr {$CDC_RATIO-1}] -hold -end -from [get_clocks "pu_clk"] -to [get_clocks "cb_clk"]

# Fast-to-Slow CDC
set_multicycle_path $CDC_RATIO -setup -start -from [get_clocks "cb_clk"] -to [get_clocks "pu_clk"] 
set_multicycle_path [expr {$CDC_RATIO-1}] -hold -from [get_clocks "cb_clk"] -to [get_clocks "pu_clk"]

#set_false_path -from [get_ports "i_rst"] -to [all_registers];
