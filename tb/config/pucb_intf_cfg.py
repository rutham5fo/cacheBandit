import os
import math
import argparse
import copy
from . import config_utils as cfg_util

class default_params:

    def __init__(self):
        ## Params
        self.def_param = {
            ## DUT PARAMS
            'WORDS_PER_LINE'    : 16,           # Number of words per line
            #'PRIV_ASSOC'        : 4,
            'ASSOC'             : 4,            # Number of lines per set
            'SETS'              : 32,           # Number of sets
            'PAGES'             : 2,
            'PAGE_SIZE'         : 16,
            'DEC_DEP'           : 1,
            'PIPE_DEP'          : 3,
            'ADDR_W'            : 32,
            'DATA_W'            : 32,
            'PU_CLK_PERIOD'     : 10,
            'CB_CLK_PERIOD'     : 5,
            ## TB params
            'CLK_Q'             : 1,
            'CLK_UNIT'          : "ns"
        }

    def get_params(self):
        return copy.deepcopy(self.def_param)

class config:

    def __init__(self, params):

        # Params
        self._params = params

        # Gen params
        self.words_per_line = self._params['WORDS_PER_LINE']
        self.assoc = self._params['ASSOC']
        self.sets = self._params['SETS']
        self.page_size = self._params['PAGE_SIZE']
        self.dec_len = self._params['DEC_DEP']
        self.pipe_len = self._params['PIPE_DEP']
        self.addr_len = self._params['ADDR_W']
        self.data_len = self._params['DATA_W']

        self.line_w = int(math.ceil(math.log(self.words_per_line, 2)))
        self.assoc_w = int(math.ceil(math.log(self.assoc, 2)))
        self.set_w = int(math.ceil(math.log(self.sets, 2)))
        self.addr_w = self.addr_len
        self.data_w = self.data_len
        self.ptag_w = int(math.ceil(math.log(self.page_size, 2)))
        self.vtag_w = self.set_w + self.assoc_w
        self.full_atag_w = self.addr_w - self.set_w
        self.valid_atag_w = self.data_w - self.ptag_w
        self.atag_w = self.full_atag_w if (self.full_atag_w < self.valid_atag_w) else self.valid_atag_w
        self.dec_w = int(math.ceil(math.log(self.dec_len+1, 2)))
        self.offs_w = int(math.ceil(math.log(self.pipe_len, 2)))
        self.flru_nodes = self.assoc-1

        ## Regs
        self._regs = {
            ### REG_NAME                  : INIT_VAL
            ## Writeback Pipeline regs
            'r_wb_cb_consume'           : 0,
            'r_wb_cb_miss'              : 0,
            'r_wb_cb_ctag'              : 0,
            'r_wb_cb_cline'             : 0,
            'r_wb_buf_phy_loc'          : 0,
            'r_wb_atag'                 : 0,
            'r_wb_set_sel'              : 0,
            'r_wb_cb_rev_ptr_null'      : 0,
            'r_wb_cb_rev_ptr'           : 0,
            'r_wb_cb_consume'           : 0,
            'r_wb_cb_vtp_offset'        : 0,
            ## output regs
            'r_dma_rd_addr'             : 0,
            'r_heap_wr_addr'            : 0,
            'r_dma_en'                  : 0
        }

        ## Nets
        # Use in conjuction with cocotb for cosim scoreboarding.
        # The "scored @ writeback_phase" (values: 0/1) indicates on which edge the 
        # monitor will sample the port at. A value of '0' signifies the Cosim model 
        # scores this port during the combinational logic (live-value), and value of '1'
        # is for when the port is scored during the writeback phase (registered-value).
        self._nets = {
            ### NET_NAME              : [SCORE, scored @ writeback_phase]
            'w_field_stall'             : [0, 0],
            'w_field_set'               : [0, 0],
            'w_field_atag'              : [0, 0],
            'w_field_wen'               : [0, 0],
            'w_cb_tdm'                  : [0, 0],
            'w_cb_hit'                  : [0, 0],
            'w_cb_miss'                 : [0, 0],
            'w_cb_consume'              : [0, 0],
            'w_cb_cline'                : [0, 0],
            'w_cb_ctag'                 : [0, 0],
            'w_cb_lru_cur_bits'         : [0, 0],
            'w_cb_lru_cur'              : [0, 0],
            'w_cb_lru_nxt_bits'         : [0, 0],
            'w_cb_lru_nxt'              : [0, 0],
            'w_cb_consume_sel'          : [0, 0],
            'w_cb_vtp_offset'           : [0, 0],
            'w_cb_rev_ptr'              : [0, 0],
            'w_cb_rev_ptr_null'         : [0, 0],
            'wo_validity_set'           : [0, 1],
            'wo_validity_set_addr'      : [0, 1],
            'wo_validity_clear'         : [0, 1],
            'wo_validity_clear_addr'    : [0, 1],
            'wo_field_table_wr_en'      : [0, 1],
            'wo_field_table_wr_data'    : [0, 1],
            'wo_field_table_wr_addr'    : [0, 1],
            'wo_rev_ptr_wr_addr'        : [0, 1],
            'wo_rev_ptr_wr_data'        : [0, 1],
            'wo_rev_ptr_wr_en'          : [0, 1],
            'wo_cb_consume'             : [0, 1],
            'wo_cb_vtp_offset'          : [0, 1],
            ## Output sigs
            'o_dma_rd_addr'         : [0, 0],
            'o_heap_wr_addr'        : [0, 0],
            'o_dma_en'              : [0, 0],
            'o_mem_addr'            : [0, 0],
            'o_mem_wen'             : [0, 0],
            'o_cb_miss'             : [0, 0]
        }

        ## Expose Ports
        # No output port will be generated if 'width' is 0
        self._dbg_ports = [
            {'modName': 'pucb_intf', 'netName': 'w_field_stall', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': 1},
            {'modName': 'pucb_intf', 'netName': 'w_field_set', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.set_w},
            {'modName': 'pucb_intf', 'netName': 'w_field_atag', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.atag_w},
            {'modName': 'pucb_intf', 'netName': 'w_field_wen', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': 1},
            {'modName': 'pucb_intf', 'netName': 'w_cb_hit', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': 1},
            {'modName': 'pucb_intf', 'netName': 'w_cb_miss', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': 1},
            {'modName': 'pucb_intf', 'netName': 'w_cb_consume', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': 1},
            {'modName': 'pucb_intf', 'netName': 'w_cb_cline', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.assoc_w},
            {'modName': 'pucb_intf', 'netName': 'w_cb_ctag', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.ptag_w},
            {'modName': 'pucb_intf', 'netName': 'w_cb_lru_cur_bits', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.flru_nodes},
            {'modName': 'pucb_intf', 'netName': 'w_cb_lru_cur', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.assoc_w},
            {'modName': 'pucb_intf', 'netName': 'w_cb_lru_nxt_bits', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.flru_nodes},
            {'modName': 'pucb_intf', 'netName': 'w_cb_lru_nxt', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.assoc_w},
            {'modName': 'pucb_intf', 'netName': 'w_cb_consume_sel', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.dec_w},
            {'modName': 'pucb_intf', 'netName': 'w_cb_vtp_offset', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.offs_w},
            {'modName': 'pucb_intf', 'netName': 'w_cb_rev_ptr', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.vtag_w},
            {'modName': 'pucb_intf', 'netName': 'w_cb_rev_ptr_null', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': 1},
            {'modName': 'pucb_intf', 'netName': 'wo_validity_set', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': 1},
            {'modName': 'pucb_intf', 'netName': 'wo_validity_set_addr', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.vtag_w},
            {'modName': 'pucb_intf', 'netName': 'wo_validity_clear', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': 1},
            {'modName': 'pucb_intf', 'netName': 'wo_validity_clear_addr', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.vtag_w},
            {'modName': 'pucb_intf', 'netName': 'wo_field_table_wr_en', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': 1},
            {'modName': 'pucb_intf', 'netName': 'wo_field_table_wr_data', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.data_w},
            {'modName': 'pucb_intf', 'netName': 'wo_field_table_wr_addr', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.vtag_w},
            {'modName': 'pucb_intf', 'netName': 'wo_rev_ptr_wr_en', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': 1},
            {'modName': 'pucb_intf', 'netName': 'wo_rev_ptr_wr_addr', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.ptag_w},
            {'modName': 'pucb_intf', 'netName': 'wo_rev_ptr_wr_data', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.vtag_w},
            {'modName': 'pucb_intf', 'netName': 'wo_cb_consume', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': 1},
            {'modName': 'pucb_intf', 'netName': 'wo_cb_vtp_offset', 'netType': 'wire', 'attribute': '(* keep = \"true\" *)', 'width': self.offs_w}
        ]
    
    # Use the get methods to get a copy of the templates above
    def get_params(self):
        return self._params

    def get_regs(self):
        return self._regs

    def get_nets(self):
        return self._nets
    
    def get_dbg_ports(self):
        return self._dbg_ports
    
    def put_params(self, wb_val):
        self._params = wb_val

    def put_regs(self, wb_val):
        self._regs = wb_val
    
    def put_nets(self, wb_val):
        self._nets = wb_val

def gen_debug():
    # The make file calls gen_debug() from the 'tb' dir
    cwd = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('-make', action='store', default=cwd, dest='make', help='Cocotb build makefile')
    parser.add_argument('-hdl_src', action='store', default=cwd, dest='hdl_src', help='Source directory for HDL files')
    parser.add_argument('-hdl_dest', action='store', default=cwd, dest='hdl_dest', help='Destination directory of generated HDL')
    parser.add_argument('-sdc_src', action='store', default=cwd, dest='sdc_src', help='Source directory for Constraint files')
    parser.add_argument('-sdc_dest', action='store', default=cwd, dest='sdc_dest', help='Destination directory of generated Constraint')
    parser.add_argument('-e, --extension', action='store', default='xdc', dest='sdc_extn', help='Constraint files extension: xdc (default), sdc')
    parser.add_argument('-tmpl_src', action='store', default=cwd, dest='tmpl_src', help='Templates directory for tcl generation')
    parser.add_argument('-tmpl_dest', action='store', default=cwd, dest='tmpl_dest', help='Destination directory for generated tcl')
    parser.add_argument('-tmpl_name', action='store', default=cwd, dest='tmpl_name', help='Template filename')
    parser.add_argument('-run_name', action='store', default='synth_0', dest='run_name', help='Output directory name for run')
    parser.add_argument('-synth', action='store_true', dest='synth_en', help='Enable design synthesis this run')
    parser.add_argument('-place', action='store_true', dest='place_en', help='Enable design placement this run')
    parser.add_argument('-route', action='store_true', dest='route_en', help='Enable design routing this run')
    parser.add_argument('-bitstream', action='store_true', dest='bitstream_en', help='Enable design bitstream this run')
    parser.add_argument('-wsl', action='store_true', dest='wsl_mode', help='Running in WSL, convert paths for write')
    parser.add_argument('top', action='store', help='Top module name')
    parser.add_argument('part', action='store', help='Part number FPGA board')
    args = parser.parse_args()

    # Create config
    def_params = default_params()
    cfg_params = def_params.get_params()
    cfg = config(cfg_params)
    dbg_nets = cfg.get_dbg_ports()
    dbg_params = cfg.get_params()
    
    # Populate Vivado debug run directory
    
    # Build tree from HDL
    design_tree = cfg_util.modTree(args.top, args.hdl_src, compile_unit='systemverilog')
    # Print tree
    design_tree.print_tree()
    # Set hooks for nodes/modules
    design_tree.set_hooks()
    # call KEEP pass and create debug ports
    cfg_util.debug_net_pass(design_tree, dbg_nets)
    # call raise pass to expose internal debug ports to top module
    cfg_util.debug_raise_pass(design_tree)
    # Writeback HDL
    design_tree.write(args.hdl_dest)
    # Get constraints
    cfg_util.gen_sdc(args.top, args.sdc_src, args.sdc_dest, args.sdc_extn)
    # Gen run templates
    cfg_util.gen_script(args.top, args.part, args.run_name, args.synth_en, args.place_en, args.route_en, args.bitstream_en, args.tmpl_src, dbg_nets, dbg_params, args.tmpl_dest, args.tmpl_name)
    # Add default params to cocoMake.mk as COMPILE_ARGS
    cfg_util.edit_make(args.make, dbg_params)

if __name__ == "__main__":
    gen_debug()