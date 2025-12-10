from config.pucb_intf_cfg import config, default_params
from .test_utils import tb_clock, tb_reset
from collections import deque
from cocotb_coverage import crv, coverage
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ReadOnly
import cocotb
import logging

class send_pkt(crv.Randomized):

    _prev_addr = 0

    def __init__(self, addr_range=400, max_stride=1, logger_name=''):
        super().__init__()
        self.log = logging.getLogger(logger_name)
        self.stride_range = max_stride
        self.prev_addr = self.get_prev_addr()
        #self.log.debug(f'{send_pkt.__init__.__name__} ||| Pre_randomize Prev_addr = {self.prev_addr}')

        ## Payload
        self.addr = 0
        self.wen = 0
        self.mshr = 0

        ## payload randomization setup
        # All random variables must be defined before adding any constraint with add_constraint().
        # Therefore it is highly recommended to call add_rand in the __init__ method of your final class.
        self.add_rand("addr", list(range(addr_range)))
        self.add_rand("wen", list(range(2)))

        ## Constraint function
        def c_addr(addr, prev_addr, stride_range):
            # The new addr must be within the address space (addr_range)
            # and must exist within the maximum_stride range from previous
            range_cond = True if (addr >= 0 and addr <= addr_range) else False
            stride_max = min(prev_addr+stride_range/2, addr_range)
            stride_min = max(prev_addr-stride_range/2, 0)
            ret = True if (range_cond and addr <= stride_max and addr >= stride_min) else False
            return ret

        ## Hard constraints (constraint functions must evaluate to true/false)
        self.add_constraint(c_addr)
    
    def post_randomize(self):
        # Set prev_addr = current addr
        self.set_prev_addr(self.addr)
        #self.log.debug(f'{send_pkt.__init__.__name__} ||| Post_randomize Prev_addr = {self.get_prev_addr()}')
    
    def print(self):
        self.log.info(f'Send_Packet: addr = {self.addr}, wen = {self.wen}, mshr = {self.mshr}')

    @classmethod
    def get_prev_addr(cls):
        return cls._prev_addr
    
    @classmethod
    def set_prev_addr(cls, addr):
        cls._prev_addr = addr

class recv_pkt:

    def __init__(self, addr=0, wen=0, mshr=0, logger_name=''):
        self.log = logging.getLogger(logger_name)
        # Payload
        self.addr = addr
        self.wen = wen
        self.mshr = mshr
    
    def print(self):
        self.log.info(f'Recv_Packet: addr = {self.addr}, wen = {self.wen}, mshr = {self.mshr}')

class tx:

    def __init__(self, addr, wen, mshr, cb_en, clk_q=1, unit="ns", logger_name=''):
        self.log = logging.getLogger(logger_name)
        self.job_q = deque()
        self.addr = addr
        self.wen = wen
        self.mshr = mshr
        self.cb_en = cb_en
        self.clk_q = clk_q
        self.unit = unit

    def reset(self):
        self.cb_en.value = 0
        self.addr.value = 0
        self.wen.value = 0
        self.mshr.value = 0
    
    def send(self, pkt):
        self.job_q.append(pkt)

    async def run(self, clk):
        while True:
            # Check if job queue has data
            self.cb_en.value = 0
            if (self.job_q):
                pkt = self.job_q.popleft()
                self.addr.value = pkt.addr
                self.wen.valule = pkt.wen
                self.mshr.value = pkt.mshr
                self.cb_en.value = 1
                self.log.info(f'Sent Packet: addr = {pkt.addr}, wen = {pkt.wen}, mshr = {pkt.mshr}')
            # Wait for next cycle
            await RisingEdge(clk)
            # Clock-to-Q
            await Timer(self.clk_q, self.unit)

class monitor:

    def __init__(self, dut, dbg_nets, dbg_ports, logger_name=''):

        # Changed in cocotb_2.0 to <handle>.to_unsigned() and <handle>._id(<name>) to <handle>['<name>']

        self.log = logging.getLogger(logger_name)
        ## Create a list of ports to monitor
        # For every net in dbg_nets, get alias port in dbg_ports if exists.
        # Search dut attributes/members for corresponding port;
        # If found, append to list if found.
        self.monitor_active_posedge = {}            # A dict of port handles to monitor at the posedge
        self.monitor_active_negedge = {}            # A dict of port handles to monitor at the negedge
        get_handle_path = lambda x: dut._id(x, extended=False)._path
        get_handle = lambda y: dut._id(y, extended=False)
        for net in list(dbg_nets.keys()):
            # Get net alias if any
            net_alias = [f'dbg_{p['modName']}_{p['netName']}' for p in dbg_ports if (net == p['netName'])]
            net_alias = net_alias[0] if (net_alias) else net
            self.log.debug(f'{monitor.__init__.__name__} ||| Searching DUT handle for net "{net}" with alias "{net_alias}"')
            try:
                if (get_handle_path(net_alias)):
                    # Port exists, add to dict
                    if (dbg_nets[net][1] == 1):
                        self.monitor_active_posedge[net] = get_handle(net_alias)
                    else:
                        self.monitor_active_negedge[net] = get_handle(net_alias)
                    self.log.debug(f'{monitor.__init__.__name__} ||| Adding port "{get_handle_path(net_alias)}" to monitor_active {"negedge" if (dbg_nets[net][1]) else "posedge"}')
            except AttributeError as e:
                self.log.info(f'{monitor.__init__.__name__} {e} ||| net "{net_alias}" not present in DUT handle')
        #self.log.debug(f'{monitor.__init__.__name__} ||| monitor_active = {self.monitor_active.keys()}')
    
    async def run(self, callback, clk):
        # Monitor DUT ports from monitor_active and update their values every cycle
        # Read output signals at End-of-Time-Step using ReadOnly to get updated vals.
        score = {}
        while True:
            await FallingEdge(clk)
            await ReadOnly()
            self.log.debug(f'{monitor.run.__name__} ||| Collecting falling-edge scores!')
            for m in list(self.monitor_active_negedge.keys()):
                score[m] = self.monitor_active_negedge[m].value
            await RisingEdge(clk)
            await ReadOnly()
            self.log.debug(f'{monitor.run.__name__} ||| Collecting Rising-edge scores!')
            for m in list(self.monitor_active_posedge.keys()):
                score[m] = self.monitor_active_posedge[m].value
            # Send scores through callback to validator
            callback(score)

class validator:

    def __init__(self, logger_name=""):
        self.log = logging.getLogger(logger_name)
        self.dut_score = deque()
        self.csim_score = deque()
    
    def dut_send(self, scores):
        self.log.debug(f'{validator.dut_send.__name__} ||| Validator recieved dut scores')
        self.dut_score.append(scores)
    
    def csim_send(self, scores):
        self.log.debug(f'{validator.csim_send.__name__} ||| Validator recieved csim scores')
        self.csim_score.append(scores)
    
    async def run(self, clk):
        while True:
            await RisingEdge(clk)
            await ReadOnly()
            # Validate only on the next time step since the scores are recieved in the ReadOnly phase
            # COCOTB_HDL_TIMEPRECISION determines next step
            await Timer(1)
            # Read DUT score every cycle; if CSIM does not produce scores, drop DUT scores
            if (self.dut_score):
                dscore = self.dut_score.popleft()
                self.log.debug(f'{validator.run.__name__} ||| dut_score = {dict({k: d.integer for k, d in dscore.items()})}')
                if (self.csim_score):
                    cscore = self.csim_score.popleft()
                    # Compare the scores of dut with cosim model
                    self.log.debug(f'{validator.run.__name__} ||| csim_score = {cscore}')
                    for dk in dscore.keys():
                        assert dscore[dk].integer == cscore[dk], f'Scoreboarding FAILED: dut_score[{dk}] = {dscore[dk].integer} ; csim_score[{dk}] = {cscore[dk]}'

class Cov:

    # Coverage control parameters
    _addr_range = 400               # Must be same as test_addr_range in main TB

    def __init__(self, addr_range, logger_name):
        self.log = logging.getLogger(logger_name)
        self.set_param(addr_range)

    @classmethod
    def set_param(cls, param_val):
        cls._addr_range = param_val

    def report(self):
        coverage.coverage_db.report_coverage(self.log.info, bins=True)

    ## Coverage Group 1
    @coverage.CoverPoint(
        name = "top.i_field_addr",
        vname = "addr",
        xf = lambda a, b, c, d: a.integer,
        rel = lambda val, bin_lim: bin_lim[0] <= val <= bin_lim[1],
        # _addr_range is static read here, setting it during runtime does not reflect well here
        # Hence make sure the _addr_range here and test_addr_range are the same.
        bins = [(0, int(_addr_range/3)), (int(_addr_range/3)+1, int(2*_addr_range/3)), (int(2*_addr_range/3)+1, int(3*_addr_range/3))],
        bins_labels = ["low", "med", "high"]
    )
    @coverage.CoverPoint(
        name = "top.i_field_wen",
        vname = "wen",
        xf = lambda a, b, c, d: b.integer,
        bins = [0, 1]
    )
    @coverage.CoverPoint(
        name = "top.i_field_mshr",
        vname = "mshr",
        xf = lambda a, b, c, d: c.integer,
        bins = [0, 1]
    )
    @coverage.CoverPoint(
        name = "top.i_cb_en",
        vname = "cb_en",
        xf = lambda a, b, c, d: d.integer,
        bins = [0, 1]
    )
    @coverage.CoverCross(
        name = "top.cross_i_field_addr_wen",
        items = ["top.i_field_addr", "top.i_field_wen"],
        ign_bins = [(0, 0)]
    )
    def dut_in(self, addr, wen, mshr, cb_en):
        pass

class env:

    ## ENV init
    def __init__(self, dut, logger_name, log_level=logging.INFO):

        ## Get logger
        self.log_name = logger_name
        self.log = logging.getLogger(self.log_name)
        self.log.setLevel(log_level)

        ## Get contexts
        self.dut = dut
        self.def_params = default_params()
        self.cfg_params = self.def_params.get_params()
        self.cfg = config(self.cfg_params)
        self.dbg_nets = self.cfg.get_nets()
        self.dbg_ports = self.cfg.get_dbg_ports()
        self.dbg_params = self.cfg.get_params()

        ##----------------------------------------------------------------
        ## Cosim/DUT dependent
        ##----------------------------------------------------------------

        ## Rename Pins
        self.pu_clk = self.dut.i_pu_clk
        self.cb_clk = self.dut.i_cb_clk
        self.rst = self.dut.i_rst
        # From PU/TB
        self.cb_en = self.dut.i_cb_en
        self.oretry = self.dut.i_field_mshr
        self.oaddr_wen = self.dut.i_field_wen
        self.oaddr = self.dut.i_field_addr
        # From CB cosim
        self.cb_consume_buf = self.dut.i_cb_consume_buf
        # To PU/TB
        self.imiss = self.dut.o_field_mshr
        self.iaddr = self.dut.o_field_addr
        self.iaddr_wen = self.dut.o_field_wen
        # To CB
        self.cb_consume = self.dut.o_cb_consume
        self.cb_ptag = self.dut.o_cb_ptag
        self.cb_vtp_offset = self.dut.o_cb_vtp_offset
        # To Heap
        self.mem_wen = self.dut.o_mem_wen
        self.mem_addr = self.dut.o_mem_addr
        # To TB for Scoreboarding (exposed internal nets of DUT)
        self.sb_field_stall = self.dut.dbg_pucb_intf_w_field_stall
        self.sb_field_wen = self.dut.dbg_pucb_intf_w_field_wen
        self.sb_field_mshr = self.dut.dbg_pucb_intf_w_field_mshr
        self.sb_field_atag = self.dut.dbg_pucb_intf_w_field_atag
        self.sb_field_set = self.dut.dbg_pucb_intf_w_field_set
        self.sb_field_table_vtag = self.dut.dbg_pucb_intf_w_field_table_vtag
        self.sb_field_table_wr_data = self.dut.dbg_pucb_intf_w_field_table_wr_data
        self.sb_Rfield_table_vtag = self.dut.dbg_pucb_intf_wo_field_table_vtag
        self.sb_cb_lru_cur = self.dut.dbg_pucb_intf_w_cb_lru_cur
        self.sb_cb_lru_cur_bits = self.dut.dbg_pucb_intf_w_cb_lru_cur_bits
        self.sb_cb_lru_nxt = self.dut.dbg_pucb_intf_w_cb_lru_nxt
        self.sb_cb_lru_nxt_bits = self.dut.dbg_pucb_intf_w_cb_lru_nxt_bits
        self.sb_cb_miss = self.dut.dbg_pucb_intf_w_cb_miss
        self.sb_cb_rev_ptr = self.dut.dbg_pucb_intf_w_cb_rev_ptr
        self.sb_cb_rev_ptr_null = self.dut.dbg_pucb_intf_w_cb_rev_ptr_null
        self.sb_cb_consume = self.dut.dbg_pucb_intf_w_cb_consume
        self.sb_Rcb_consume = self.dut.dbg_pucb_intf_wo_cb_consume
        self.sb_cb_cline = self.dut.dbg_pucb_intf_w_cb_cline
        self.sb_cb_ctag = self.dut.dbg_pucb_intf_w_cb_ctag
        self.sb_Rcb_ctag = self.dut.dbg_pucb_intf_wo_cb_ctag
        self.sb_cb_vtp_offset = self.dut.dbg_pucb_intf_w_cb_vtp_offset
        self.sb_cb_consume_sel = self.dut.dbg_pucb_intf_w_cb_consume_sel
        self.sb_Rcb_lru_ptag = self.dut.dbg_pucb_intf_wo_cb_lru_ptag

        ##-----------------------------------------------------------------------
        ## ENV generic methods
        ##-----------------------------------------------------------------------

        ## Get reset coroutine
        self.RST = tb_reset(self.pu_clk, self.rst, self.log_name)
        ## Create clocks
        self.PU_CLK = tb_clock(self.dbg_params['PU_CLK_PERIOD'], unit=self.dbg_params['CLK_UNIT'])
        self.CB_CLK = tb_clock(self.dbg_params['CB_CLK_PERIOD'], unit=self.dbg_params['CLK_UNIT'])
        ## Get Agents
        self.MON = monitor(self.dut, self.dbg_nets, self.dbg_ports, self.log_name)
        self.VLD = validator(self.log_name)
        self.TX = tx(self.oaddr, self.oaddr_wen, self.oretry, self.cb_en, self.dbg_params['CLK_Q'], self.dbg_params['CLK_UNIT'], self.log_name)
        ## Start Agents/Tasks
        cocotb.start_soon(self.PU_CLK.run(self.pu_clk))
        cocotb.start_soon(self.CB_CLK.run(self.cb_clk))
        cocotb.start_soon(self.MON.run(self.VLD.dut_send, self.pu_clk))
        cocotb.start_soon(self.VLD.run(self.pu_clk))
        cocotb.start_soon(self.TX.run(self.pu_clk))
    
    def set_log_level(self, log_level=logging.INFO):
        self.log.setLevel(log_level)
    
    def get_config(self):
        return self.cfg
    
    def get_params(self):
        return self.dbg_params

    ## System Reset
    def reset(self, cycles=1, period=0, unit="ns"):
        # Drive all output signals to default value
        self.TX.reset()
        # Call tb reset routine
        #await self.RST.reset(cycles, period, unit)
        cocotb.start_soon(self.RST.reset(cycles, period, unit))
    
    ## Stimulator
    def stimulate(self, addr_range=400, max_stride=100):
        # Generate new inputs for current cycle
        tx_pkt = send_pkt(addr_range, max_stride, logger_name=self.log_name)
        tx_pkt.randomize()
        # Assert new inputs for current cycle
        self.TX.send(tx_pkt)

    ## Cosim score entry
    def score(self, csim_score):
        # Send cosim model scores to validator
        self.VLD.csim_send(csim_score)
    
    ## DUT support values from Cosim model
    def support(self, csim_consume_buf):
        for vid, v in enumerate(csim_consume_buf):
            self.cb_consume_buf[vid].value = v
        self.log.debug(f'{env.support.__name__} ||| Supporting: consume_buf = {csim_consume_buf}')
