import logging
# COCOTB related
import cocotb
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly
# COSIM related
from tb.cosim.bandit_old_master import bandit
# TB related
from .pucb_intf_utils import env, Cov

## Global Defines
test_addr_range = 400
test_addr_stride = 100

## Main pucb_intf test
@cocotb.test()
async def pucb_intf_test(dut):

    ## Test params
    logger_name = 'pucb_intf_tb'
    run_count = 7
    
    ## Setup TB
    tb = env(dut, logger_name, log_level=logging.DEBUG)
    params = tb.get_params()
    cfg = tb.get_config()
    cov = Cov(test_addr_range, logger_name)
    ## Get cosim object(s)
    csim = bandit(0, cfg, logger_name)
    
    ## Reset
    csim.reset()
    csim_consume_buf = csim.get_reg('r_cb_page_rd_buffer')
    tb.support(csim_consume_buf)
    #await tb.reset(cycles=3, unit=params['CLK_UNIT'])
    tb.reset(cycles=3, unit=params['CLK_UNIT'])

    ## Run test for run_count cycles
    for rcnt in range(run_count):
        # Verilator forces all valueChange triggers (ValueChange, Rising, Falling) to ReadWrite() phase

        # Since the dut monitor starts sampling from the first falling edge,
        # the TB must also begin operation from the first falling edge.
        await FallingEdge(tb.pu_clk)

        ## Wait for next cycle
        await RisingEdge(tb.pu_clk)
        ## Run cosim check stage
        # corresponding DUT scores collected by monitor on negedge (live-value)
        csim.run(tb.oaddr.value.integer, tb.oaddr_wen.value.integer, tb.oretry.value.integer, tb.cb_en.value.integer, phase="check")
        ## Run cosim writeback stage
        # corresponding DUT scores collected by monitor on posedge (registerd-value)
        csim.run(tb.oaddr.value.integer, tb.oaddr_wen.value.integer, tb.oretry.value.integer, tb.cb_en.value.integer, phase="writeback")

        ## Cosim Scoreboard
        csim_scoreboard = csim.get_scoreboard()
        #tb.score(csim_scoreboard)
        
        ## All stimulus generated here can be directly applied to the DUT (use appropriately) | in ReadWrite phase
        # Get support values from Cosim model and update them at clock-edge to emulate registered vals
        csim_consume_buf = csim.get_reg('r_cb_page_rd_buffer')
        tb.support(csim_consume_buf)

        ## Perform coverage
        cov.dut_in(tb.oaddr.value, tb.oaddr_wen.value, tb.oretry.value, tb.cb_en.value)
        cov.report()

        ## Go to End of time step to prepare next cycle
        await ReadOnly()

        ## Enter ReadOnly phase
        # All stimulus generated here must pass through the corresponding drivers
        # No direct assertion of stimulus is possible from here in the ReadOnly phase
        # Hence the packets are passed to drivers through queues, who then send out
        # the values to DUT during the beginning-of-time-step (BTS) phase of the next time step (clk-to-q delay)
        # This ensures:
        #   -> The monitor agent reads all signals at the ReadOnly phase of current step, i.e., after clk_edge and at End-of-Time Step
        #   -> The transmitter agent sends out packets built during ETS in the BTS phase of next time step set by clk-to-q delay

        ## Call stimulator
        if (tb.rst.value.integer == 0):
            tb.stimulate(addr_range=test_addr_range, max_stride=test_addr_stride)
            tb.log.debug(f'{pucb_intf_test.__name__} CYCLE[{rcnt}] ||| iaddr = {tb.iaddr.value.integer}, iaddr_wen = {tb.iaddr_wen.value.integer}, imiss = {tb.imiss.value.integer} | oaddr = {tb.oaddr.value.integer}, cb_en = {tb.cb_en.value.integer}')
            tb.log.debug(f'{pucb_intf_test.__name__} CYCLE[{rcnt}] ||| cb_consume = {tb.cb_consume.value.integer}, mem_addr = {tb.mem_addr.value.integer}, mem_wen = {tb.mem_wen.value.integer}')
