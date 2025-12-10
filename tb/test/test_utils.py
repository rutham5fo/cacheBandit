from cocotb.triggers import Timer, RisingEdge, FallingEdge, ReadWrite, ReadOnly
import logging

class tb_clock:

    def __init__(self, period, init_delay=None, unit="ns"):
        self.period = period
        #self.init_delay = init_delay if (init_delay) else period
        self.init_delay = init_delay if (init_delay) else 0
        self.unit = unit

    # System Clock generation
    async def run(self, clk_pin):
        high_delay = self.period/2
        low_delay = self.period/2
        initial_delay = self.init_delay
        # pre-construct triggers for performance
        high_time = Timer(high_delay, units=self.unit)
        low_time = Timer(low_delay, units=self.unit)
        # Start clock low
        clk_pin.value = 0
        if (self.init_delay > 0):
            await Timer(initial_delay, units=self.unit)
        while True:
            clk_pin.value = 1
            await high_time
            clk_pin.value = 0
            await low_time

class tb_reset:

    def __init__(self, clk, rst, logger_name):
        self.log = logging.getLogger(logger_name)
        self.clk = clk
        self.rst = rst
        # Initialize to zero
        self.rst.value = 0
    
    # System Reset
    async def reset(self, cycles=1, period=0, unit="ns"):
        if (period == 0):
            for r in range(cycles):
                self.rst.value = 1
                self.log.debug(f'{tb_reset.reset.__name__} ||| Rising edge of clock | rst_cnt = {r+1}')
                await RisingEdge(self.clk)
        else:
            await Timer(period, units=unit)
        self.rst.value = 0
        # Exit reset at ReadOnly phase inorder to start TB in this phase
        #await ReadOnly()
