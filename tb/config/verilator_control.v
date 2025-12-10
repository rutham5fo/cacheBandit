// Custom Global verilog defines
`ifndef SIMLIB_VERILATOR
`define SIMLIB_VERILATOR
`endif

// lint rules ignore
`verilator_config

lint_off -rule GENUNNAMED
lint_off -rule WIDTHTRUNC
lint_off -rule WIDTHEXPAND
