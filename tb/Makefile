# Makefile

# defaults
SIM ?= verilator
TOPLEVEL_LANG ?= verilog

# Logging
COCOTB_LOG_LEVEL = INFO

# Treat undefined values
COCOTB_RESOLVE_X = VALUE_ERROR

# Python profiling
COCOTB_ENABLE_PROFILING = 0

# COCOTB_TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
#COCOTB_TOPLEVEL = rand_mask_gen
COCOTB_TOPLEVEL = pucb_intf

# COCOTB_TEST_MODULES is the basename of the Python test file(s)
#COCOTB_TEST_MODULES = prmg_test
COCOTB_TEST_MODULES = test.pucb_intf_tb

# COCOTB testcases (@cocotb.test)
COCOTB_TEST_CASE = 

COCOTB_HDL_TIMEUNIT = 1ns

COCOTB_HDL_TIMEPRECISION = 1ps

MODULE = $(COCOTB_TEST_MODULES)
TOPLEVEL = $(COCOTB_TOPLEVEL)
TESTCASE = $(COCOTB_TEST_CASE)
#RANDOM_SEED = 1763170162
RANDOM_SEED = 1763308036
#--------------------------------------------------------------------
# Custom Vars
#--------------------------------------------------------------------

# Simulation type
# source	:= Sim untouched source hdl files
# debug		:= Sim generated debug hdl files
# run		:= Sim generated synthesis netlist (verilog file)
# [example]	: make sim|linter SIM_TYPE=<sim_type> 
SIM_TYPE ?= source

# Debug setup
# yosys		:= Setup debug env vars for yosys flow
# vivado	:= Setup debug env vars for vivado flow
# [example]	: make debug DEBUG_SETUP=<flow>
DEBUG_SETUP ?= yosys

# Vivado
VIVADO_BIN_WIN := C:\\Xilinx\\Vivado\\2023.2\\bin\\vivado
YOSYS_BIN := $(PWD)/../yosys

# Include pwd in python path (where first make is launched from) to search for python test modules
PYTHONPATH := $(PWD)/

# Name of the test_build dir: contains sim_build, debug_build, etc.
BUILD_NAME := pucb_intf_tb
COCO_BUILD := $(PWD)/$(BUILD_NAME)_$(SIM)_coco

# include cocotb's make rules to take care of the simulator setup
COCO_MAKEFILE_CALL := "include $(shell cocotb-config --makefiles)/Makefile.sim"

COCO_MAKEFILE_INCL := cocoMake.mk

DUMP_DIR := $(PWD)/logs

CONFIG_DIR := $(PWD)/config

DEBUG_DIR := $(COCO_BUILD)/debug

DEBUG_RUN_DIR := $(DEBUG_DIR)/runs

DEBUG_HDL_SRC := $(PWD)/../hdl

DEBUG_HDL_DST := $(DEBUG_DIR)/hdl

DEBUG_SDC_SRC := $(PWD)/../constraints

DEBUG_SDC_DST := $(DEBUG_DIR)/constraints

DEBUG_TMPL_SRC := $(PWD)/../templates

ifeq ($(DEBUG_SETUP), yosys)
	DEBUG_RUN_TMPL := dbg_yosys_run.ys
	DEBUG_ARGS += -e sdc
else
	DEBUG_RUN_TMPL := dbg_vivado_run.tcl
	DEBUG_ARGS += -e xdc
endif

DEBUG_RUN_PART := xc7z020clg400-1
DEBUG_RUN_NAME := synth

#DEBUG_ARGS += --assoc=4 --sets=32 --pageSize=16 -bram -synth -wsl
DEBUG_ARGS += -make=$(COCO_BUILD)/$(COCO_MAKEFILE_INCL) -synth -wsl
DEBUG_ARGS += -hdl_src=$(DEBUG_HDL_SRC) -hdl_dest=$(DEBUG_HDL_DST)
DEBUG_ARGS += -sdc_src=$(DEBUG_SDC_SRC) -sdc_dest=$(DEBUG_SDC_DST)
DEBUG_ARGS += -tmpl_src=$(DEBUG_TMPL_SRC) -tmpl_dest=$(DEBUG_DIR)
DEBUG_ARGS += -tmpl_name=$(DEBUG_RUN_TMPL) -run_name=$(DEBUG_RUN_NAME)

#--------------------------------------------------------------------
# Simulator Makefile will be called from within SIM_BUILD directory
#--------------------------------------------------------------------

SIM_BUILD ?= $(COCO_BUILD)/build
SIM_CMD_SUFFIX := > $(DUMP_DIR)/$(BUILD_NAME).log 2>&1
# Verilator Env
VERILATOR_ROOT ?= $(PWD)/../verilator
VERILATOR_BIN_DIR ?= $(VERILATOR_ROOT)/bin
# Verilator ARGS
# Extra_Args set by gen_debug() method in dut config file
EXTRA_ARGS += --trace --trace-fst --trace-structs --coverage

ifeq ($(SIM_TYPE), run)
	VERILOG_SRC_DIR = $(DEBUG_RUN_DIR)/$(DEBUG_RUN_NAME)
else ifeq ($(SIM_TYPE), debug)
	VERILOG_SRC_DIR = $(DEBUG_HDL_DST)
else
	VERILOG_SRC_DIR = $(DEBUG_HDL_SRC)
endif
# Set verilator_control file as the first file
VERILOG_SOURCES += $(CONFIG_DIR)/verilator_control.v
VERILOG_SOURCES += $(wildcard $(VERILOG_SRC_DIR)/*.v)
VERILOG_SOURCES += $(wildcard $(VERILOG_SRC_DIR)/*.sv)
# use VHDL_SOURCES for VHDL files

# Export all the above vars to shell for usage by cocotb makefiles called through include
export

# Verilator_linter
VLINT_CMD = $(VERILATOR_BIN_DIR)/verilator --lint-only
VLINT_ARGS += -Wall -top $(COCOTB_TOPLEVEL)

# Verilator_coverage
VCOV_CMD = $(VERILATOR_BIN_DIR)/verilator_coverage
VCOV_ARGS += --annotate
VCOV_RPRT = --rank
VCOV_SRC = $(COCO_BUILD)/coverage.dat
VCOV_DST = $(COCO_BUILD)/coverage

.PHONY: sim

sim: build
	$(MAKE) -C $(COCO_BUILD) -f $(COCO_MAKEFILE_INCL)

build: dir
	sudo mkdir -p $(SIM_BUILD)

dir:
	sudo mkdir -p $(DUMP_DIR)
	sudo mkdir -p $(COCO_BUILD)
	@if [ ! -e $(COCO_BUILD)/$(COCO_MAKEFILE_INCL) ]; then \
		echo $(COCO_MAKEFILE_CALL) > $(COCO_BUILD)/$(COCO_MAKEFILE_INCL); \
	fi

vivado: debug
	cd $(DEBUG_DIR); \
	wslpath -w $(DEBUG_DIR)/$(COCOTB_TOPLEVEL)_run.tcl | sed 's/\\/\\\\/g' | xargs -p -I{} \
	powershell.exe $(VIVADO_BIN_WIN) -mode batch -source {}

yosys: debug
	$(YOSYS_BIN)/yosys -l $(DUMP_DIR)/yosys.log -s $(DEBUG_DIR)/$(COCOTB_TOPLEVEL)_run.ys

debug: dir
	sudo mkdir -p $(DEBUG_DIR)
	sudo mkdir -p $(DEBUG_HDL_DST)
	sudo mkdir -p $(DEBUG_SDC_DST)
	sudo mkdir -p $(DEBUG_RUN_DIR)
	python3 -m config.$(COCOTB_TOPLEVEL)_cfg $(DEBUG_ARGS) $(COCOTB_TOPLEVEL) $(DEBUG_RUN_PART) > $(DUMP_DIR)/gen_debug.log

linter: debug
	$(VLINT_CMD) $(VLINT_ARGS) $(VERILOG_SOURCES) > $(DUMP_DIR)/linter.log 2>&1

coverage: sim
	cd $(COCO_BUILD); \
	$(VCOV_CMD) $(VCOV_ARGS) $(VCOV_DST) $(VCOV_SRC); \
	$(VCOV_CMD) $(VCOV_RPRT) $(VCOV_SRC) > $(VCOV_DST)/vcov_rank.rpt

clean:
	rm -rf $(COCO_BUILD)
