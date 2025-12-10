
set TOP {{ ctxt.top }}
set PART {{ ctxt.part }}
set RUN_NAME {{ ctxt.run_name }}
set SYNTH_EN {{ ctxt.synth_en }}
set PLACE_EN {{ ctxt.place_en }}
set ROUTE_EN {{ ctxt.route_en }}
set BITSTREAM_EN {{ ctxt.bitstream_en }}

# Vivado Init
create_project -in_memory -part $PART

# This script must be placed within the local-vivado (in-memory) project DIRECTORY.
# In this flow $CWD always points to where this script is sourced from
set CWD [get_property DIRECTORY [current_project]]
# Output directory for run generated files
set ODIR $CWD/runs/$RUN_NAME
# Custom tcl scripts can be hooked into the template.
# The custom scripts can be placed in the TCL_INCLUDE dir.
set TCL_INCLUDE $CWD/tcl
# All HDL source files for the current run are placed in HDL_INCLUDE dir.
set HDL_INCLUDE $CWD/hdl
# All constraint files for the current run are placed in SDC_INCLUDE dir.
set SDC_INCLUDE $CWD/constraints

# Run Parameters for Top module
set PU_CLK_PERIOD {{ ctxt.dbg_params.PU_CLK_PERIOD }}
set CB_CLK_PERIOD {{ ctxt.dbg_params.CB_CLK_PERIOD }}
#set BRAM_EN {{ ctxt.dbg_params.BRAM_EN }}
set ASSOC {{ ctxt.dbg_params.ASSOC }}
set SETS {{ ctxt.dbg_params.SETS }}
set ADDR_W {{ ctxt.dbg_params.ADDR_W }}
set DATA_W {{ ctxt.dbg_params.DATA_W }}
set PAGE_SIZE {{ ctxt.dbg_params.PAGE_SIZE }}
set PIPE_DEP {{ ctxt.dbg_params.PIPE_DEP }}
set DEC_DEP {{ ctxt.dbg_params.DEC_DEP }}

proc get_dbg_nets {moduleName netName netWidth} {
    if {$netWidth > 1} {
        set ret_val {}
        set lst [get_nets -hierarchical "$netName[*]" -filter {PARENT_CELL =~ "*$moduleName*" || PARENT_CELL =~ ""}]
        # Sort the list (ascending)
        for {set x 0} {$x < $netWidth} {incr x} {
            set ind [lsearch -exact $lst "$netName[$x]"]
            set lval [lindex $lst $ind]
            lappend ret_val $lval
        }
        return $ret_val
    } else {
        set ret_val [get_nets -hierarchical "$netName" -filter {PARENT_CELL =~ "*$moduleName*" || PARENT_CELL =~ ""}]
        return $ret_val
    }
}

file mkdir $ODIR
file mkdir $TCL_INCLUDE

#
# setup design sources and constraints for active run's file_set
#
read_verilog -sv [glob -nocomplain $HDL_INCLUDE/*.sv]
# SystemVerilog and Verilog have different compilation units
#read_verilog [glob -nocomplain $HDL_INCLUDE/*.v]
# Read active run's constraints
read_xdc [glob -nocomplain $SDC_INCLUDE/*.xdc]

#
# Run synthesis, report utilization and timing estimates, write checkpoint design
#
if {$SYNTH_EN} {
    synth_design -top $TOP -part $PART -generic PU_CLK_PERIOD=$PU_CLK_PERIOD -generic CB_CLK_PERIOD=$CB_CLK_PERIOD \
                 -generic ASSOC=$ASSOC -generic SETS=$SETS -generic ADDR_W=$ADDR_W \
                 -generic DATA_W=$DATA_W -generic BLOCK_W=$BLOCK_W -generic PIPE_DEP=$PIPE_DEP -generic DEC_DEP=$DEC_DEP -debug_log
    write_checkpoint -force $ODIR/synth_dcp
    report_timing_summary -file $ODIR/synth_timing_summary.rpt
    report_power -file $ODIR/synth_power.rpt
    report_utilization -hierarchical -file $ODIR/synth_utilization.rpt
    
    #----------------------------------------------------------------------------------
    # Post Synth commands START here
    #----------------------------------------------------------------------------------
    
    # Setup debug ports
    set dbg_port_name [list \
        {% for module in ctxt.dbg_ports %}
        {% set dbg_nets = module.nets %}
        {% for net in dbg_nets %}
        "dbg_{{ module.modName }}_{{ net.netName }}" \
        {% endfor %}
        {% endfor %}
    ]
    set dbg_port_width [list \
        {% for module in ctxt.dbg_ports %}
        {% set dbg_nets = module.nets %}
        {% for net in dbg_nets %}
        "{{ net.width }}" \
        {% endfor %}
        {% endfor %}
    ]
    # Create debug ports
    foreach pname $dbg_port_name pwidth $dbg_port_width {
        set bus_msb [expr {$pwidth - 1}]
        create_port -direction OUT -from 0 -to $bus_msb $pname
    }

    # Setup debug nets
    set dbg_bus [list \
        {% for module in ctxt.dbg_ports %}
        {% set dbg_nets = module.nets %}
        {% for net in dbg_nets %}
        [get_dbg_nets "{{ module.modName }}" "{{ net.netName }}" "{{ net.width }}"] \
        {% endfor %}
        {% endfor %}
    ]
    # Connect debug nets
    set dbg_port_cnt [llength $dbg_port_name]
    for {set x 0} {$x < $dbg_port_cnt} {incr x} {
        set pname [lindex $dbg_port_name $x]
        set pwidth [lindex $dbg_port_width $x]
        set bus [lindex $dbg_bus $x]
        for {set y 0} {$y < $pwidth} {incr y} {
            set Net [lindex $bus $y]
            set objs [lindex [get_ports $pname] $y]
            if {$Net != {}} {
                connect_net -net $Net -objects $objs
            } else {
                puts "WARNING: run.tcl : Port $objs left unconnected due to absence of driving net."
            }
        }
    }

    #----------------------------------------------------------------------------------
    # Post Synth commands END here
    #----------------------------------------------------------------------------------

    write_verilog -mode funcsim -include_xilinx_libs -force [join [list "$ODIR" {/} "$TOP" {_vivado_synth_netlist.sv}] ""]
}

#
# Run placement and logic optimzation, report utilization and timing estimates, write checkpoint design
#
if {$PLACE_EN} {
    opt_design
    place_design
    phys_opt_design
    write_checkpoint -force $ODIR/place_dcp
    report_timing_summary -file $ODIR/place_timing_summary.rpt
}

#
# Run router, report actual utilization and timing, write checkpoint design, run drc, write verilog and xdc out
#
if {$ROUTE_EN} {
    route_design
    write_checkpoint -force $ODIR/route_dcp
    report_timing_summary -file $ODIR/route_timing_summary.rpt
    report_timing -sort_by group -max_paths 100 -path_type summary -file $ODIR/route_timing.rpt
    report_clock_utilization -file $ODIR/clock_util.rpt
    report_utilization -file $ODIR/post_route_util.rpt
    report_power -file $ODIR/post_route_power.rpt
    report_drc -file $ODIR/post_imp_drc.rpt
    write_verilog -force $ODIR/$TOP_impl_netlist.v
    write_xdc -no_fixed_only -force $ODIR/$TOP_impl.xdc
}

#
# Generate a bitstream
#
if {$BITSTREAM_EN} { 
    write_bitstream -force $ODIR/$TOP.bit
}
{# (End template) #}