# ============================================================================
# build_cnn_bd.tcl - Vivado project + block design for the CNN accelerator
#
#   Zynq PS7 (UART0, DDR) --M_AXI_GP0--> SmartConnect --> ds_cnn s_axi_control
#   ds_cnn m_axi_gmem --> SmartConnect --> PS7 S_AXI_HP0 (DDR)
#   FCLK_CLK0 = 100 MHz for the PL
#
# Usage (from this folder, after the HLS IP has been exported):
#   vivado -mode batch -source build_cnn_bd.tcl -tclargs ../hls
# or in the Vivado Tcl console:  set argv ../hls ; source build_cnn_bd.tcl
#
# Produces: vivado_cnn/ project, shape_cnn.xsa (with bitstream) and
#           reports/{utilization,power,timing}.rpt
# Needs the PYNQ-Z2 board files: installed automatically from the Xilinx board
# store if missing (internet needed); the script stops if that fails.
# ============================================================================
set hls_dir [expr {[llength $argv] > 0 ? [lindex $argv 0] : "../hls"}]
set here [file normalize [file dirname [info script]]]
cd $here

# ---- locate the exported HLS IP (component.xml) below hls_dir
proc find_ip {dir depth} {
    if {$depth < 0} { return "" }
    if {[file exists [file join $dir component.xml]]} { return $dir }
    foreach d [glob -nocomplain -type d -directory $dir *] {
        set r [find_ip $d [expr {$depth - 1}]]
        if {$r ne ""} { return $r }
    }
    return ""
}
set ip_dir [find_ip [file normalize $hls_dir] 7]
if {$ip_dir eq ""} {
    error "No exported HLS IP (component.xml) found below $hls_dir. Run the HLS package/export step first."
}
puts "HLS IP found in $ip_dir"

# ---- project
# Boards installed from the Vivado Store live in the user's xhub folder, which
# batch mode does not always search: add it (and any XILINX_BOARD_PATH) to the
# board repository paths before the project is created.
set vver [version -short]
set board_dirs [list [file join $::env(HOME) .Xilinx Vivado $vver xhub board_store \
                          xilinx_board_store XilinxBoardStore Vivado $vver boards]]
if {[info exists ::env(XILINX_BOARD_PATH)]} { lappend board_dirs $::env(XILINX_BOARD_PATH) }
set repo [get_param board.repoPaths]
foreach d $board_dirs {
    if {[file isdirectory $d] && [lsearch -exact $repo $d] < 0} { lappend repo $d }
}
set_param board.repoPaths $repo
puts "Board repository paths: $repo"

create_project shape_cnn ./vivado_cnn -part xc7z020clg400-1 -force

# The PS7 preset (DDR3 timing, UART0 on MIO 14-15, clocks) comes from the
# PYNQ-Z2 board files. Without them the bitstream builds but the board will
# not run the application, so install them or stop.
proc pynq_board_part {} {
    set bp [lindex [lsort [get_board_parts -quiet *pynq-z2*]] end]
    # the pattern search can miss it: try the known name directly
    if {$bp eq ""} {
        if {![catch {set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]}]} {
            set bp tul.com.tw:pynq-z2:part0:1.0
        }
    }
    return $bp
}
if {[pynq_board_part] eq ""} {
    puts "PYNQ-Z2 board files not found: installing them from the Xilinx board store (needs internet)..."
    if {[catch {
        xhub::refresh_catalog [xhub::get_xstores xilinx_board_store]
        set items [xhub::get_xitems -quiet *pynq-z2*]
        if {[llength $items] == 0} { error "pynq-z2 not found in the board store catalog" }
        xhub::install $items
    } err]} {
        puts "Board store installation failed: $err"
    }
}
set bp [pynq_board_part]
if {$bp eq ""} {
    error "PYNQ-Z2 board files are still missing. If they were just installed, run this script again;\
           otherwise install them via Vivado GUI: Tools > Vivado Store > Boards > pynq-z2."
}
set_property board_part $bp [current_project]
puts "Using board part $bp"
set_property ip_repo_paths [list $ip_dir] [current_project]
update_ip_catalog

create_bd_design design_1

# ---- Zynq processing system
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} $ps
set_property -dict [list \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_UART0_PERIPHERAL_ENABLE {1} ] $ps

# ---- CNN accelerator
set cnn [create_bd_cell -type ip -vlnv sdu:hls:ds_cnn:1.0 ds_cnn_0]

# control: PS GP0 -> CNN AXI-Lite
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {/ps7/FCLK_CLK0 (100 MHz)} Clk_slave {/ps7/FCLK_CLK0 (100 MHz)} \
    Clk_xbar {/ps7/FCLK_CLK0 (100 MHz)} Master {/ps7/M_AXI_GP0} Slave {/ds_cnn_0/s_axi_control} \
    intc_ip {New AXI SmartConnect} master_apm {0}] [get_bd_intf_pins ds_cnn_0/s_axi_control]

# data: CNN m_axi -> PS HP0 -> DDR
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {/ps7/FCLK_CLK0 (100 MHz)} Clk_slave {/ps7/FCLK_CLK0 (100 MHz)} \
    Clk_xbar {/ps7/FCLK_CLK0 (100 MHz)} Master {/ds_cnn_0/m_axi_gmem} Slave {/ps7/S_AXI_HP0} \
    intc_ip {New AXI SmartConnect} master_apm {0}] [get_bd_intf_pins ps7/S_AXI_HP0]

assign_bd_address
validate_bd_design
save_bd_design

# ---- wrapper, synthesis, implementation, bitstream
set bd_file [get_files design_1.bd]
make_wrapper -files $bd_file -top
add_files -norecurse [file join [file dirname $bd_file] hdl design_1_wrapper.v]
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "Implementation failed, see vivado_cnn/shape_cnn.runs/impl_1/runme.log"
}

# ---- reports for the comparison table
open_run impl_1
file mkdir reports
report_utilization -hierarchical -file reports/utilization_hier.rpt
report_utilization -file reports/utilization.rpt
report_power -file reports/power.rpt
report_timing_summary -file reports/timing.rpt

# ---- hardware platform for Vitis
write_hw_platform -fixed -include_bit -force -file shape_cnn.xsa
puts "DONE: shape_cnn.xsa and reports/ written in $here"
