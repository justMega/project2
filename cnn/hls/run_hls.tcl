# Tcl flow (classic vitis_hls, still accepted by Vitis 2026.1):
#   vitis-run --mode hls --tcl run_hls.tcl        (2024.2+)
#   vitis_hls -f run_hls.tcl                      (older versions)
# Set RUN_COSIM to 1 for RTL co-simulation (several minutes).
set RUN_COSIM 0

open_project -reset ds_cnn_prj
set_top ds_cnn
add_files ds_cnn.cpp
add_files ds_cnn.h
add_files ds_cnn_weights_data.h
add_files -tb ds_cnn_tb.cpp
add_files -tb tb_data

open_solution -reset sol1 -flow_target vivado
set_part xc7z020clg400-1
create_clock -period 10 -name default

csim_design
csynth_design
if {$RUN_COSIM} { cosim_design -trace_level none }
export_design -format ip_catalog -vendor sdu -library hls -version 1.0
exit
