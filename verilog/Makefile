VERILATOR_FLAGS = -Wall -Wno-fatal --trace

CFLAGS = -std=c++14 -Wall

TOP_PE = top_processing_element
TOP_PE_ARRAY = top_processing_element_array
TOP_PE_ARRAY_VERIFY = top_processing_element_array_verify

SRC_PE = processing_element.sv
SRC_PE_ARRAY = processing_element.sv processing_element_array.sv

TB_SRC_PE = testbenches/tb_processing_element.cpp
TB_SRC_PE_ARRAY = testbenches/tb_performance.cpp
TB_SRC_PE_ARRAY_VERIFY = testbenches/tb_verification.cpp

OBJ_DIR = obj_dir

EXECUTABLE_PE = $(OBJ_DIR)/$(TOP_PE)
EXECUTABLE_PE_ARRAY = $(OBJ_DIR)/$(TOP_PE_ARRAY)
EXECUTABLE_PE_ARRAY_VERIFY = $(OBJ_DIR)/$(TOP_PE_ARRAY_VERIFY)

SIM_FLAGS ?=

all: build_all

build_all: build_pe build_pe_array build_pe_array_verify

build_pe: $(EXECUTABLE_PE)

build_pe_array: $(EXECUTABLE_PE_ARRAY)

build_pe_array_verify: $(EXECUTABLE_PE_ARRAY_VERIFY)

run_pe: $(EXECUTABLE_PE)
	$(EXECUTABLE_PE) $(SIM_FLAGS)

run_pe_array: $(EXECUTABLE_PE_ARRAY)
	$(EXECUTABLE_PE_ARRAY) $(SIM_FLAGS)

run_pe_array_verify: $(EXECUTABLE_PE_ARRAY_VERIFY)
	$(EXECUTABLE_PE_ARRAY_VERIFY) $(SIM_FLAGS)

$(EXECUTABLE_PE): $(SRC_PE) $(TB_SRC_PE)
	verilator $(VERILATOR_FLAGS) --cc $(SRC_PE) --top-module $(TOP_PE) --exe $(TB_SRC_PE) -o $(TOP_PE) --CFLAGS "$(CFLAGS)"
	make -j -C $(OBJ_DIR) -f V$(TOP_PE).mk

$(EXECUTABLE_PE_ARRAY): $(SRC_PE_ARRAY) $(TB_SRC_PE_ARRAY)
	verilator $(VERILATOR_FLAGS) --cc $(SRC_PE_ARRAY) --top-module $(TOP_PE_ARRAY) --exe $(TB_SRC_PE_ARRAY) -o $(TOP_PE_ARRAY) --CFLAGS "$(CFLAGS)"
	make -j -C $(OBJ_DIR) -f V$(TOP_PE_ARRAY).mk

$(EXECUTABLE_PE_ARRAY_VERIFY): $(SRC_PE_ARRAY) $(TB_SRC_PE_ARRAY_VERIFY)
	verilator $(VERILATOR_FLAGS) --cc $(SRC_PE_ARRAY) --top-module $(TOP_PE_ARRAY) --exe $(TB_SRC_PE_ARRAY_VERIFY) -o $(TOP_PE_ARRAY_VERIFY) --CFLAGS "$(CFLAGS)"
	make -j -C $(OBJ_DIR) -f V$(TOP_PE_ARRAY).mk

clean:
	rm -rf $(OBJ_DIR)
	rm -f *.vcd

.PHONY: all build_all build_pe build_pe_array build_pe_array_verify run_pe run_pe_array run_pe_array_verify clean 