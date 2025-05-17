VERILATOR_FLAGS = -Wall -Wno-fatal --trace

CFLAGS = -std=c++14 -Wall

TOP = top_processing_element

SRC = processing_element.sv
TB_SRC = testbenches/tb_processing_element.cpp

OBJ_DIR = obj_dir

EXECUTABLE = $(OBJ_DIR)/$(TOP)

all: $(EXECUTABLE) run

$(EXECUTABLE): $(SRC) $(TB_SRC)
	verilator $(VERILATOR_FLAGS) --cc $(SRC) --top-module $(TOP) --exe $(TB_SRC) -o $(TOP) --CFLAGS "$(CFLAGS)"
	make -j -C $(OBJ_DIR) -f V$(TOP).mk

run: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -rf $(OBJ_DIR)

.PHONY: all run clean 