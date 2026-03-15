# Variables
NVCC        := nvcc
CFLAGS      := -O3
LIBS        := -lcublas
BIN_DIR     := bins

# Fat Binary: Compiles for both P100 (60) and RTX 40-series (89)
# compute_XX is the virtual architecture (PTX), sm_XX is the binary (SASS)
ARCH_FLAGS  := -gencode arch=compute_60,code=sm_60 \
               -gencode arch=compute_89,code=sm_89

# Define targets explicitly based on source files
SRCS        := partA.cu partB.cu
TARGETS     := $(SRCS:%.cu=$(BIN_DIR)/%.bin)

# Default target
all: $(TARGETS)

# Pattern Rule
$(BIN_DIR)/%.bin: %.cu | $(BIN_DIR)
	$(NVCC) $(ARCH_FLAGS) $(CFLAGS) $< $(LIBS) -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

.PHONY: all clean partA partB

# Shortcuts to build specific files
partA: $(BIN_DIR)/partA.bin
partB: $(BIN_DIR)/partB.bin

clean:
	rm -rf $(BIN_DIR)