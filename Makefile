# Variables - easier to update in one place
NVCC        := nvcc
ARCH        := sm_89
CFLAGS      := -O3
LIBS        := -lcublas
BIN_DIR     := bins
TARGET      := $(BIN_DIR)/partA.bin
SRC         := partA.cu

# Default target
all: $(TARGET)

# The actual compilation rule
# $@ is a shorthand for the target (the .bin file)
# $< is a shorthand for the first dependency (the .cu file)
$(TARGET): $(SRC) | $(BIN_DIR)
	$(NVCC) -arch=$(ARCH) $(CFLAGS) $< $(LIBS) -o $@

# This creates the directory ONLY if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# "Phony" targets aren't actual files
.PHONY: all clean partA

partA: $(TARGET)

clean:
	rm -rf $(BIN_DIR)