CC=nvcc

BIN_DIR := bin
OBJ_DIR := obj
SRC_DIR := src

PROGRAM_NAME := $(BIN_DIR)/fglt

# Set the appropriate compile/link flags
CFLAGS :=
LFLAGS :=  
GENERAL_FLAGS := -std=c++17 --compiler-options -Wall

BUILD_TYPE ?= release
BUILD_ENV ?= container

# Add flags based on BUILD_ENV (container or local)
ifeq ($(BUILD_ENV), container)
	LFLAGS += -lcusparse
else ifeq ($(BUILD_ENV), local)
	LFLAGS += -lcusparse_static -lculibos
endif

## Add flags based on BUILD_TYPE (release or debug)
ifeq ($(BUILD_TYPE), release)
	GENERAL_FLAGS := -O3 $(GENERAL_FLAGS)
else ifeq ($(BUILD_TYPE), debug)
	BIN_DIR := ${BIN_DIR}_debug
	OBJ_DIR := ${OBJ_DIR}_debug
	GENERAL_FLAGS :=  -g -G -O0 -DDEBUG $(GENERAL_FLAGS)
endif

# Initiate all variables needed for building
EXEC_SRC := $(wildcard $(SRC_DIR)/*.cu $(SRC_DIR)/*.cpp)

EXEC_OBJ_TEMP = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(EXEC_SRC))
EXEC_OBJ := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(EXEC_OBJ_TEMP))

EXEC_BIN_TEMP = $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%, $(EXEC_SRC))
EXEC_BIN := $(patsubst $(SRC_DIR)/%.cpp, $(BIN_DIR)/%, $(EXEC_BIN_TEMP))

COMMON_SRC := $(wildcard $(SRC_DIR)/common/*.cu $(SRC_DIR)/common/*.cpp)
                
COMMON_OBJ_TEMP = $(patsubst $(SRC_DIR)/common/%.cu, $(OBJ_DIR)/common/%.o, $(COMMON_SRC))
COMMON_OBJ := $(patsubst $(SRC_DIR)/common/%.cpp, $(OBJ_DIR)/common/%.o, $(COMMON_OBJ_TEMP))

.SECONDARY: $(COMMON_OBJ) $(EXEC_OBJ) #Added this so that .o files aren't deleted

DEPS := $(wildcard $(SRC_DIR)/common/*.hpp $(SRC_DIR)/common/*.cuh)

# Actual build target
target: $(EXEC_BIN)

# CUDA object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(DEPS)
	@mkdir -p '$(@D)'
	@echo COMPILING $< TO $@
	$(CC) -c -o $@ $< $(CFLAGS) $(GENERAL_FLAGS)

# C++ object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(DEPS)
	@mkdir -p '$(@D)'
	@echo COMPILING $< TO $@
	$(CC) -c -o $@ $< $(CFLAGS) $(GENERAL_FLAGS)

# Link object files to create executables
$(EXEC_BIN): $(BIN_DIR)/%: $(OBJ_DIR)/%.o $(COMMON_OBJ)
	@mkdir -p '$(@D)'
	@echo LINKING $^ TO $@
	$(CC) -o $@ $^ $(LFLAGS) $(GENERAL_FLAGS)

run: ${PROGRAM_NAME}
	./${PROGRAM_NAME} ${ARGS}

.PHONY: clean print

clean:
	rm -rf $(BIN_DIR)/* $(OBJ_DIR)/* 

print:
	@echo EXEC_SRC: $(EXEC_SRC)
	@echo EXEC_OBJ: $(EXEC_OBJ)
	@echo EXEC_BIN: $(EXEC_BIN)
	@echo COMMON_SRC: $(COMMON_SRC)
	@echo COMMON_OBJ: $(COMMON_OBJ)