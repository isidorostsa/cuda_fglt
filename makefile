CC=nvcc

BIN_DIR := bin
OBJ_DIR := obj
SRC_DIR := src

PROGRAM_NAME := $(BIN_DIR)/fglt

# Set the appropriate compile/link flags
CFLAGS := -std=c++17
LFLAGS := -std=c++17 -lcusparse_static -lculibos

BUILD_TYPE ?= release

## Add flags based on BUILD_TYPE (release or debug)
ifeq ($(BUILD_TYPE), release)
	CFLAGS += -O3
	LFLAGS += -O3
else ifeq ($(BUILD_TYPE), debug)
	BIN_DIR := ${BIN_DIR}_debug
	OBJ_DIR := ${OBJ_DIR}_debug
	CFLAGS += -g -O0 -DDEBUG
	LFLAGS += -g -O0 -DDEBUG
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

DEPS := $(wildcard $(SRC_DIR)/common/*.hpp)

# Actual build target
do_build: $(EXEC_BIN)

# Build rules t

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(DEPS)
	@mkdir -p '$(@D)'
	$(CC) -c -o $@ $< $(CFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(DEPS)
	@mkdir -p '$(@D)'
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXEC_BIN): $(BIN_DIR)/%: $(OBJ_DIR)/%.o $(COMMON_OBJ)
	@mkdir -p '$(@D)'
	$(CC) -o $@ $^ $(LFLAGS)

run: ${PROGRAM_NAME}
	./${PROGRAM_NAME} ${ARGS}

.PHONY: clean
clean:
	rm -rf $(BIN_DIR)/* $(OBJ_DIR)/*
	@echo EXEC_SRC: $(EXEC_SRC)
	@echo EXEC_OBJ: $(EXEC_OBJ)
	@echo EXEC_BIN: $(EXEC_BIN)
	@echo COMMON_SRC: $(COMMON_SRC)
	@echo COMMON_OBJ: $(COMMON_OBJ)
