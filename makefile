BUILD_DIR := ./build
SRC_DIR := ./src

INC_DIRS := $(shell find $(SRC_DIR) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

SRCS := $(shell find $(SRC_DIR) -name '*.c')
OBJS := $(SRCS:./src/%.c=$(BUILD_DIR)/%.o)

all:
	echo $(OBJS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	mkdir -p $(dir $@)
	gcc -O2 $(INC_FLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)
