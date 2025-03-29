# Model Activation Visualizer Makefile

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -D_DEFAULT_SOURCE
LDFLAGS = -lm -lpthread

# GGML/llama.cpp paths - adjust these to match your setup
GGML_DIR = ../../llama.cpp
INCLUDE_DIRS = -I$(GGML_DIR)
LIB_DIRS = -L$(GGML_DIR)/build
LIBS = -lggml -lllama

# Source files
SRCS = mav.c
OBJS = $(SRCS:.c=.o)
EXECUTABLE = mav

# Define the main executable
$(EXECUTABLE): $(OBJS)
	$(CC) $(CFLAGS) -DMAV_MAIN_INCLUDED -o $@ $^ $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $(LDFLAGS)

# Pattern rule for object files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -c -o $@ $<

# Build as a library
libmav.a: $(OBJS)
	ar rcs $@ $^

# Testing targets
test: test.c libmav.a
	$(CC) $(CFLAGS) -o test test.c -L. -lmav $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $(LDFLAGS)
	./test

# Clean build files
clean:
	rm -f $(EXECUTABLE) $(OBJS) libmav.a test

# Install
install: $(EXECUTABLE)
	mkdir -p $(DESTDIR)/usr/local/bin
	cp $(EXECUTABLE) $(DESTDIR)/usr/local/bin/
	chmod 755 $(DESTDIR)/usr/local/bin/$(EXECUTABLE)

# Uninstall
uninstall:
	rm -f $(DESTDIR)/usr/local/bin/$(EXECUTABLE)

# Default target
all: $(EXECUTABLE)

.PHONY: all clean install uninstall test