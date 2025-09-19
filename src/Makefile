#compile malloc_hook.c

CC = gcc
CFLAGS = -Wall -Wextra -Werror -fPIC -ldl -g
LDFLAGS = -shared


LIB_DIR = ../lib
all: libMallocHook.so

libMallocHook.so: malloc_hook.c
	mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) -o $(LIB_DIR)/$@ $< $(LDFLAGS)

clean:
	rm $(LIB_DIR)/libMallocHook.so