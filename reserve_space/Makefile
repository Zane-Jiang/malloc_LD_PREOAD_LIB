CC = gcc
CFLAGS = -Wall -Wextra -fPIC -O2
LDFLAGS = -shared
LIB_DIR = ../lib
TARGET = libreserve_space.so
SRC = reserve_space.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o ${LIB_DIR}/$@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
