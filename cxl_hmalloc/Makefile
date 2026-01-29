SRCS := $(wildcard *.c)
OBJS := $(SRCS:.c=.o)

CFLAGS=-Wall -g -ggdb3 -O0 -lnuma -std=c++11
LDFLAGS=-lnuma
LIB_DIR = ../lib
.PHONY: all clean
all: makefile.dep libCXLMalloc.so

makefile.dep: $(SRCS) $(wildcard *.h)
	for i in $(SRCS); do g++ -MM "$${i}" ${CFLAGS}; done > $@

-include makefile.dep

libCXLMalloc.so: $(OBJS)
	g++ -shared -Wl,-soname,libCXLMalloc.so -o $(LIB_DIR)/$@ $(OBJS) -ldl -lpthread -lnuma -ljemalloc

%.o: %.c
	g++ -fPIC ${CFLAGS} -c $< -o $@

clean:
	rm -f *.o *.so makefile.dep
