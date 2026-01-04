SRCS := cxl_malloc.c 
OBJS := $(SRCS:.c=.o)

CFLAGS=-Wall -g -ggdb3 -O0 -lnuma 
LIB_DIR = ../lib
.PHONY: all clean
all: makefile.dep libCXLMalloc.so

makefile.dep: $(SRCS) $(wildcard *.h)
	for i in $(SRCS); do gcc -MM "$${i}" ${CFLAGS}; done > $@

-include makefile.dep

libCXLMalloc.so: $(OBJS)
	gcc -shared -Wl,-soname,libCXLMalloc.so -o $(LIB_DIR)/$@ $(OBJS) -ljemalloc	-ldl -lpthread -lnuma

%.o: %.c
	gcc -fPIC ${CFLAGS} -c $< -o $@

clean:
	rm -f *.o *.so makefile.dep
