SRCS := cxl_malloc.c numa_tuner.c
OBJS := $(SRCS:.c=.o)

# FLAG5_MODE: direct  or cold (default)
FLAG5_MODE ?= cold
ifeq ($(FLAG5_MODE),cold)
  FLAG5_DEF = -DCXL_FLAG5_USE_CXL_DIRECT=0
else
  FLAG5_DEF = -DCXL_FLAG5_USE_CXL_DIRECT=1
endif

CFLAGS=-Wall -g -ggdb3 -O2 -lnuma $(FLAG5_DEF)
LIB_DIR = ../lib
.PHONY: all clean
all: makefile.dep libCXLMalloc.so

makefile.dep: $(SRCS) $(wildcard *.h)
	for i in $(SRCS); do gcc -MM "$${i}" ${CFLAGS}; done > $@

-include makefile.dep

libCXLMalloc.so: $(OBJS)
	gcc -shared -Wl,-soname,libCXLMalloc.so -o $(LIB_DIR)/$@ $(OBJS) -ldl -lpthread -lnuma

%.o: %.c
	gcc -fPIC ${CFLAGS} -c $< -o $@

clean:
	rm -f *.o *.so makefile.dep
