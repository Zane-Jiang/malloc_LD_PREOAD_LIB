CFLAGS=-Wall -g -ggdb3 -O0 -lnuma -std=c++11
LDFLAGS=-lnuma
LIB_DIR = ../lib
.PHONY: all clean
all: makefile.dep libCXLMalloc.so

makefile.dep: *.[Cch]
	for i in *.[Cc]; do gcc -MM "$${i}" ${CFLAGS}; done > $@

-include makefile.dep

libCXLMalloc.so: cxl_malloc.c
	g++ -fPIC ${CFLAGS} -c cxl_malloc.c
	g++ -shared -Wl,-soname,libCXLMalloc.so -o $(LIB_DIR)/$@ cxl_malloc.o -ldl -lpthread -lnuma

clean:
	rm -f *.o *.so makefile.dep
	