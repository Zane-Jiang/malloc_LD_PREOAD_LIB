#compile malloc_hook.c
CFLAGS=-Wall -g -ggdb3 -O0 -lnuma -std=c++11
LDFLAGS=-lnuma
LIB_DIR = ../lib

.PHONY: all clean
all: makefile.dep libMallocHook.so


makefile.dep: *.[Cch]
	for i in *.[Cc]; do gcc -MM "$${i}" ${CFLAGS}; done > $@

-include makefile.dep

libMallocHook.so: malloc_hook.c
	mkdir -p $(LIB_DIR)
	g++ -fPIC ${CFLAGS} -c malloc_hook.c
	g++ -shared -Wl,-soname,libpmalloc.so -o $(LIB_DIR)/$@ malloc_hook.o -ldl -lpthread -lnuma

clean:
	rm -f *.o *.so makefile.dep