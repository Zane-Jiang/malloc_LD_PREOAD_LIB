# malloc_LD_PREOAD_LIB
This repo is intended to probe the heap varialbes memory access.
we can record the [adrress \ time \ size] of heap variables while the process allocate the variables and release it.

Currently suport C++ \Fortran and C lang.

## Usage:
#### Building:
  ```bash
    git clone https://github.com/ljx0525/malloc_LD_PREOAD_LIB.git
    cd malloc_LD_PREOAD_LIB/src
    make
  ```
#### Usage of LD_PRELOAD environment variable:
    export LD_PRELOAD=$LD_PRELOAD:$(pwd)/lib/libMallocHook.so

### sora
  `sora` dir copy from sora, has greater perfomance in mult-thread environment than mine。

