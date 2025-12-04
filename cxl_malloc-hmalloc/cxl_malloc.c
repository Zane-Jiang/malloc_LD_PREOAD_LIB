#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define __USE_GNU




#include "env.h"
#include <jemalloc/jemalloc.h>
#include <stdbool.h>
#include <stddef.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <execinfo.h>
#include <sys/types.h>
#include <numa.h>
#include <ctype.h>
#include <sched.h>
#include <errno.h>
#include <numaif.h>


#define NUMA_DEFAULT_NODE (-1)
#define NUMA_LOCAL_NODE 0
#define NUMA_CXL_NODE 2          /* Node to allocate CXL memory */
#define CXL_MALLOC_ENABLE_LOG 
#ifdef CXL_MALLOC_ENABLE_LOG
#define CXL_LOG(fmt, ...) printf("[cxl_malloc] " fmt "\n", ##__VA_ARGS__)
#else
#define CXL_LOG(...) ((void)0)
#endif

#define ARR_SIZE 550000            /* Max number of malloc per core */
#define MAX_TID 512                /* Max number of tids to profile */

#define USE_FRAME_POINTER   0      /* Use Frame Pointers to compute the stack trace (faster) */
#define CALLCHAIN_SIZE      5      /* stack trace length */


#define NB_ALLOC_TO_IGNORE   0     /* Ignore the first X allocations. */
#define IGNORE_FIRST_PROCESS 0     /* Ignore the first process (and all its threads). Useful for processes */

#define MAX_OBJECTS          30000

#define MAX_LINES 1000
#define MAX_NAME_LENGTH 20

#define MIN_SAFE_FREE_MEMORY (100 * 1024 * 1024)  // 100MB safety margin

static char *obj_names[MAX_LINES];
static size_t obj_retain[MAX_LINES];
static int obj_count = 0;
const uint64_t SYS_RETAIN_CAPACITY = 1000 * 1024 *1024; // systerm retain capacity


struct addr_seg {
    long unsigned start;
    long unsigned end;
};

static struct addr_seg addr_segs[MAX_OBJECTS];

void __attribute__((constructor)) m_init(void);


static int __thread _in_trace = 0;
#define get_bp(bp) asm("movq %%rbp, %0" : "=r" (bp) :)

static char empty_data[32];


#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define __unused __attribute__((unused))
#define is_pow2(val) (((val) & ((val)-1)) == 0)
/* global variables set by environment variables */
static bool use_jemalloc;
static unsigned long nodemask;
static int mpol_mode;

static unsigned arena_index;
static extent_hooks_t *hooks;

static int maxnode;



void *hmalloc(size_t size);
void hfree(void *ptr);
void *hcalloc(size_t nmemb, size_t size);
void *hrealloc(void *ptr, size_t size);
void *haligned_alloc(size_t alignment, size_t size);
int hposix_memalign(void **memptr, size_t alignment, size_t size);
void *hmmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int hmunmap(void *addr, size_t length);
size_t hmalloc_usable_size(void *ptr);


static void *(*libc_malloc)(size_t) = NULL;
static void *(*libc_calloc)(size_t, size_t) = NULL;
static void *(*libc_realloc)(void *, size_t) = NULL;
static void (*libc_free)(void *) = NULL;
static void *(*libc_mmap)(void *, size_t, int, int, int, off_t) = NULL;
static void *(*libc_mmap64)(void *, size_t, int, int, int, off_t) = NULL;
static int (*libc_munmap)(void *, size_t) = NULL;
static void *(*libc_memalign)(size_t, size_t) = NULL;
static int (*libc_posix_memalign)(void **, size_t, size_t) = NULL;



int get_trace(size_t *size, void **strings)
{
    if (_in_trace)
        return 1;
    _in_trace = 1;

#if USE_FRAME_POINTER
    int i;
    struct stack_frame *frame;
    get_bp(frame);
    for (i = 0; i < CALLCHAIN_SIZE; i++) {
        strings[i] = (void*)frame->return_address;
        *size = i + 1;
        frame = frame->next_frame;
        if (!frame)
            break;
    }
#else
    int depth = backtrace(strings, CALLCHAIN_SIZE);
    if (depth < 0)
        depth = 0;
    *size = (size_t)depth;
#endif
    _in_trace = 0;
    return 0;
}

static long long get_node_free_bytes(int node)
{
    long long free_bytes = 0;
    if (numa_node_size64(node, &free_bytes) < 0)
        return -1;
    return free_bytes;
}

// Get physically allocated memory for a NUMA node from /proc/numa_maps
static long long get_node_physical_used_bytes(int node)
{
    FILE *fp;
    char line[256];
    long long used_bytes = 0;
    
    fp = fopen("/proc/self/numa_maps", "r");
    if (!fp)
        return -1;
    
    while (fgets(line, sizeof(line), fp)) {
        unsigned long addr;
        int pages_node;
        char heap_stack[32];
        
        if (sscanf(line, "%lx prefer:%d %31s N=%*d %*d", &addr, &pages_node, heap_stack) == 3) {
            if (pages_node == node) {
                // Count pages on this node
                char *p = strchr(line, 'N');
                if (p) {
                    int node_id, pages_count;
                    if (sscanf(p, "N=%d %d", &node_id, &pages_count) == 2 && node_id == node) {
                        used_bytes += (long long)pages_count * 4096;  // 4KB pages
                    }
                }
            }
        }
    }
    fclose(fp);
    return used_bytes;
}

// More accurate memory check considering lazy allocation
static long long get_node_available_bytes(int node, int exclude_node)
{
    long long total_bytes = 0;
    long long used_bytes = 0;
    
    // Get total node memory
    if (numa_node_size64(node, &total_bytes) < 0)
        return -1;
    
    // Get physically allocated memory on this node
    used_bytes = get_node_physical_used_bytes(node);
    if (used_bytes < 0)
        used_bytes = 0;
    
    long long available = total_bytes - used_bytes;
    
    // Add extra safety margin to account for:
    // 1. Memory that will be allocated but not yet paged
    // 2. System memory needed for page tables and metadata
    long long safety_margin = MIN_SAFE_FREE_MEMORY + (total_bytes / 20);  // 5% extra buffer
    
    return available - safety_margin;
}


int get_numa_node(void *string, size_t sz)
{
    if (numa_available() < 0)
        return NUMA_DEFAULT_NODE;

    const char *symbol = (const char *)string;
    const char *start = strrchr(symbol, '[');
    const char *end = strrchr(symbol, ']');
    if (!start || !end || end <= start + 1) {
        start = symbol;
        end = symbol + strlen(symbol);
    } else {
        start++;
    }
    while (start < end && isspace((unsigned char)*start)) start++;
    while (end > start && isspace((unsigned char)*(end - 1))) end--;
    size_t addr_len = (size_t)(end - start);
    if (addr_len >= MAX_NAME_LENGTH) addr_len = MAX_NAME_LENGTH - 1;

    char addr_buf[MAX_NAME_LENGTH];
    memcpy(addr_buf, start, addr_len);
    addr_buf[addr_len] = '\0';

    if (addr_buf[0] == '\0') {
        return NUMA_DEFAULT_NODE;
    }

    for (int i = 0; i < obj_count; i++) {
        if (strcmp(addr_buf, obj_names[i]) == 0) {
            static int local_node = NUMA_LOCAL_NODE;
            
            long long local_available = get_node_available_bytes(local_node, NUMA_CXL_NODE);
            if (local_available < 0)
                return NUMA_DEFAULT_NODE;
            
            // Check if local node has enough space (considering actual physical usage)
            if (local_available > (long long)sz) {
                return NUMA_DEFAULT_NODE;
            }

            // Local node doesn't have enough, try CXL
            long long cxl_available = get_node_available_bytes(NUMA_CXL_NODE, local_node);
            if (cxl_available < 0)
                return NUMA_DEFAULT_NODE;
            
            if (cxl_available > (long long)sz) {
                CXL_LOG("Allocating %zu bytes to CXL node (available: %lld)", sz, cxl_available);
                return NUMA_CXL_NODE;
            }
            
            // Both nodes under pressure, fallback to default
            CXL_LOG("Both nodes under pressure (local: %lld, CXL: %lld), using default", 
                    local_available, cxl_available);
            return NUMA_DEFAULT_NODE;
        }
    }
    return NUMA_DEFAULT_NODE;
}


void record_seg(unsigned long addr, size_t size)
{
    int i;
    for (i = 0; i < MAX_OBJECTS; i += 1) {
        struct addr_seg *seg = &addr_segs[i];
        if (seg->start == 0 && seg->end == 0) {
            seg->start = addr;
            seg->end = addr + size;
            return;
        }
    }
}

static struct addr_seg *find_seg(unsigned long addr)
{
    for (int i = 0; i < MAX_OBJECTS; ++i) {
        if (addr_segs[i].start == addr)
            return &addr_segs[i];
    }
    return NULL;
}

static inline size_t seg_length(const struct addr_seg *seg)
{
    return seg ? (size_t)(seg->end - seg->start) : 0;
}

static inline void clear_seg(struct addr_seg *seg)
{
    if (seg) {
        seg->start = 0;
        seg->end = 0;
    }
}

size_t check_seg(unsigned long addr)
{
    struct addr_seg *seg = find_seg(addr);
    size_t size_to_free = seg_length(seg);
    clear_seg(seg);
    return size_to_free;
}


void *malloc(size_t sz)
{
    if (!libc_malloc){
        m_init();
    }
    void *addr;
    size_t callchain_size_local = 0;
    void *callchain_strings_local[CALLCHAIN_SIZE];

    if (!_in_trace) {
        get_trace(&callchain_size_local, callchain_strings_local);
    }

    if (sz > 4096 && !_in_trace) {
        if (callchain_size_local >= 4) {
            char **strings = backtrace_symbols(callchain_strings_local, (int)callchain_size_local);
            if (strings) {
                int numa_node = get_numa_node(strings[3], sz);
                libc_free(strings);
                if(NUMA_CXL_NODE == numa_node){
                    addr = hmalloc(sz);
                    if (addr){
                        record_seg((unsigned long)addr, sz);
                    }
                    else {
                        // CXL allocation failed, fallback to libc
                        CXL_LOG("hmalloc failed for size %zu, falling back to libc_malloc", sz);
                        addr = libc_malloc(sz);
                    }
                }else{
                    addr = libc_malloc(sz);
                }
            } else {
                addr = libc_malloc(sz);
            }
        } else {
            addr = libc_malloc(sz);
        }
    } else {
        addr = libc_malloc(sz);
    }
    
    if (!addr) {
        CXL_LOG("malloc failed for size %zu", sz);
    }
    return addr;
}

void *calloc(size_t nmemb, size_t size)
{
    void *addr;
    if (!libc_calloc) {
        memset(empty_data, 0, sizeof(*empty_data));
        addr = empty_data;
    } else {
        addr = libc_calloc(nmemb, size);
    }
    if (!_in_trace && libc_calloc) {
        size_t callchain_size_local = 0;
        void *callchain_strings_local[CALLCHAIN_SIZE];
        get_trace(&callchain_size_local, callchain_strings_local);
    }
    return addr;
}

void *realloc(void *ptr, size_t size)
{
    struct addr_seg *seg = ptr ? find_seg((unsigned long)ptr) : NULL;
    void *addr;

    if (seg) {
        if (size == 0) {
            clear_seg(seg);
            hfree(ptr);
            addr = NULL;
        } else {
            addr = hmalloc(size);
            if (!addr)
                return NULL;
            size_t old_size = seg_length(seg);
            memcpy(addr, ptr, old_size < size ? old_size : size);
            clear_seg(seg);
            record_seg((unsigned long)addr, size);
            hfree(ptr);
        }
    } else {
        addr = libc_realloc(ptr, size);
    }

    if (!_in_trace) {
        size_t callchain_size_local = 0;
        void *callchain_strings_local[CALLCHAIN_SIZE];
        get_trace(&callchain_size_local, callchain_strings_local);
    }
    return addr;
}

void *memalign(size_t align, size_t sz)
{
    void *addr = libc_memalign(align, sz);
    if (!_in_trace) {
        size_t callchain_size_local = 0;
        void *callchain_strings_local[CALLCHAIN_SIZE];
        get_trace(&callchain_size_local, callchain_strings_local);
    }
    return addr;
}

int posix_memalign(void **ptr, size_t align, size_t sz)
{
    int ret = libc_posix_memalign(ptr, align, sz);
    if (!_in_trace) {
        size_t callchain_size_local = 0;
        void *callchain_strings_local[CALLCHAIN_SIZE];
        get_trace(&callchain_size_local, callchain_strings_local);
    }
    return ret;
}

void free(void *p)
{
    if (!libc_free)
        m_init();
    if (!libc_free)
        return;

    if (!_in_trace && libc_free) {
        size_t size_to_free = check_seg((unsigned long) p);
        if (size_to_free > 0) {
            hfree(p);
            return;
        } else {
            libc_free(p);
            return;
        }
    } else if (libc_free) {
        libc_free(p);
    }
}

void *mmap(void *start, size_t length, int prot, int flags, int fd, off_t offset)
{
    void *addr = libc_mmap(start, length, prot, flags, fd, offset);
    if (!_in_trace) {
        size_t callchain_size_local = 0;
        void *callchain_strings_local[CALLCHAIN_SIZE];
        get_trace(&callchain_size_local, callchain_strings_local);
    }
    return addr;
}

void *mmap64(void *start, size_t length, int prot, int flags, int fd, off_t offset)
{
    void *addr = libc_mmap64(start, length, prot, flags, fd, offset);

    if (!_in_trace) {
        size_t callchain_size_local = 0;
        void *callchain_strings_local[CALLCHAIN_SIZE];
        get_trace(&callchain_size_local, callchain_strings_local);
    }
    return addr;
}

int munmap(void *start, size_t length)
{
    int ret = libc_munmap(start, length);
    return ret;
}


void *extent_alloc(extent_hooks_t *extent_hooks __unused, void *new_addr, size_t size,
                   size_t alignment __unused, bool *zero __unused, bool *commit __unused,
                   unsigned arena_ind __unused) {
    new_addr = hmmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, 0, 0);
    return new_addr;
}

static bool extent_dalloc(extent_hooks_t *extent_hooks __unused, void *addr, size_t size,
                          bool committed __unused, unsigned arena_ind __unused) {
    return hmunmap(addr, size);
}

static extent_hooks_t extent_hooks = {
    .alloc = extent_alloc,
    .dalloc = extent_dalloc,
};

void update_env(void) {
    use_jemalloc = getenv_jemalloc();
    nodemask = getenv_nodemask();
    mpol_mode = getenv_mpol_mode();
}

__attribute__((constructor)) void hmalloc_init(void) {
    int err __unused;
    size_t unsigned_size = sizeof(unsigned);

    update_env();

    if (use_jemalloc) {
        maxnode = numa_max_node() + 2;
        hooks = &extent_hooks;
        err = mallctl("arenas.create", &arena_index, &unsigned_size, (void *)&hooks,
                      sizeof(extent_hooks_t *));
        assert(!err);
    }
}

void __attribute__((constructor)) m_init(void)
{

    libc_malloc = (void * ( *)(size_t))dlsym(RTLD_NEXT, "malloc");
    libc_realloc = (void * ( *)(void *, size_t))dlsym(RTLD_NEXT, "realloc");
    libc_calloc = (void * ( *)(size_t, size_t))dlsym(RTLD_NEXT, "calloc");
    libc_free = (void ( *)(void *))dlsym(RTLD_NEXT, "free");
    libc_mmap = (void * ( *)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap");
    libc_munmap = (int ( *)(void *, size_t))dlsym(RTLD_NEXT, "munmap");
    libc_mmap64 = (void * ( *)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap64");
    libc_memalign = (void * ( *)(size_t, size_t))dlsym(RTLD_NEXT, "memalign");
    libc_posix_memalign = (int ( *)(void **, size_t, size_t))dlsym(RTLD_NEXT, "posix_memalign");

    FILE *file;
    char line[256];
    obj_count = 0;

    const char *filename = getenv("CXL_MALLOC_OBJ_RANK_RESULT");
    if (!filename) {
        printf("Environment variable CXL_MALLOC_OBJ_RANK_RESULT not set.");
        return ;
    }
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Can not open file: %s", filename);
        return ;
    }
    

    while (fgets(line, sizeof(line), file) != NULL && obj_count < MAX_LINES) {
        char *cursor = line;
        size_t len = strcspn(cursor, "\r\n");
        cursor[len] = '\0';
        while (*cursor && isspace((unsigned char)*cursor)) {
            cursor++;
        }
        if (*cursor == '\0') {
            continue;
        }
        char *comma = strchr(cursor, ',');
        if (!comma)
            continue;
        *comma = '\0';
        char *retain_str = comma + 1;
        while (*retain_str && isspace((unsigned char)*retain_str))
            retain_str++;
        size_t copy_len = strlen(cursor);
        if (copy_len >= MAX_NAME_LENGTH)
            copy_len = MAX_NAME_LENGTH - 1;
        obj_names[obj_count] = (char*)libc_malloc(copy_len + 1);
        strncpy(obj_names[obj_count], cursor, copy_len);
        obj_names[obj_count][copy_len] = '\0';
        obj_retain[obj_count] = (size_t)strtoull(retain_str, NULL, 10);
        obj_count++;
    }
    fclose(file);
}


//Copy from hmsdk hmalloc.c
void *hmmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    void *new_addr = mmap(addr, length, prot, flags, fd, offset);
    if (unlikely(new_addr == MAP_FAILED))
        return MAP_FAILED;

    if (nodemask > 0) {
        long ret = mbind(new_addr, length, mpol_mode, &nodemask, maxnode, 0);
        if (unlikely(ret)) {
            int mbind_errno = errno;
            munmap(new_addr, length);
            errno = mbind_errno;
            return NULL;
        }
    }
    return new_addr;
}

int hmunmap(void *addr, size_t length) {
    return munmap(addr, length);
}

void *hmalloc(size_t size) {
    void *ptr;
    if (!use_jemalloc)
        return libc_malloc(size);
    
    ptr = mallocx(size, MALLOCX_ARENA(arena_index) | MALLOCX_TCACHE_NONE);
    if (errno == ENOMEM)
        return NULL;
    return ptr;
}

void hfree(void *ptr) {
    if (unlikely(ptr == NULL))
        return;
    if (!use_jemalloc) {
        libc_free(ptr);
        return;
    }
    dallocx(ptr, MALLOCX_ARENA(arena_index) | MALLOCX_TCACHE_NONE);
}

void *hcalloc(size_t nmemb, size_t size) {
    void *ptr = hmalloc(nmemb * size);

    if (likely(ptr))
        memset(ptr, 0, nmemb * size);
    return ptr;
}

void *hrealloc(void *ptr, size_t size) {
    if (!use_jemalloc)
        return realloc(ptr, size);

    if (ptr == NULL)
        return hmalloc(size);

    if (size == 0) {
        hfree(ptr);
        return NULL;
    }
    return rallocx(ptr, size, MALLOCX_ARENA(arena_index) | MALLOCX_TCACHE_NONE);
}

void *haligned_alloc(size_t alignment, size_t size) {
    if (!use_jemalloc)
        return aligned_alloc(alignment, size);

    /* NOTE: ptmalloc in glibc ignores all these checks unlike jemalloc */
    if (unlikely(alignment == 0 || !is_pow2(alignment))) {
        errno = EINVAL;
        return NULL;
    }

    return mallocx(size,
                   MALLOCX_ALIGN(alignment) | MALLOCX_ARENA(arena_index) | MALLOCX_TCACHE_NONE);
}

int hposix_memalign(void **memptr, size_t alignment, size_t size) {
    int old_errno;

    if (!use_jemalloc)
        return libc_posix_memalign(memptr, alignment, size);

    old_errno = errno;

    if (unlikely(alignment == 0 || !is_pow2(alignment))) {
        *memptr = NULL;
        return EINVAL;
    }

    *memptr =
        mallocx(size, MALLOCX_ALIGN(alignment) | MALLOCX_ARENA(arena_index) | MALLOCX_TCACHE_NONE);

    if (unlikely(*memptr == NULL)) {
        int ret = errno;
        errno = old_errno;
        return ret;
    }
    return 0;
}

size_t hmalloc_usable_size(void *ptr) {
    if (!use_jemalloc)
        return malloc_usable_size(ptr);

    if (unlikely(ptr == NULL))
        return 0;
    return sallocx(ptr, 0);
}