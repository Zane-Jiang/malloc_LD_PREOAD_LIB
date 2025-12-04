#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define __USE_GNU
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
#include <new>
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
#define RESOLVE_SYMBS       1      /* Resolve symbols at the end of the execution; quite costly */

#define NB_ALLOC_TO_IGNORE   0     /* Ignore the first X allocations. */
#define IGNORE_FIRST_PROCESS 0     /* Ignore the first process (and all its threads). Useful for processes */

#define MAX_OBJECTS          30000



#if IGNORE_FIRST_PROCESS
static int first_pid;
#endif
#if NB_ALLOC_TO_IGNORE > 0
static int __thread nb_allocs = 0;
#endif

#define MAX_LINES 1000
#define MAX_NAME_LENGTH 20
char *obj_names[MAX_LINES];
size_t obj_retain[MAX_LINES];
const uint64_t SYS_RETAIN_CAPACITY = 100 * 1024 *1024; // systerm retain capacity
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int __thread pid;
static int __thread tid;
static int __thread tid_index;
static int tids[MAX_TID];

struct addr_seg {
    long unsigned start;
    long unsigned end;
};

static struct addr_seg addr_segs[MAX_OBJECTS];

struct log {
    uint64_t rdt;
    void *addr;
    size_t size;
    long entry_type; /* 0 free 1 malloc >=100 mmap */
    size_t callchain_size;
    void *callchain_strings[CALLCHAIN_SIZE];
};
struct log *log_arr[MAX_TID];
static size_t log_index[MAX_TID];

void __attribute__((constructor)) m_init(void);

#ifdef __x86_64__
#define rdtscll(val) { \
    unsigned int __a,__d;                                        \
    asm volatile("rdtsc" : "=a" (__a), "=d" (__d));              \
    (val) = ((unsigned long)__a) | (((unsigned long)__d)<<32);   \
}

#else
#define rdtscll(val) __asm__ __volatile__("rdtsc" : "=A" (val))
#endif

static int __thread _in_trace = 0;
#define get_bp(bp) asm("movq %%rbp, %0" : "=r" (bp) :)

static __attribute__((unused)) int in_first_dlsym = 0;
static char empty_data[32];

static void *(*libc_malloc)(size_t);
static void *(*libc_calloc)(size_t, size_t);
static void *(*libc_realloc)(void *, size_t);
static void (*libc_free)(void *);

static void *(*libc_mmap)(void *, size_t, int, int, int, off_t);
static void *(*libc_mmap64)(void *, size_t, int, int, int, off_t);
static int (*libc_munmap)(void *, size_t);
static void *(*libc_memalign)(size_t, size_t);
static int (*libc_posix_memalign)(void **, size_t, size_t);

static int obj_count = 0;
struct log *get_log()
{
#if IGNORE_FIRST_PROCESS
    if (!first_pid)
        first_pid = _pid;
    if (_pid == first_pid)
        return NULL;
#endif

#if NB_ALLOC_TO_IGNORE > 0
    nb_allocs++;
    if (nb_allocs < NB_ALLOC_TO_IGNORE)
        return NULL;
#endif

    int i;
    if (!tid) {
        tid = (int) syscall(186); /* 64b only */
        pthread_mutex_lock(&lock);
        for (i = 1; i < MAX_TID; i++) {
            if (tids[i] == 0) {
                tids[i] = tid;
                tid_index = i;
                break;
            }
        }
        if (tid_index == 0) {
            fprintf(stderr, "Too many threads!\n");
            exit(-1);
        }
        pthread_mutex_unlock(&lock);
    }
    if (!log_arr[tid_index])
        log_arr[tid_index] = (struct log*) libc_malloc(sizeof(*log_arr[tid_index]) * ARR_SIZE);
    if (log_index[tid_index] >= ARR_SIZE)
        return NULL;

    struct log *l = &log_arr[tid_index][log_index[tid_index]];
    log_index[tid_index]++;
    return l;
}

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

int get_numa_node(void *string, size_t sz)
{
    return NUMA_CXL_NODE;
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
            if (numa_available() < 0)
                return NUMA_DEFAULT_NODE;
            // static int local_node = -2;
            // if (local_node == -2) {
            //     int cpu = sched_getcpu();
            //     local_node = cpu >= 0 ? numa_node_of_cpu(cpu) : -1;
            //     if (local_node < 0)
            //         local_node = numa_preferred();
            //     if (local_node < 0)
            //         local_node = 0;
            // }
            static int local_node = NUMA_LOCAL_NODE;
            long long local_free = get_node_free_bytes(local_node);
            if (local_free < 0)
                return NUMA_DEFAULT_NODE;
            long long available = local_free - (long long)obj_retain[i] - SYS_RETAIN_CAPACITY;
            if (available > (long long)sz)
                return NUMA_DEFAULT_NODE;

            long long cxl_free = get_node_free_bytes(NUMA_CXL_NODE);
            if (cxl_free > (long long)sz)
                return NUMA_CXL_NODE;
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

size_t check_seg(unsigned long addr)
{
    int i;
    size_t size_to_free;
    size_to_free = 0;
    for (i = 0; i < MAX_OBJECTS; i += 1) {
        struct addr_seg *seg = &addr_segs[i];
        if (seg->start <= addr && seg->end > addr) {
            size_to_free = (size_t) (seg->end - addr);
            if (seg->start == addr) {
                seg->start = 0;
                seg->end = 0;
            } else {
                seg->end = addr;
            }
            break;
        }
    }
    return size_to_free;
}


static inline int prefer_node(int nid) {
    unsigned long nodemask = 1UL << nid;
    int maxnode = sizeof(nodemask) * 8;
    return set_mempolicy(MPOL_PREFERRED, &nodemask, maxnode);
}

static inline int reset_mempolicy() {
    return set_mempolicy(MPOL_DEFAULT, NULL, 0);
}
void *alloc_on_node(size_t sz,int node){
    void *p = NULL;
    if(node == NUMA_DEFAULT_NODE){
        p = libc_malloc(sz);  
        return p;
    }
    if (prefer_node(node) != 0) {
        perror("prefer_node(0) failed");
        return libc_malloc(sz);
    }
    p = libc_malloc(sz);  
    reset_mempolicy();
    return p;
}


extern "C" void *malloc(size_t sz)
{
    if (!libc_malloc)
        m_init();
    void *addr;
    struct log *log_arr;
    if (!_in_trace) {
        log_arr = get_log();
        if (log_arr) {
            rdtscll(log_arr->rdt);
            log_arr->size = sz;
            log_arr->entry_type = 1;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
        }
    }
    if (sz > 4096 && !_in_trace && log_arr) {
        if (log_arr->callchain_size >= 4) {
            char **strings = backtrace_symbols(log_arr->callchain_strings, log_arr->callchain_size);
            if (strings) {
                int numa_node = get_numa_node(strings[3], sz);
                libc_free(strings);
                addr = alloc_on_node(sz, numa_node);
            } else {
                addr = libc_malloc(sz);
            }
            if (log_arr)
                log_arr->addr = addr;
        } else {
            addr = libc_malloc(sz);
        }
    } else {
        addr = libc_malloc(sz);
    }
    return addr;
}

extern "C" void *calloc(size_t nmemb, size_t size)
{
    void *addr;
    if (!libc_calloc) {
        memset(empty_data, 0, sizeof(*empty_data));
        addr = empty_data;
    } else {
        addr = libc_calloc(nmemb, size);
    }
    if (!_in_trace && libc_calloc) {
        struct log *log_arr = get_log();
        if (log_arr) {
            rdtscll(log_arr->rdt);
            log_arr->addr = addr;
            log_arr->size = nmemb * size;
            log_arr->entry_type = 1;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
        }
    }
    return addr;
}

extern "C" void *realloc(void *ptr, size_t size)
{
    void *addr = libc_realloc(ptr, size);
    if (!_in_trace) {
        struct log *log_arr = get_log();
        if (log_arr) {
            rdtscll(log_arr->rdt);
            log_arr->addr = addr;
            log_arr->size = size;
            log_arr->entry_type = 1;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
        }
    }
    return addr;
}

extern "C" void *memalign(size_t align, size_t sz)
{
    void *addr = libc_memalign(align, sz);
    if (!_in_trace) {
        struct log *log_arr = get_log();
        if (log_arr) {
            rdtscll(log_arr->rdt);
            log_arr->addr = addr;
            log_arr->size = sz;
            log_arr->entry_type = 1;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
        }
    }
    return addr;
}

extern "C" int posix_memalign(void **ptr, size_t align, size_t sz)
{
    int ret = libc_posix_memalign(ptr, align, sz);
    if (!_in_trace) {
        struct log *log_arr = get_log();
        if (log_arr) {
            rdtscll(log_arr->rdt);
            log_arr->addr = *ptr;
            log_arr->size = sz;
            log_arr->entry_type = 1;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
        }
    }
    return ret;
}

extern "C" void free(void *p)
{
    if (!libc_free)
        m_init();
    if (!libc_free)
        return;

    if (!_in_trace && libc_free) {
        size_t size_to_free = check_seg((unsigned long) p);
        if (size_to_free > 0) {
            numa_free(p, size_to_free);
            return;
        } else {
            libc_free(p);
            return;
        }
    } else if (libc_free) {
        libc_free(p);
    }
    /* libc_free(p); */
}

void *operator new(size_t sz) throw(std::bad_alloc)
{
    return malloc(sz);
}

void *operator new(size_t sz, const std::nothrow_t &) throw()
{
    return malloc(sz);
}

void *operator new[](size_t sz) throw(std::bad_alloc)
{
    return malloc(sz);
}

void *operator new[](size_t sz, const std::nothrow_t &) throw()
{
    return malloc(sz);
}

void operator delete(void *ptr)
{
    free(ptr);
}

void operator delete[](void *ptr)
{
    free(ptr);
}

extern "C" void *mmap(void *start, size_t length, int prot, int flags, int fd, off_t offset)
{
    void *addr = libc_mmap(start, length, prot, flags, fd, offset);

    if (!_in_trace) {
        struct log *log_arr = get_log();
        if (log_arr) {
            rdtscll(log_arr->rdt);
            log_arr->addr = addr;
            log_arr->size = length;
            log_arr->entry_type = flags + 100;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
        }
    }

    return addr;
}

extern "C" void *mmap64(void *start, size_t length, int prot, int flags, int fd, off_t offset)
{
    void *addr = libc_mmap64(start, length, prot, flags, fd, offset);

    if (!_in_trace) {
        struct log *log_arr = get_log();
        if (log_arr) {
            rdtscll(log_arr->rdt);
            log_arr->addr = addr;
            log_arr->size = length * 4 * 1024;
            log_arr->entry_type = flags + 100;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
        }
    }

    return addr;
}

extern "C" int munmap(void *start, size_t length)
{
    int addr = libc_munmap(start, length);
    struct log log_arr;
    rdtscll(log_arr.rdt);
    log_arr.addr = start;
    log_arr.size = length;
    log_arr.entry_type = 2;
    return addr;
}

int __thread bye_done = 0;
void __attribute__((destructor)) bye(void)
{
    if (bye_done)
        return;
    bye_done = 1;
}

int read_score(){
    FILE *file;
    char line[256];
    obj_count = 0;

    const char *filename = getenv("CXL_MALLOC_OBJ_RANK_RESULT");
    if (!filename) {
        printf("Environment variable CXL_MALLOC_OBJ_RANK_RESULT not set.");
        return 1;
    }
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Can not open file: %s", filename);
        return 1;
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
    return 0;
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

    read_score();
}
