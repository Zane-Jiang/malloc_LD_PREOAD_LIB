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
#include <errno.h> 
#include <sys/stat.h>
#include <numa.h>

#define ARR_SIZE 9500000 
#define MAX_TID 512                /* Max number of tids to profile */

#define USE_FRAME_POINTER   0      /* Use Frame Pointers to compute the stack trace (faster) */
#define CALLCHAIN_SIZE      10     /* stack trace length - 增大以区分更深的调用路径 */
#define RESOLVE_SYMBS       1      /* Resolve symbols at the end of the execution; quite costly */
#define FNV_PRIME (1099511628211ULL)
#define NB_ALLOC_TO_IGNORE   0     /* Ignore the first X allocations. */
#define IGNORE_FIRST_PROCESS 0     /* Ignore the first process (and all its threads). Useful for processes */

#if IGNORE_FIRST_PROCESS
static int first_pid;
#endif
#if NB_ALLOC_TO_IGNORE > 0
static int __thread nb_allocs = 0;
#endif

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int __thread pid;
static int __thread tid;
static int __thread tid_index;
static int tids[MAX_TID];

struct log {
    uint64_t rdt;
    void *addr;
    size_t size;
    long entry_type; /* 0 free 1 malloc >=100 mmap */

    size_t callchain_size;
    void *callchain_strings[CALLCHAIN_SIZE];
    unsigned long callchain_offsets[CALLCHAIN_SIZE];  // Add offsets for addr2line
    uint64_t callstack_hash;                     // Deterministic hash of the call stack
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
static char empty_data[1024];              // Increased from 32 to 1024
static char empty_realloc_data[8192];      // Increased from 4096 to 8192
static char empty_calloc_data[8192];       // Separate buffer for calloc
static int init_done = 0;
static int in_init = 0;    // Prevent recursive initialization

static void *(*libc_malloc)(size_t);
static void *(*libc_calloc)(size_t, size_t);
static void *(*libc_realloc)(void *, size_t);
static void (*libc_free)(void *);

static void *(*libc_mmap)(void *, size_t, int, int, int, off_t);
static void *(*libc_mmap64)(void *, size_t, int, int, int, off_t);
static int (*libc_munmap)(void *, size_t);
static void *(*libc_memalign)(size_t, size_t);
static int (*libc_posix_memalign)(void **, size_t, size_t);

FILE *open_file(int tid)
{
    char buff[256];
    char dir_buff[256];
    const char* prof_env = getenv("INSTRU_PROF_DIR");
    const char* prof_dir = prof_env ? prof_env : "/home/jz/instru_prof/";
    
      
    snprintf(dir_buff, sizeof(dir_buff), "%s", prof_dir);
    char *p = dir_buff;
    while (*p) {
        if (*p == '/' && p != dir_buff) {
            *p = '\0';
            mkdir(dir_buff, 0755);
            *p = '/';
        }
        p++;
    }
    mkdir(dir_buff, 0755);
    
    snprintf(buff, sizeof(buff), "%sdata.raw.%d", prof_dir, tid);

    FILE *dump = fopen(buff, "a+");
    if (!dump) {
        fprintf(stderr, "[MallocHook] open %s failed: %s (error code: %d)\n", buff, strerror(errno), errno);
        return NULL; 
    }
    return dump;
}

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
    if (!log_arr[tid_index]) {
        fprintf(stderr, "[MallocHook] Failed to allocate log array for tid_index=%d\n", tid_index);
        return NULL;
    }
    if (log_index[tid_index] >= ARR_SIZE)
        return NULL;

    struct log *l = &log_arr[tid_index][log_index[tid_index]];
    log_index[tid_index]++;
    return l;
}

//  (FNV-1a)
static uint64_t fnv1a_hash(const unsigned char *data, size_t len)
{
    uint64_t hash = 14695981039346656037ULL; // FNV offset basis
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

static uint64_t compute_callstack_hash(void **callchain, size_t callchain_size, size_t alloc_size)
{
    unsigned long offsets[CALLCHAIN_SIZE];
    Dl_info info;
    size_t offset_count = 0;
    
    for (size_t i = 0; i < callchain_size && i < CALLCHAIN_SIZE; i++) {
        if (dladdr(callchain[i], &info) && info.dli_fbase) {
            // Skip shared library frames (only hash main program frames)
            if (info.dli_fname && strstr(info.dli_fname, ".so")) {
                continue;
            }
            offsets[offset_count++] = (unsigned long)callchain[i] - (unsigned long)info.dli_fbase;
        }
    }
    
    if (offset_count == 0) {
        return fnv1a_hash((const unsigned char *)&alloc_size, sizeof(alloc_size));
    }
    
    uint64_t hash = fnv1a_hash((const unsigned char *)offsets, offset_count * sizeof(unsigned long));
    hash ^= alloc_size;
    hash *= FNV_PRIME;
    
    return hash;
}

int get_trace(size_t *size, void **strings)
{
    if (_in_trace)
        return 1;
    _in_trace = 1;
    *size = backtrace(strings, CALLCHAIN_SIZE);
    _in_trace = 0;
    return 0;
}


// Helper function to get symbol offset from ELF binary
static unsigned long get_symbol_offset(void *addr)
{
    Dl_info info;
    if (dladdr(addr, &info) && info.dli_fbase) {
        return (unsigned long)addr - (unsigned long)info.dli_fbase;
    }
    return (unsigned long)addr;
}

static int is_unhelpful_symbol(const char *symbol)
{
    if (!symbol) {
        return 1;
    }
    return strstr(symbol, "__libc_start_call_main") ||
           strstr(symbol, "__libc_start_main") ||
           strstr(symbol, "start_thread") ||
           strstr(symbol, "clone") ||
           strstr(symbol, "malloc_hook") ||
           strstr(symbol, "libMallocHook") ||
           strstr(symbol, "malloc(") ||
           strstr(symbol, "calloc(") ||
           strstr(symbol, "realloc(") ||
           strstr(symbol, "operator new") ||
           strstr(symbol, "_ZNSt15__new_allocator") ||
           strstr(symbol, "_Znwm") ||
           strstr(symbol, "_Znam");
}

static int is_system_alloc_symbol(const char *symbol)
{
    if (!symbol) {
        return 0;
    }
    return strstr(symbol, "malloc") ||
           strstr(symbol, "calloc") ||
           strstr(symbol, "realloc") ||
           strstr(symbol, "free") ||
           strstr(symbol, "memalign") ||
           strstr(symbol, "posix_memalign") ||
           strstr(symbol, "mmap") ||
           strstr(symbol, "munmap") ||
           strstr(symbol, "brk") ||
           strstr(symbol, "_int_malloc") ||
           strstr(symbol, "_int_free") ||
           strstr(symbol, "sysmalloc") ||
           strstr(symbol, "tcache") ||
           strstr(symbol, "operator new") ||
           strstr(symbol, "_Znwm") ||
           strstr(symbol, "_Znam");
}

static int is_system_or_kernel_file(const char *file)
{
    if (!file) {
        return 0;
    }
    return strstr(file, "libc.so") ||
           strstr(file, "ld-linux") ||
           strstr(file, "linux-vdso") ||
           strstr(file, "[vdso]") ||
           strstr(file, "[vsyscall]") ||
           strstr(file, "[kernel]");
}

static int is_system_only_allocation(void **callchain, char **symbols, size_t callchain_size)
{
    Dl_info info;
    int seen_frame = 0;

    for (size_t i = 0; i < callchain_size; i++) {
        const char *symbol = symbols ? symbols[i] : NULL;
        const char *file = NULL;

        if (dladdr(callchain[i], &info) && info.dli_fname) {
            file = info.dli_fname;
            seen_frame = 1;
        }

        if (!symbol) {
            continue;
        }

        if (!is_system_alloc_symbol(symbol)) {
            return 0;
        }

        if (file && !is_system_or_kernel_file(file)) {
            return 0;
        }
    }

    return seen_frame;
}

static int choose_report_frame(void **callchain, char **symbols, size_t callchain_size)
{
    Dl_info info;
    int fallback_main_exe = -1;
    int fallback_any = -1;

    for (size_t i = 0; i < callchain_size; i++) {
        const char *symbol = symbols ? symbols[i] : NULL;
        if (dladdr(callchain[i], &info) && info.dli_fbase) {
            int is_shared = info.dli_fname && strstr(info.dli_fname, ".so");
            if (!is_shared && fallback_main_exe < 0) {
                fallback_main_exe = (int)i;
            }
            if (!is_shared && !is_unhelpful_symbol(symbol)) {
                return (int)i;
            }
        }
        if (fallback_any < 0 && symbol && !is_unhelpful_symbol(symbol)) {
            fallback_any = (int)i;
        }
    }

    if (fallback_main_exe >= 0) {
        return fallback_main_exe;
    }
    if (fallback_any >= 0) {
        return fallback_any;
    }
    return callchain_size >= 4 ? 3 : (callchain_size > 0 ? 0 : -1);
}

extern "C" void *malloc(size_t sz)
{
    if (!libc_malloc) {
        if (in_init) {
            // During initialization, return static buffer for small allocations
            if (sz <= sizeof(empty_data)) {
                return empty_data;
            }
            return NULL;
        }
        m_init();
    }
    if (!libc_malloc)
        return NULL;
    
    void *addr;
    struct log *log_arr;
    addr = libc_malloc(sz);
    if (!_in_trace && init_done) {
        log_arr = get_log();
        if (log_arr) {
            rdtscll(log_arr->rdt);
            log_arr->addr = addr;
            log_arr->size = sz;
            log_arr->entry_type = 1;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
            log_arr->callstack_hash = compute_callstack_hash(log_arr->callchain_strings, log_arr->callchain_size, sz);

            // Store offsets for addr2line
            for (size_t i = 0; i < log_arr->callchain_size; i++) {
                log_arr->callchain_offsets[i] = get_symbol_offset(log_arr->callchain_strings[i]);
            }
        }
    }
    return addr;
}

extern "C" void *calloc(size_t nmemb, size_t size)
{
    size_t total = nmemb * size;
    
    if (!libc_calloc) {
        // Try to initialize first
        if (!in_init) {
            m_init();
        }
        if (!libc_calloc) {
            if (total <= sizeof(empty_calloc_data)) {
                memset(empty_calloc_data, 0, total);
                return empty_calloc_data;
            }
            return NULL;
        }
    }
    
    void *addr = libc_calloc(nmemb, size);
    if (!_in_trace && init_done) {
        struct log *log_arr = get_log();
        if (log_arr) {
            rdtscll(log_arr->rdt);
            log_arr->addr = addr;
            log_arr->size = total;
            log_arr->entry_type = 3;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
            log_arr->callstack_hash = compute_callstack_hash(log_arr->callchain_strings, log_arr->callchain_size, total);
        }
    }
    return addr;
}

extern "C" void *realloc(void *ptr, size_t size)
{
    // Handle realloc before initialization
    if (!libc_realloc) {
        if (in_init) {
            // During init: if ptr is our static buffer or NULL, use static buffer
            if (ptr == NULL || ptr == empty_data || ptr == empty_realloc_data) {
                if (size <= sizeof(empty_realloc_data)) {
                    return empty_realloc_data;
                }
            }
            return NULL;
        }
        m_init();
    }
    if (!libc_realloc)
        return NULL;
    
    // Don't pass our static buffers to real realloc
    if (ptr == empty_data || ptr == empty_realloc_data) {
        void *new_ptr = libc_malloc(size);
        if (new_ptr && ptr) {
            size_t copy_size = (ptr == empty_data) ? sizeof(empty_data) : sizeof(empty_realloc_data);
            if (size < copy_size) copy_size = size;
            memcpy(new_ptr, ptr, copy_size);
        }
        return new_ptr;
    }
    
    void *addr = libc_realloc(ptr, size);
    if (!_in_trace && init_done) {
        struct log *log_arr = get_log();
        if (log_arr) {
            rdtscll(log_arr->rdt);
            log_arr->addr = addr;
            log_arr->size = size;
            log_arr->entry_type = 4;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
            log_arr->callstack_hash = compute_callstack_hash(log_arr->callchain_strings, log_arr->callchain_size, size);
        }
    }
    return addr;
}

extern "C" void *memalign(size_t align, size_t sz)
{
    if (!libc_memalign) {
        m_init();
    }
    if (!libc_memalign)
        return NULL;
    
    void *addr = libc_memalign(align, sz);
    if (!_in_trace && init_done) {
        struct log *log_arr = get_log();
        if (log_arr) {
            rdtscll(log_arr->rdt);
            log_arr->addr = addr;
            log_arr->size = sz;
            log_arr->entry_type = 5;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
            log_arr->callstack_hash = compute_callstack_hash(log_arr->callchain_strings, log_arr->callchain_size, sz);
        }
    }
    return addr;
}

extern "C" int posix_memalign(void **ptr, size_t align, size_t sz)
{
    if (!libc_posix_memalign) {
        m_init();
    }
    if (!libc_posix_memalign)
        return ENOMEM;
    
    int ret = libc_posix_memalign(ptr, align, sz);
    if (!_in_trace && init_done) {
        struct log *log_arr = get_log();
        if (log_arr) {
            rdtscll(log_arr->rdt);
            log_arr->addr = *ptr;
            log_arr->size = sz;
            log_arr->entry_type = 6;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
            log_arr->callstack_hash = compute_callstack_hash(log_arr->callchain_strings, log_arr->callchain_size, sz);
        }
    }
    return ret;
}

extern "C" void free(void *p)
{
    // Skip NULL pointer
    if (!p)
        return;
    
    // Skip the static buffers (used during early initialization)
    if (p == empty_data || p == empty_realloc_data || p == empty_calloc_data)
        return;
    
    // Skip if libc_free is not initialized yet
    if (!libc_free)
        return;
    
    struct log *log_arr;
    if (!_in_trace && init_done) {
        log_arr = get_log();
        if (log_arr) {
            rdtscll(log_arr->rdt);
            log_arr->addr = p;
            log_arr->size = 0;
            log_arr->entry_type = 2;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
            log_arr->callstack_hash = compute_callstack_hash(log_arr->callchain_strings, log_arr->callchain_size, 0);
        }
    }
    libc_free(p);
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
            log_arr->callstack_hash = compute_callstack_hash(log_arr->callchain_strings, log_arr->callchain_size, length);
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
            log_arr->entry_type = flags + 200;
            get_trace(&log_arr->callchain_size, log_arr->callchain_strings);
            log_arr->callstack_hash = compute_callstack_hash(log_arr->callchain_strings, log_arr->callchain_size, length * 4 * 1024);
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
    log_arr.entry_type = 90;
    return addr;
}

int __thread bye_done = 0;
static int filter_system_malloc = 1;  

void __attribute__((destructor)) bye(void)
{
    if (bye_done)
        return;
    bye_done = 1;

    
    unsigned int i, j;
    int files_written = 0;
    int total_records = 0;
    
    for (i = 1; i < MAX_TID; i++) {
        if (tids[i] == 0)
            break;
        if (!log_arr[i]) {
            continue;
        }
        
        total_records += log_index[i];
        
        if (log_index[i] == 0) {
            continue;
        }
        
        FILE *file = open_file(tids[i]);
        if (!file) {
            fprintf(stderr, "[MallocHook] Failed to open file for TID %d\n", tids[i]);
            continue;
        }
        
        
        int skipped_system_only = 0;
        for (j = 0; j < log_index[i]; j++) {
            struct log *l = &log_arr[i][j];
            _in_trace = 1;
            char **strings = backtrace_symbols(l->callchain_strings, l->callchain_size);
            if (!strings) {
                fprintf(file, "0x%016lx ", (unsigned long)l->callstack_hash);
                fprintf(file, "backtrace_failed ");
                fprintf(file, "%lu %lu %lx %d\n", l->rdt, (long unsigned)l->size,
                (long unsigned)l->addr, (int)l->entry_type);
                _in_trace = 0;
                continue;
            }
            
            if (filter_system_malloc) {
                if (is_system_only_allocation(l->callchain_strings, strings, l->callchain_size)) {
                    libc_free(strings);
                    _in_trace = 0;
                    skipped_system_only++;
                    continue;
                }
            }
            
            int frame_idx = choose_report_frame(l->callchain_strings, strings, l->callchain_size);
            if (frame_idx >= 0 && (size_t)frame_idx < l->callchain_size) {
                fprintf(file, "0x%016lx ", (unsigned long)l->callstack_hash);
                fprintf(file, "%s ", strings[frame_idx]);
                fprintf(file, "%lu %lu %lx %d ", l->rdt, (long unsigned)l->size,
                (long unsigned)l->addr, (int)l->entry_type);

                Dl_info info;
                // Try to get binary name and offset for addr2line
                if (dladdr(l->callchain_strings[frame_idx], &info) && info.dli_fname) {
                    fprintf(file, "%s:0x%lx \n", info.dli_fname,
                            (unsigned long)l->callchain_strings[frame_idx] - (unsigned long)info.dli_fbase);
                }
            }else {
                fprintf(file, "0x%016lx ", (unsigned long)l->callstack_hash);
                fprintf(file, "unknown ");
                fprintf(file, "%lu %lu %lx %d\n", l->rdt, (long unsigned)l->size,
                (long unsigned)l->addr, (int)l->entry_type);
            }
            libc_free(strings);

        }
        if (filter_system_malloc && skipped_system_only > 0) {
            fprintf(stderr, "[MallocHook] TID %d: skipped %d system-only allocations\n", 
                    tids[i], skipped_system_only);
        }
        fclose(file);
    }
}

void __attribute__((constructor)) m_init(void)
{
    if (init_done || in_init)
        return;
    
    in_init = 1;
    
    libc_malloc = (void * ( *)(size_t))dlsym(RTLD_NEXT, "malloc");
    libc_realloc = (void * ( *)(void *, size_t))dlsym(RTLD_NEXT, "realloc");
    libc_calloc = (void * ( *)(size_t, size_t))dlsym(RTLD_NEXT, "calloc");
    libc_free = (void ( *)(void *))dlsym(RTLD_NEXT, "free");
    libc_mmap = (void * ( *)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap");
    libc_munmap = (int ( *)(void *, size_t))dlsym(RTLD_NEXT, "munmap");
    libc_mmap64 = (void * ( *)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap64");
    libc_memalign = (void * ( *)(size_t, size_t))dlsym(RTLD_NEXT, "memalign");
    libc_posix_memalign = (int ( *)(void **, size_t, size_t))dlsym(RTLD_NEXT, "posix_memalign");
    
    in_init = 0;
    init_done = 1;
}
