#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define __USE_GNU

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
#define NUMA_CXL_NODE 1
#define PAGE_SIZE 4096           /* Page size for interleaving */
#define CXL_MALLOC_ENABLE_LOG 
#ifdef CXL_MALLOC_ENABLE_LOG
#define CXL_LOG(fmt, ...) printf("[cxl_malloc] " fmt "\n", ##__VA_ARGS__)
#else
#define CXL_LOG(...) ((void)0)
#endif

#define MAX_OBJECTS          30000
#define INTERLEAVE_THRESHOLD 4096  /* Minimum size for interleave allocation */
#define MAX_HASH_ENTRIES 1000
static uint64_t obj_hashes[MAX_HASH_ENTRIES];
static int obj_place_flags[MAX_HASH_ENTRIES];          // 0: libc, 1: interweave, 2: h-alloc (CXL)
static long long obj_heate_cnts[MAX_HASH_ENTRIES];     // optional
static double obj_bw_scores[MAX_HASH_ENTRIES];         // optional
static int obj_count = 0;

struct addr_seg {
    long unsigned start;
    long unsigned end;
    int place_flag;   // 0: libc, 1: interweave, 2: h-alloc
};
static int __thread _in_trace = 0;
static struct addr_seg addr_segs[MAX_OBJECTS];
static pthread_mutex_t seg_lock = PTHREAD_MUTEX_INITIALIZER;
#define CALLCHAIN_SIZE      10     /* stack trace length */


#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define __unused __attribute__((unused))
#define is_pow2(val) (((val) & ((val)-1)) == 0)


/* Function declarations */
void *interweave_malloc(size_t size);
void interweave_free(void *ptr);
void *interweave_calloc(size_t nmemb, size_t size);
void *interweave_realloc(void *ptr, size_t size);
void *interweave_aligned_alloc(size_t alignment, size_t size);
int interweave_posix_memalign(void **memptr, size_t alignment, size_t sz);
void ensure_mapping_funcs(void);

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

static size_t (*libc_malloc_usable_size)(void *) = NULL;
static void *(*libc_aligned_alloc)(size_t, size_t) = NULL;


// Deterministic hash function (FNV-1a)
static uint64_t fnv1a_hash(const unsigned char *data, size_t len)
{
    uint64_t hash = 14695981039346656037ULL; // FNV offset basis
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 1099511628211ULL; // FNV prime
    }
    return hash;
}

// Compute deterministic callstack hash (based on offsets, not runtime addresses)
static uint64_t compute_callstack_hash(void **callchain, size_t size)
{
    unsigned long offsets[CALLCHAIN_SIZE];
    Dl_info info;
    size_t offset_count = 0;
    
    for (size_t i = 0; i < size && i < CALLCHAIN_SIZE; i++) {
        if (dladdr(callchain[i], &info) && info.dli_fbase) {
            // Skip shared library frames (only hash main program frames)
            if (info.dli_fname && strstr(info.dli_fname, ".so")) {
                continue;
            }
            // Use offset relative to library base (deterministic)
            offsets[offset_count++] = (unsigned long)callchain[i] - (unsigned long)info.dli_fbase;
        }
    }
    
    if (offset_count == 0) {
        return 0;
    }
    
    return fnv1a_hash((const unsigned char *)offsets, offset_count * sizeof(unsigned long));
}

// Based on hash value, determine placement (0/1/2)
static inline int get_place_flag_by_hash(uint64_t hash)
{
    if (hash == 0) return 0;
    for (int i = 0; i < obj_count; i++) {
        if (obj_hashes[i] == hash) {
            return obj_place_flags[i];
        }
    }
    return 0;
}

/* Record allocation segment */
void record_seg(unsigned long addr, size_t size, int place_flag)
{
    pthread_mutex_lock(&seg_lock);
    for (int i = 0; i < MAX_OBJECTS; i++) {
        struct addr_seg *seg = &addr_segs[i];
        if (seg->start == 0 && seg->end == 0) {
            seg->start = addr;
            seg->end = addr + size;
            seg->place_flag = place_flag;
            pthread_mutex_unlock(&seg_lock);
            return;
        }
    }
    pthread_mutex_unlock(&seg_lock);
    CXL_LOG("Warning: addr_segs full, cannot record segment");
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
        seg->place_flag = 0;
    }
}


int get_trace(size_t *size, void **strings)
{
    if (_in_trace)
        return 1;
    _in_trace = 1;
    int depth = backtrace(strings, CALLCHAIN_SIZE);
    if (depth < 0)
        depth = 0;
    *size = (size_t)depth;
    _in_trace = 0;
    return 0;
}

size_t check_seg(unsigned long addr, int *place_flag)
{
    pthread_mutex_lock(&seg_lock);
    struct addr_seg *seg = find_seg(addr);
    size_t size_to_free = 0;
    if (seg) {
        size_to_free = seg_length(seg);
        if (place_flag) {
            *place_flag = seg->place_flag;
        }
        clear_seg(seg);
    } else if (place_flag) {
        *place_flag = 0;
    }
    pthread_mutex_unlock(&seg_lock);
    return size_to_free;
}

size_t malloc_usable_size(void *ptr)
{
    if (unlikely(!libc_malloc_usable_size)) {
        libc_malloc_usable_size = (size_t (*)(void *))dlsym(RTLD_NEXT, "malloc_usable_size");
        if (!libc_malloc_usable_size) {
            libc_malloc_usable_size = (size_t (*)(void *))dlsym(RTLD_NEXT, "__malloc_usable_size");
        }
        if (!libc_malloc_usable_size) {
            return 0;
        }
    }

    pthread_mutex_lock(&seg_lock);
    struct addr_seg *seg = find_seg((unsigned long)ptr);
    size_t sz = seg ? seg_length(seg) : 0;
    int pf = seg ? seg->place_flag : 0;
    pthread_mutex_unlock(&seg_lock);

    if (seg) {
        if (pf == 2) {
            return hmalloc_usable_size(ptr);
        }
        if (sz > 0) {
            return sz;
        }
    }

    return libc_malloc_usable_size(ptr);
}


size_t __malloc_usable_size(void *ptr)
{
    return malloc_usable_size(ptr);
}

void *malloc(size_t sz)
{
    if (unlikely(!libc_malloc)) {
        libc_malloc = (void * (*)(size_t))dlsym(RTLD_NEXT, "malloc");
        CXL_LOG("malloc dlsym libc_malloc=%p", libc_malloc);
        if (!libc_malloc) return NULL;
    }

    if (sz > INTERLEAVE_THRESHOLD) {
        void *addr;
        size_t callchain_size_local = 0;
        void *callchain_strings_local[CALLCHAIN_SIZE];

        if (!_in_trace) {
            get_trace(&callchain_size_local, callchain_strings_local);
        }
        if (callchain_size_local >= 4) {
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings_local, callchain_size_local);
            int pf = get_place_flag_by_hash(callstack_hash);
            if (pf == 2) {
                addr = hmalloc(sz);
                if (addr) {
                    record_seg((unsigned long)addr, 0, 2);
                    return addr;
                }
            } else if (pf == 1) {
                addr = interweave_malloc(sz);
                if (addr) {
                    return addr;
                }
            }
        }
    }
    return libc_malloc(sz);
}

void *calloc(size_t nmemb, size_t size)
{
    if (unlikely(!libc_calloc)) {
        libc_calloc = (void * (*)(size_t, size_t))dlsym(RTLD_NEXT, "calloc");
        if (!libc_calloc) return NULL;
    }

    if (size != 0 && nmemb > SIZE_MAX / size) {
        errno = ENOMEM;
        return NULL;
    }

    size_t total = nmemb * size;

    if (total > INTERLEAVE_THRESHOLD) {
        size_t callchain_size_local = 0;
        void *callchain_strings_local[CALLCHAIN_SIZE];

        if (!_in_trace) {
            get_trace(&callchain_size_local, callchain_strings_local);
        }
        if (callchain_size_local >= 4) {
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings_local, callchain_size_local);
            int pf = get_place_flag_by_hash(callstack_hash);
            if (pf == 2) {
                void *addr = hcalloc(nmemb, size);
                if (addr) {
                    record_seg((unsigned long)addr, 0, 2);
                    return addr;
                }
            } else if (pf == 1) {
                void *addr = interweave_calloc(nmemb, size);
                if (addr) {
                    return addr;
                }
                CXL_LOG("interweave_calloc failed, falling back to libc_calloc");
            }
        }
    }

    return libc_calloc(nmemb, size);
}

void *realloc(void *ptr, size_t size)
{
    if (unlikely(!libc_realloc)) {
        libc_realloc = (void * (*)(void *, size_t))dlsym(RTLD_NEXT, "realloc");
        if (!libc_realloc) return NULL;
    }
    if (!ptr) {
        return malloc(size);
    }
    
    if (size == 0) {
        free(ptr);
        return NULL;
    }
    
    pthread_mutex_lock(&seg_lock);
    struct addr_seg *seg = find_seg((unsigned long)ptr);
    int pf = seg ? seg->place_flag : 0;
    pthread_mutex_unlock(&seg_lock);
    
    if (pf == 1) {
        return interweave_realloc(ptr, size);
    } else if (pf == 2) {
        return hrealloc(ptr, size);
    }
    
    return libc_realloc(ptr, size);
}

void *memalign(size_t align, size_t sz)
{
    if (unlikely(!libc_memalign)) {
        libc_memalign = (void * (*)(size_t, size_t))dlsym(RTLD_NEXT, "memalign");
        if (!libc_memalign) return NULL;
    }
    if (sz > INTERLEAVE_THRESHOLD) {
        size_t callchain_size_local = 0;
        void *callchain_strings_local[CALLCHAIN_SIZE];

        if (!_in_trace) {
            get_trace(&callchain_size_local, callchain_strings_local);
        }
        if (callchain_size_local >= 4) {
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings_local, callchain_size_local);
            int pf = get_place_flag_by_hash(callstack_hash);
            if (pf == 2) {
                void *addr = haligned_alloc(align, sz);
                if (addr) {
                    record_seg((unsigned long)addr, 0, 2);
                    return addr;
                }
            } else if (pf == 1) {
                void *addr = interweave_aligned_alloc(align, sz);
                if (addr) {
                    return addr;
                }
            }
        }
    }
    
    return libc_memalign(align, sz);
}

int posix_memalign(void **ptr, size_t align, size_t sz)
{
    if (unlikely(!libc_posix_memalign)) {
        libc_posix_memalign = (int (*)(void **, size_t, size_t))dlsym(RTLD_NEXT, "posix_memalign");
        if (!libc_posix_memalign) return ENOMEM;
    }
    if (sz > INTERLEAVE_THRESHOLD) {
        size_t callchain_size_local = 0;
        void *callchain_strings_local[CALLCHAIN_SIZE];

        if (!_in_trace) {
            get_trace(&callchain_size_local, callchain_strings_local);
        }
        if (callchain_size_local >= 4) {
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings_local, callchain_size_local);
            int pf = get_place_flag_by_hash(callstack_hash);
            if (pf == 2) {
                int ret = hposix_memalign(ptr, align, sz);
                if (ret == 0) {
                    record_seg((unsigned long)(*ptr), 0, 2);
                    return 0;
                }
            } else if (pf == 1) {
                int ret = interweave_posix_memalign(ptr, align, sz);
                if (ret == 0) {
                    return 0;
                }
            }
        }
    }
    
    return libc_posix_memalign(ptr, align, sz);
}

void free(void *p)
{
    if (unlikely(!libc_free)) {
        libc_free = (void (*)(void *))dlsym(RTLD_NEXT, "free");
        if (!libc_free || !p) return;
    }

    int place_flag = 0;
    size_t size_to_free = check_seg((unsigned long)p, &place_flag);
    
    if (place_flag == 1) {
        ensure_mapping_funcs();
        if (likely(libc_munmap && size_to_free > 0)) {
            libc_munmap(p, size_to_free);
        }
    } else if (place_flag == 2) {
        hfree(p);
    } else {
        libc_free(p);
    }
}

void *mmap(void *start, size_t length, int prot, int flags, int fd, off_t offset)
{
    if (unlikely(!libc_mmap)) {
        libc_mmap = (void * (*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap");
        if (!libc_mmap) return MAP_FAILED;
    }
    return libc_mmap(start, length, prot, flags, fd, offset);
}

void *mmap64(void *start, size_t length, int prot, int flags, int fd, off_t offset)
{
    if (unlikely(!libc_mmap64)) {
        libc_mmap64 = (void * (*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap64");
        if (!libc_mmap64) return MAP_FAILED;
    }
    return libc_mmap64(start, length, prot, flags, fd, offset);
}

int munmap(void *start, size_t length)
{
    if (unlikely(!libc_munmap)) {
        libc_munmap = (int (*)(void *, size_t))dlsym(RTLD_NEXT, "munmap");
        if (!libc_munmap) return -1;
    }
    return libc_munmap(start, length);
}


static unsigned arena_index;
static extent_hooks_t *hooks;
static int maxnode;

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
__attribute__((constructor)) void hmalloc_init(void) {
    int err __unused;
    size_t unsigned_size = sizeof(unsigned);
    maxnode = numa_max_node() + 2;
    hooks = &extent_hooks;
    err = mallctl("arenas.create", &arena_index, &unsigned_size, (void *)&hooks,
                      sizeof(extent_hooks_t *));
    assert(!err);
}

__attribute__((constructor)) void m_init(void) {
    libc_malloc = (void * (*)(size_t))dlsym(RTLD_NEXT, "malloc");
    libc_realloc = (void * (*)(void *, size_t))dlsym(RTLD_NEXT, "realloc");
    libc_calloc = (void * (*)(size_t, size_t))dlsym(RTLD_NEXT, "calloc");
    libc_free = (void (*)(void *))dlsym(RTLD_NEXT, "free");
    libc_mmap = (void * (*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap");
    libc_munmap = (int (*)(void *, size_t))dlsym(RTLD_NEXT, "munmap");
    libc_mmap64 = (void * (*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap64");
    libc_memalign = (void * (*)(size_t, size_t))dlsym(RTLD_NEXT, "memalign");
    libc_posix_memalign = (int (*)(void **, size_t, size_t))dlsym(RTLD_NEXT, "posix_memalign");
    libc_aligned_alloc = (void * (*)(size_t, size_t))dlsym(RTLD_NEXT, "aligned_alloc");

    FILE *file;
    char line[256];
    obj_count = 0;

    const char *filename = getenv("CXL_MALLOC_OBJ_RANK_RESULT");
    if (!filename) {
        printf("Environment variable CXL_MALLOC_OBJ_RANK_RESULT not set.\n");
        return;
    }
    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Can not open file: %s\n", filename);
        return;
    }

    // Config file format:
    // hash_value,place_flag[,heate_count][,BW_Bound_score]
    // Examples:
    // 0x1234567890abcdef,1
    // 0x1234567890abcdef,0,0,0.0
    // 0x1234567890abcdef,0,12,0.87
    while (fgets(line, sizeof(line), file) != NULL && obj_count < MAX_HASH_ENTRIES) {
        char *cursor = line;
        size_t len = strcspn(cursor, "\r\n");
        cursor[len] = '\0';
        while (*cursor && isspace((unsigned char)*cursor)) cursor++;
        if (*cursor == '\0' || *cursor == '#') continue;

        // tokenize by comma
        char *t1 = cursor;
        char *t2 = strchr(t1, ','); if (!t2) continue; *t2++ = '\0';
        while (*t2 && isspace((unsigned char)*t2)) t2++;
        char *t3 = strchr(t2, ','); if (t3) { *t3++ = '\0'; while (*t3 && isspace((unsigned char)*t3)) t3++; }
        char *t4 = NULL;
        if (t3) {
            t4 = strchr(t3, ',');
            if (t4) { *t4++ = '\0'; while (*t4 && isspace((unsigned char)*t4)) t4++; }
        }

        uint64_t hash_value = strtoull(t1, NULL, 0);
        int place_flag = (int)strtol(t2, NULL, 10);
        long long heate_cnt = (t3 && *t3) ? strtoll(t3, NULL, 10) : LLONG_MIN; // LLONG_MIN -> not provided
        double bw_score = (t4 && *t4) ? strtod(t4, NULL) : 0.0;

        obj_hashes[obj_count] = hash_value;
        obj_place_flags[obj_count] = place_flag;
        obj_heate_cnts[obj_count] = heate_cnt;
        obj_bw_scores[obj_count] = bw_score;
        obj_count++;
    }
    fclose(file);

    // Apply placement rules:
    // 1) heate_count == 0 -> flag = 2
    // 2) Among the rest, top 10% by BW_Bound_score -> flag = 1
    // 3) Others -> flag = 0
    if (obj_count > 0) {
        // collect eligible scores (heate_count != 0)
        int eligible = 0;
        for (int i = 0; i < obj_count; ++i) {
            if (obj_heate_cnts[i] == 0) continue;
            eligible++;
        }
        int topk = eligible / 10;
        if (topk < 1 && eligible > 0) topk = 1;

        double threshold = 0.0;
        if (eligible > 0 && topk > 0) {
            // copy scores
            double *scores = (double *)libc_malloc(sizeof(double) * eligible);
            if (scores) {
                int idx = 0;
                for (int i = 0; i < obj_count; ++i) {
                    if (obj_heate_cnts[i] == 0) continue;
                    scores[idx++] = obj_bw_scores[i];
                }
                // sort descending
                int cmp_desc(const void *a, const void *b) {
                    double da = *(const double *)a;
                    double db = *(const double *)b;
                    if (da < db) return 1;
                    if (da > db) return -1;
                    return 0;
                }
                qsort(scores, eligible, sizeof(double), cmp_desc);
                threshold = scores[topk - 1];
                libc_free(scores);
            }
        }

        for (int i = 0; i < obj_count; ++i) {
            if (obj_heate_cnts[i] == 0) {
                obj_place_flags[i] = 2;
            } else if (eligible > 0 && obj_bw_scores[i] >= threshold) {
                obj_place_flags[i] = 1;
            } else {
                obj_place_flags[i] = 0;
            }
        }
    }

    CXL_LOG("Loaded %d hash entries (place_flag applied)", obj_count);
}
 
/*
 * Interweave malloc: allocates memory and binds pages according to interleave ratio.
 */
void ensure_mapping_funcs(void)
{
    if (unlikely(!libc_mmap)) {
        libc_mmap = (void * (*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap");
    }
    if (unlikely(!libc_munmap)) {
        libc_munmap = (int (*)(void *, size_t))dlsym(RTLD_NEXT, "munmap");
    }
}



static int apply_interleave_policy(void *addr, size_t len)
{
    unsigned long nodemask = (1UL << NUMA_LOCAL_NODE) | (1UL << NUMA_CXL_NODE);
    unsigned long maxnode = sizeof(nodemask) * 8;

    if (mbind(addr, len, MPOL_WEIGHTED_INTERLEAVE, &nodemask, maxnode, MPOL_MF_MOVE) != 0) {
        CXL_LOG("mbind interleave failed: %s", strerror(errno));
        return -1;
    }
    return 0;
}

void *interweave_malloc(size_t size)
{
    ensure_mapping_funcs();
    if (unlikely(!libc_mmap || !libc_munmap)) {
        errno = ENOSYS;
        return NULL;
    }
    /* Allocate a contiguous virtual memory region */
    size_t aligned_size = (size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    void *base = libc_mmap(NULL, aligned_size, PROT_READ | PROT_WRITE, 
                           MAP_PRIVATE | MAP_ANON, -1, 0);
    if (base == MAP_FAILED) {
        return NULL;
    }

    if (apply_interleave_policy(base, aligned_size) != 0) {
        libc_munmap(base, aligned_size);
        return NULL;
    }

    record_seg((unsigned long)base, aligned_size, 1);
    return base;
}

void interweave_free(void *ptr)
{
    // 保留原实现（free 中直接使用 libc_munmap(size_to_free)）
    ensure_mapping_funcs();
    if (unlikely(!libc_munmap) || ptr == NULL)
        return;
    pthread_mutex_lock(&seg_lock);
    struct addr_seg *seg = find_seg((unsigned long)ptr);
    size_t size = seg ? seg_length(seg) : 0;
    pthread_mutex_unlock(&seg_lock);
    if (size > 0) {
        libc_munmap(ptr, size);
    }
    return;
}

void *interweave_calloc(size_t nmemb, size_t size)
{
    if (size != 0 && nmemb > SIZE_MAX / size) {
        errno = ENOMEM;
        return NULL;
    }
    
    size_t total = nmemb * size;
    void *ptr = interweave_malloc(total);
    
    if (likely(ptr)) {
        memset(ptr, 0, total);
    }
    return ptr;
}

void *interweave_realloc(void *ptr, size_t size)
{
    ensure_mapping_funcs();
    if (unlikely(!libc_munmap))
        return NULL;

    if (ptr == NULL)
        return interweave_malloc(size);
    
    if (size == 0) {
        interweave_free(ptr);
        return NULL;
    }
    
    /* Find old size */
    pthread_mutex_lock(&seg_lock);
    struct addr_seg *seg = find_seg((unsigned long)ptr);
    size_t old_size = seg ? seg_length(seg) : 0;
    pthread_mutex_unlock(&seg_lock);
    
    if (old_size == 0) {
        /* Not our allocation */
        return libc_realloc(ptr, size);
    }
    
    /* Allocate new memory */
    void *new_ptr = interweave_malloc(size);
    if (!new_ptr)
        return NULL;
    
    /* Copy data */
    memcpy(new_ptr, ptr, old_size < size ? old_size : size);
    
    /* Free old memory - need to clear seg first */
    int is_interleaved = 0;
    check_seg((unsigned long)ptr, &is_interleaved);
    libc_munmap(ptr, old_size);
    
    return new_ptr;
}

void *interweave_aligned_alloc(size_t alignment, size_t size)
{
    ensure_mapping_funcs();
    if (unlikely(!libc_mmap || !libc_munmap)) {
        errno = ENOSYS;
        return NULL;
    }
    if (unlikely(alignment == 0 || !is_pow2(alignment))) {
        errno = EINVAL;
        return NULL;
    }
    
    /* Ensure alignment is at least PAGE_SIZE for interleaving to work correctly */
    if (alignment < PAGE_SIZE)
        alignment = PAGE_SIZE;
    
    /* For aligned allocation with interleaving, we need to ensure
     * the base address is aligned */
    size_t aligned_size = (size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    size_t map_size = aligned_size + alignment;
    
    void *base = libc_mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANON, -1, 0);
    if (base == MAP_FAILED)
        return NULL;
    
    /* Find aligned address within the mapping */
    uintptr_t base_addr = (uintptr_t)base;
    uintptr_t aligned_addr = (base_addr + alignment - 1) & ~(alignment - 1);
    void *aligned_ptr = (void *)aligned_addr;
    
    /* Unmap the unused portions */
    if (aligned_addr > base_addr) {
        libc_munmap(base, aligned_addr - base_addr);
    }
    size_t end_excess = (base_addr + map_size) - (aligned_addr + aligned_size);
    if (end_excess > 0) {
        libc_munmap((void *)(aligned_addr + aligned_size), end_excess);
    }
    
    if (apply_interleave_policy(aligned_ptr, aligned_size) != 0) {
        libc_munmap(aligned_ptr, aligned_size);
        return NULL;
    }

    record_seg((unsigned long)aligned_ptr, aligned_size, 1);
    return aligned_ptr;
}

void *aligned_alloc(size_t alignment, size_t size)
{
    if (unlikely(!libc_aligned_alloc)) {
        libc_aligned_alloc = (void * (*)(size_t, size_t))dlsym(RTLD_NEXT, "aligned_alloc");
        if (!libc_aligned_alloc) {
            errno = ENOSYS;
            return NULL;
        }
    }
    if (size > INTERLEAVE_THRESHOLD) {
        size_t callchain_size_local = 0;
        void *callchain_strings_local[CALLCHAIN_SIZE];
        if (!_in_trace) {
            get_trace(&callchain_size_local, callchain_strings_local);
        }
        if (callchain_size_local >= 4) {
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings_local, callchain_size_local);
            int pf = get_place_flag_by_hash(callstack_hash);
            if (pf == 2) {
                void *addr = haligned_alloc(alignment, size);
                if (addr) {
                    record_seg((unsigned long)addr, 0, 2);
                    return addr;
                }
                CXL_LOG("haligned_alloc failed, falling back to libc aligned_alloc");
            } else if (pf == 1) {
                void *addr = interweave_aligned_alloc(alignment, size);
                if (addr) {
                    return addr;
                }
                CXL_LOG("interweave_aligned_alloc failed, falling back to libc aligned_alloc");
            }
        }
    }
    return libc_aligned_alloc(alignment, size);
}

int interweave_posix_memalign(void **memptr, size_t alignment, size_t sz)
{
    if (!memptr || alignment < sizeof(void *) || alignment % sizeof(void *) || !is_pow2(alignment)) {
        return EINVAL;
    }

    void *ptr = interweave_aligned_alloc(alignment, sz);
    if (!ptr) {
        return ENOMEM;
    }

    *memptr = ptr;
    return 0;
}



//Copy from hmsdk hmalloc.c
void *hmmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    void *new_addr = mmap(addr, length, prot, flags, fd, offset);
    if (unlikely(new_addr == MAP_FAILED))
        return MAP_FAILED;
    unsigned long nodemask = (1UL << NUMA_LOCAL_NODE) | (1UL << NUMA_CXL_NODE);
    unsigned long maxnode = sizeof(nodemask) * 8;
    long ret = mbind(new_addr, length, MPOL_BIND, &nodemask, maxnode, 0);
    if (unlikely(ret)) {
        int mbind_errno = errno;
        munmap(new_addr, length);
        errno = mbind_errno;
        return NULL;
    }
    
    return new_addr;
}

int hmunmap(void *addr, size_t length) {
    return munmap(addr, length);
}

void *hmalloc(size_t size) {
    void *ptr;
    ptr = mallocx(size, MALLOCX_ARENA(arena_index) | MALLOCX_TCACHE_NONE);
    if (errno == ENOMEM)
        return NULL;
    return ptr;
}

void hfree(void *ptr) {
    if (unlikely(ptr == NULL))
        return;
    dallocx(ptr, MALLOCX_ARENA(arena_index) | MALLOCX_TCACHE_NONE);
}

void *hcalloc(size_t nmemb, size_t size) {
    void *ptr = hmalloc(nmemb * size);

    if (likely(ptr))
        memset(ptr, 0, nmemb * size);
    return ptr;
}

void *hrealloc(void *ptr, size_t size) {
    if (ptr == NULL)
        return hmalloc(size);

    if (size == 0) {
        hfree(ptr);
        return NULL;
    }
    return rallocx(ptr, size, MALLOCX_ARENA(arena_index) | MALLOCX_TCACHE_NONE);
}

void *haligned_alloc(size_t alignment, size_t size) {
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
    if (unlikely(ptr == NULL))
        return 0;
    return sallocx(ptr, 0);
}
