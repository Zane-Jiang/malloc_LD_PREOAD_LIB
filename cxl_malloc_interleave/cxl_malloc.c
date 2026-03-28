#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define __USE_GNU

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
#include <limits.h>
#include <numaif.h>

#define MADV_NOMIGRATE 26

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


#define FNV_PRIME (1099511628211ULL)

#define MAX_OBJECTS          30000
#define INTERLEAVE_THRESHOLD 4096  /* Minimum size for interleave allocation */
#define MAX_HASH_ENTRIES 1000
static uint64_t obj_hashes[MAX_HASH_ENTRIES];
static int obj_place_flags[MAX_HASH_ENTRIES];          // 0: local, 1..4: interleave tiers, 5: CXL cold
static long long obj_heate_cnts[MAX_HASH_ENTRIES];     // optional
static double obj_bw_scores[MAX_HASH_ENTRIES];         // optional
static int obj_count = 0;
static int enable_madvise = 1;  /* Controls whether madvise is enabled (default: enabled) */

static int interleave_node0_weights[6] = {0, 5, 4, 3, 2, 0};
static int interleave_node1_weights[6] = {0, 1, 1, 1, 1, 0};
static int current_interleave_flag = -1;
static pthread_mutex_t interleave_cfg_lock = PTHREAD_MUTEX_INITIALIZER;

static int parse_ratio_pair(const char *s, int *a, int *b)
{
    if (!s || !a || !b) return -1;
    int x = -1, y = -1;
    if (sscanf(s, "%d:%d", &x, &y) != 2) return -1;
    if (x <= 0 || y <= 0) return -1;
    *a = x;
    *b = y;
    return 0;
}

static void load_interleave_ratio_env(void)
{
    const char *keys[4] = {
        "CXL_MALLOC_INTERLEAVE_L1",
        "CXL_MALLOC_INTERLEAVE_L2",
        "CXL_MALLOC_INTERLEAVE_L3",
        "CXL_MALLOC_INTERLEAVE_L4",
    };
    for (int i = 0; i < 4; i++) {
        const char *v = getenv(keys[i]);
        if (!v) continue;
        int n0 = 0, n1 = 0;
        if (parse_ratio_pair(v, &n0, &n1) == 0) {
            interleave_node0_weights[i + 1] = n0;
            interleave_node1_weights[i + 1] = n1;
            CXL_LOG("%s=%d:%d", keys[i], n0, n1);
        }
    }
}

static int write_int_to_sysfs(const char *path, int value)
{
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;
    int ret = fprintf(fp, "%d", value);
    fclose(fp);
    return (ret > 0) ? 0 : -1;
}

static int apply_interleave_weights_for_flag(int place_flag)
{
    if (place_flag < 1 || place_flag > 4) {
        return 0;
    }
    if (current_interleave_flag == place_flag) {
        return 0;
    }

    pthread_mutex_lock(&interleave_cfg_lock);
    if (current_interleave_flag != place_flag) {
        int w0 = interleave_node0_weights[place_flag];
        int w1 = interleave_node1_weights[place_flag];
        int ok0 = write_int_to_sysfs("/sys/kernel/mm/mempolicy/weighted_interleave/node0", w0);
        int ok1 = write_int_to_sysfs("/sys/kernel/mm/mempolicy/weighted_interleave/node1", w1);
        if (ok0 == 0 && ok1 == 0) {
               current_interleave_flag = place_flag;
        } else {
            CXL_LOG("set weighted interleave failed for flag=%d (%d:%d), errno=%d", place_flag, w0, w1, errno);
            pthread_mutex_unlock(&interleave_cfg_lock);
            return -1;
        }
    }
    pthread_mutex_unlock(&interleave_cfg_lock);
    return 0;
}

struct addr_seg {
    long unsigned start;
    long unsigned end;
    int place_flag;   // 0: libc, 1: interleave, 2: h-alloc
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
void *interleave_malloc_with_flag(size_t size, int place_flag);
void *interleave_calloc_with_flag(size_t nmemb, size_t size, int place_flag);
void *interleave_realloc(void *ptr, size_t size);
void *interleave_aligned_alloc_with_flag(size_t alignment, size_t size, int place_flag);
int interleave_posix_memalign_with_flag(void **memptr, size_t alignment, size_t sz, int place_flag);
void ensure_mapping_funcs(void);


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
        hash *= FNV_PRIME; // FNV prime
    }
    return hash;
}

// Compute deterministic callstack hash (based on offsets, not runtime addresses)
static uint64_t compute_callstack_hash(void **callchain, size_t size, size_t alloc_size)
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
        return fnv1a_hash((const unsigned char *)&alloc_size, sizeof(alloc_size));
    }
    
    // Compute hash of offsets first
    uint64_t hash = fnv1a_hash((const unsigned char *)offsets, offset_count * sizeof(unsigned long));
    
    hash ^= alloc_size;
    hash *= FNV_PRIME; // FNV prime
    
    return hash;
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
    pthread_mutex_unlock(&seg_lock);

    if (seg) {
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
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings_local, callchain_size_local, sz);
            int pf = get_place_flag_by_hash(callstack_hash);
            if (pf == 5 || pf == 0) {
                addr = libc_malloc(sz);
                if(addr) {
                    if (enable_madvise && pf == 5) madvise(addr, sz, MADV_COLD);
                    return addr;
                }
            } else if (pf >= 1 && pf <= 4) {
                addr = interleave_malloc_with_flag(sz, pf);
                if (addr) {
                    return addr;
                }
                CXL_LOG("interleave_malloc failed for size %zu, falling back to libc_malloc", sz);
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
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings_local, callchain_size_local, total);
            int pf = get_place_flag_by_hash(callstack_hash);
            if (pf == 5 || pf == 0) {
                void *addr = libc_calloc(nmemb, size);
                if (addr) {
                    if (enable_madvise && pf == 5) madvise(addr, total, MADV_COLD);
                    return addr;
                }
            } else if (pf >= 1 && pf <= 4) {
                void *addr = interleave_calloc_with_flag(nmemb, size, pf);
                if (addr) {
                    return addr;
                }
                CXL_LOG("interleave_calloc failed for size %zu, falling back to libc_calloc", total);
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
    
    if (pf >= 1 && pf <= 4) {
        return interleave_realloc(ptr, size);
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
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings_local, callchain_size_local, sz);
            int pf = get_place_flag_by_hash(callstack_hash);
            if (pf == 5 || pf == 0) {
                void *addr = libc_memalign(align, sz);
                if (addr) {
                    if (enable_madvise && pf == 5) madvise(addr, sz, MADV_COLD);
                    return addr;
                }
            } else if (pf >= 1 && pf <= 4) {
                void *addr = interleave_aligned_alloc_with_flag(align, sz, pf);
                if (addr) {
                    return addr;
                }
                CXL_LOG("interleave_aligned_alloc failed for size %zu, falling back to libc_memalign", sz);
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
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings_local, callchain_size_local, sz);
            int pf = get_place_flag_by_hash(callstack_hash);
            if (pf == 5 || pf == 0) {
                int ret = libc_posix_memalign(ptr, align, sz);
                if (ret == 0) {
                    if (enable_madvise && pf == 5) madvise(*ptr, sz, MADV_COLD);
                    return 0;
                }
            } else if (pf >= 1 && pf <= 4) {
                int ret = interleave_posix_memalign_with_flag(ptr, align, sz, pf);
                if (ret == 0) {
                    return 0;
                }
                CXL_LOG("interleave_posix_memalign failed for size %zu, falling back to libc", sz);
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
    
    if (place_flag >= 1 && place_flag <= 4) {
        ensure_mapping_funcs();
        if (likely(libc_munmap && size_to_free > 0)) {
            libc_munmap(p, size_to_free);
        }
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

__attribute__((constructor)) void m_init(void) {
    /* Read madvise enable flag from environment variable */
    const char *enable_madvise_env = getenv("CXL_MALLOC_ENABLE_MADVISE");
    if (enable_madvise_env) {
        enable_madvise = (atoi(enable_madvise_env) != 0) ? 1 : 0;
        CXL_LOG("CXL_MALLOC_ENABLE_MADVISE=%d", enable_madvise);
    }
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

    load_interleave_ratio_env();

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
        if (place_flag < 0) place_flag = 0;
        if (place_flag > 5) place_flag = 5;
        long long heate_cnt = (t3 && *t3) ? strtoll(t3, NULL, 10) : LLONG_MIN; // LLONG_MIN -> not provided
        double bw_score = (t4 && *t4) ? strtod(t4, NULL) : 0.0;

        obj_hashes[obj_count] = hash_value;
        obj_place_flags[obj_count] = place_flag;
        obj_heate_cnts[obj_count] = heate_cnt;
        obj_bw_scores[obj_count] = bw_score;
        obj_count++;
    }
    fclose(file);
}
 
/*
 * interleave malloc: allocates memory and binds pages according to interleave ratio.
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



static int apply_interleave_policy(void *addr, size_t len, int place_flag)
{
    unsigned long nodemask = (1UL << NUMA_LOCAL_NODE) | (1UL << NUMA_CXL_NODE);
    unsigned long maxnode = sizeof(nodemask) * 8;

    if (mbind(addr, len, MPOL_WEIGHTED_INTERLEAVE, &nodemask, maxnode, MPOL_MF_MOVE) != 0) {
        CXL_LOG("mbind interleave failed: %s", strerror(errno));
        return -1;
    }
    // Only set MADV_NOMIGRATE for tier >= 3 
    if (place_flag >= 3) {
        if (madvise(addr, len, MADV_NOMIGRATE) != 0) {
            CXL_LOG("madvise MADV_NOMIGRATE failed: %s", strerror(errno));
            return -1;
        }
    }
    return 0;
}

void *interleave_malloc_with_flag(size_t size, int place_flag)
{
    ensure_mapping_funcs();
    if (unlikely(!libc_mmap || !libc_munmap)) {
        errno = ENOSYS;
        return NULL;
    }

    // Apply the appropriate interleave weights for this place_flag
    if (place_flag >= 1 && place_flag <= 4) {
        apply_interleave_weights_for_flag(place_flag);
    }

    /* Allocate a contiguous virtual memory region */
    size_t aligned_size = (size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    void *base = libc_mmap(NULL, aligned_size, PROT_READ | PROT_WRITE, 
                           MAP_PRIVATE | MAP_ANON, -1, 0);
    if (base == MAP_FAILED) {
        return NULL;
    }

    if (apply_interleave_policy(base, aligned_size, place_flag) != 0) {
        libc_munmap(base, aligned_size);
        return NULL;
    }

    record_seg((unsigned long)base, aligned_size, place_flag);
    return base;
}

void *interleave_calloc_with_flag(size_t nmemb, size_t size, int place_flag)
{
    if (size != 0 && nmemb > SIZE_MAX / size) {
        errno = ENOMEM;
        return NULL;
    }
    
    size_t total = nmemb * size;
    void *ptr = interleave_malloc_with_flag(total, place_flag);
    
    if (likely(ptr)) {
        memset(ptr, 0, total);
    }
    return ptr;
}

void *interleave_realloc(void *ptr, size_t size)
{
    ensure_mapping_funcs();
    if (unlikely(!libc_munmap))
        return NULL;

    if (ptr == NULL)
        return interleave_malloc_with_flag(size, 1);
    
    if (size == 0) {
        // Free old memory
        pthread_mutex_lock(&seg_lock);
        struct addr_seg *seg = find_seg((unsigned long)ptr);
        size_t old_size = seg ? seg_length(seg) : 0;
        pthread_mutex_unlock(&seg_lock);
        if (old_size > 0) {
            check_seg((unsigned long)ptr, NULL);
            libc_munmap(ptr, old_size);
        }
        return NULL;
    }
    
    /* Find old size */
    pthread_mutex_lock(&seg_lock);
    struct addr_seg *seg = find_seg((unsigned long)ptr);
    size_t old_size = seg ? seg_length(seg) : 0;
    int old_place_flag = seg ? seg->place_flag : 1;
    pthread_mutex_unlock(&seg_lock);
    
    if (old_size == 0) {
        /* Not our allocation */
        return libc_realloc(ptr, size);
    }
    
    /* Allocate new memory with same place_flag */
    void *new_ptr = interleave_malloc_with_flag(size, old_place_flag);
    if (!new_ptr)
        return NULL;
    
    /* Copy data */
    memcpy(new_ptr, ptr, old_size < size ? old_size : size);
    
    /* Free old memory */
    check_seg((unsigned long)ptr, NULL);
    libc_munmap(ptr, old_size);
    
    return new_ptr;
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
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings_local, callchain_size_local, size);
            int pf = get_place_flag_by_hash(callstack_hash);
            if (pf == 5 || pf == 0) {
                void *addr = libc_aligned_alloc(alignment, size);
                if (addr) {
                    if (enable_madvise && pf == 5) madvise(addr, size, MADV_COLD);
                    return addr;
                }
            } else if (pf >= 1 && pf <= 4) {
                void *addr = interleave_aligned_alloc_with_flag(alignment, size, pf);
                if (addr) {
                    return addr;
                }
                CXL_LOG("interleave_aligned_alloc failed, falling back to libc aligned_alloc");
            }
        }
    }
    return libc_aligned_alloc(alignment, size);
}

int interleave_posix_memalign_with_flag(void **memptr, size_t alignment, size_t sz, int place_flag)
{
    if (!memptr || alignment < sizeof(void *) || alignment % sizeof(void *) || !is_pow2(alignment)) {
        return EINVAL;
    }

    void *ptr = interleave_aligned_alloc_with_flag(alignment, sz, place_flag);
    if (!ptr) {
        return ENOMEM;
    }

    *memptr = ptr;
    return 0;
}
