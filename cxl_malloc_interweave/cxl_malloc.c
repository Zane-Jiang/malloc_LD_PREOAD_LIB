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

struct addr_seg {
    long unsigned start;
    long unsigned end;
    int is_interleaved;
};

static struct addr_seg addr_segs[MAX_OBJECTS];
static pthread_mutex_t seg_lock = PTHREAD_MUTEX_INITIALIZER;


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


/* Record allocation segment */
void record_seg(unsigned long addr, size_t size, int is_interleaved)
{
    pthread_mutex_lock(&seg_lock);
    for (int i = 0; i < MAX_OBJECTS; i++) {
        struct addr_seg *seg = &addr_segs[i];
        if (seg->start == 0 && seg->end == 0) {
            seg->start = addr;
            seg->end = addr + size;
            seg->is_interleaved = is_interleaved;
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
        seg->is_interleaved = 0;
    }
}

size_t check_seg(unsigned long addr, int *is_interleaved)
{
    pthread_mutex_lock(&seg_lock);
    struct addr_seg *seg = find_seg(addr);
    size_t size_to_free = 0;
    if (seg) {
        size_to_free = seg_length(seg);
        if (is_interleaved) {
            *is_interleaved = seg->is_interleaved;
        }
        clear_seg(seg);
    } else if (is_interleaved) {
        *is_interleaved = 0;
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

    if (sz > 0)
        return sz;

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
        void *addr = interweave_malloc(sz);
        if (addr) {
            return addr;
        }
        CXL_LOG("interweave_malloc failed for size %zu, falling back to libc_malloc", sz);
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
        void *addr = interweave_calloc(nmemb, size);
        if (addr) {
            return addr;
        }
        CXL_LOG("interweave_calloc failed, falling back to libc_calloc");
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
    size_t old_size = 0;
    if (seg) {
        old_size = seg_length(seg);
    }
    pthread_mutex_unlock(&seg_lock);
    
    if (old_size > 0) {
        return interweave_realloc(ptr, size);
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
        void *addr = interweave_aligned_alloc(align, sz);
        if (addr) {
            return addr;
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
        int ret = interweave_posix_memalign(ptr, align, sz);
        if (ret == 0) {
            return 0;
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

    int is_interleaved = 0;
    size_t size_to_free = check_seg((unsigned long)p, &is_interleaved);
    
    if (size_to_free > 0) {
        interweave_free(p);
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
 }
 



/*
 * Interweave malloc: allocates memory and binds pages according to interleave ratio.
 */
static void ensure_mapping_funcs(void)
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
    ensure_mapping_funcs();
    if (unlikely(!libc_munmap) || ptr == NULL)
        return;

    /* Find the segment to get the size */
    pthread_mutex_lock(&seg_lock);
    struct addr_seg *seg = find_seg((unsigned long)ptr);
    size_t size = 0;
    if (seg) {
        size = seg_length(seg);
        /* Note: segment already cleared by check_seg in free() */
    }
    pthread_mutex_unlock(&seg_lock);
    
    libc_munmap(ptr, size);
    return ;
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
        void *addr = interweave_aligned_alloc(alignment, size);
        if (addr) {
            return addr;
        }
        CXL_LOG("interweave_aligned_alloc failed, falling back to libc aligned_alloc");
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


