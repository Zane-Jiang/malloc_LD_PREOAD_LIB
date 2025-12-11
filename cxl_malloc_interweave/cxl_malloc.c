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

/* global variables set by environment variables */
static unsigned long maxnode;

/* Interleave ratio configuration */
static int g_local_ratio = 1;   /* Default: 1 page on local */
static int g_remote_ratio = 1;  /* Default: 1 page on CXL */

/* Arena indices for local and CXL nodes */
static unsigned local_arena_index;
static unsigned cxl_arena_index;
static extent_hooks_t *local_hooks;
static extent_hooks_t *cxl_hooks;
static extent_hooks_t local_extent_hooks;
static extent_hooks_t cxl_extent_hooks;
static unsigned long local_nodemask;
static unsigned long cxl_nodemask;


/* Function declarations */
void *interweave_malloc(size_t size, int local_ratio, int remote_ratio);
void interweave_free(void *ptr);
void *interweave_calloc(size_t nmemb, size_t size, int local_ratio, int remote_ratio);
void *interweave_realloc(void *ptr, size_t size, int local_ratio, int remote_ratio);
void *interweave_aligned_alloc(size_t alignment, size_t size, int local_ratio, int remote_ratio);
int interweave_posix_memalign(void **memptr, size_t alignment, size_t size, int local_ratio, int remote_ratio);
// size_t interweave_malloc_usable_size(void *ptr);

void *local_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
void *cxl_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int hmunmap(void *addr, size_t length);


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


static pthread_once_t jemalloc_once = PTHREAD_ONCE_INIT;
static bool jemalloc_initialized = false;

static void setup_jemalloc(void) {

    size_t ulen = sizeof(unsigned);
    int err __unused = mallctl("arenas.create", &local_arena_index, &ulen, NULL, 0);
    assert(!err);
    {
        extent_hooks_t *hooks_ptr = &local_extent_hooks;
        char key[64];
        snprintf(key, sizeof(key), "arena.%u.extent_hooks", local_arena_index);
        err = mallctl(key, NULL, NULL, &hooks_ptr, sizeof(hooks_ptr));
        assert(!err);
        local_hooks = &local_extent_hooks;
    }

    err = mallctl("arenas.create", &cxl_arena_index, &ulen, NULL, 0);
    assert(!err);
    {
        extent_hooks_t *hooks_ptr = &cxl_extent_hooks;
        char key[64];
        snprintf(key, sizeof(key), "arena.%u.extent_hooks", cxl_arena_index);
        err = mallctl(key, NULL, NULL, &hooks_ptr, sizeof(hooks_ptr));
        assert(!err);
        cxl_hooks = &cxl_extent_hooks;
    }

    jemalloc_initialized = true;
    CXL_LOG("Initialized: local_arena=%u, cxl_arena=%u, ratio=%d:%d",
            local_arena_index, cxl_arena_index, g_local_ratio, g_remote_ratio);
}

static inline bool ensure_jemalloc_ready(void) {
    pthread_once(&jemalloc_once, setup_jemalloc);
    return jemalloc_initialized;
}


/* Get interleave ratio from environment or use default */
void get_interweave_ratio(int* local, int* remote)
{
    *local = g_local_ratio;
    *remote = g_remote_ratio;
}


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

/* 覆盖 malloc_usable_size：若 ptr 属于我们的 interleaved 区段則返回记录的大小，
   否则转发给 libc 的实现。 */
size_t malloc_usable_size(void *ptr)
{
    /* 首先解析 libc 实现（按需） */
    if (unlikely(!libc_malloc_usable_size)) {
        libc_malloc_usable_size = (size_t (*)(void *))dlsym(RTLD_NEXT, "malloc_usable_size");
        if (!libc_malloc_usable_size) {
            libc_malloc_usable_size = (size_t (*)(void *))dlsym(RTLD_NEXT, "__malloc_usable_size");
        }
        /* 如果都找不到，保守返回 0 */
        if (!libc_malloc_usable_size) {
            return 0;
        }
    }

    /* 如果是我们记录的 segment，返回记录的长度 */
    pthread_mutex_lock(&seg_lock);
    struct addr_seg *seg = find_seg((unsigned long)ptr);
    size_t sz = seg ? seg_length(seg) : 0;
    pthread_mutex_unlock(&seg_lock);

    if (sz > 0)
        return sz;

    /* 否则转发给 libc 实现 */
    return libc_malloc_usable_size(ptr);
}

/* 有些程序或 libc 内部可能直接调用 __malloc_usable_size，提供包装 */
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
    if (sz > INTERLEAVE_THRESHOLD && ensure_jemalloc_ready()) {
        int local_ratio, remote_ratio;
        get_interweave_ratio(&local_ratio, &remote_ratio);
        
        void *addr = interweave_malloc(sz, local_ratio, remote_ratio);
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

    if (total > INTERLEAVE_THRESHOLD && ensure_jemalloc_ready()) {
        int local_ratio, remote_ratio;
        get_interweave_ratio(&local_ratio, &remote_ratio);
        
        void *addr = interweave_calloc(nmemb, size, local_ratio, remote_ratio);
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
    
    if (old_size > 0 && ensure_jemalloc_ready()) {
        int local_ratio, remote_ratio;
        get_interweave_ratio(&local_ratio, &remote_ratio);
        return interweave_realloc(ptr, size, local_ratio, remote_ratio);
    }
    
    return libc_realloc(ptr, size);
}

void *memalign(size_t align, size_t sz)
{
    if (unlikely(!libc_memalign)) {
        libc_memalign = (void * (*)(size_t, size_t))dlsym(RTLD_NEXT, "memalign");
        if (!libc_memalign) return NULL;
    }
    if (sz > INTERLEAVE_THRESHOLD && ensure_jemalloc_ready()) {
        int local_ratio, remote_ratio;
        get_interweave_ratio(&local_ratio, &remote_ratio);
        
        void *addr = interweave_aligned_alloc(align, sz, local_ratio, remote_ratio);
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
    if (sz > INTERLEAVE_THRESHOLD && ensure_jemalloc_ready()) {
        int local_ratio, remote_ratio;
        get_interweave_ratio(&local_ratio, &remote_ratio);
        
        int ret = interweave_posix_memalign(ptr, align, sz, local_ratio, remote_ratio);
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


/* Extent hooks for local node */
static void *local_extent_alloc(extent_hooks_t *extent_hooks __unused, void *new_addr, size_t size,
                                size_t alignment __unused, bool *zero __unused, bool *commit __unused,
                                unsigned arena_ind __unused) {
    return local_mmap(new_addr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
}

static bool local_extent_dalloc(extent_hooks_t *extent_hooks __unused, void *addr, size_t size,
                                bool committed __unused, unsigned arena_ind __unused) {
    return hmunmap(addr, size) != 0;
}

static extent_hooks_t local_extent_hooks = {
    .alloc = local_extent_alloc,
    .dalloc = local_extent_dalloc,
};

/* Extent hooks for CXL node */
static void *cxl_extent_alloc(extent_hooks_t *extent_hooks __unused, void *new_addr, size_t size,
                              size_t alignment __unused, bool *zero __unused, bool *commit __unused,
                              unsigned arena_ind __unused) {
    return cxl_mmap(new_addr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
}

static bool cxl_extent_dalloc(extent_hooks_t *extent_hooks __unused, void *addr, size_t size,
                              bool committed __unused, unsigned arena_ind __unused) {
    return hmunmap(addr, size) != 0;
}

static extent_hooks_t cxl_extent_hooks = {
    .alloc = cxl_extent_alloc,
    .dalloc = cxl_extent_dalloc,
};



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

    setup_jemalloc();
    maxnode = numa_max_node() + 1;
    local_nodemask = 1UL << NUMA_LOCAL_NODE;
    cxl_nodemask = 1UL << NUMA_CXL_NODE;
 }
 
/* mmap with binding to local node */
void *local_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    void *new_addr = libc_mmap(addr, length, prot, flags, fd, offset);
    if (unlikely(new_addr == MAP_FAILED))
        return MAP_FAILED;

    long ret = mbind(new_addr, length, MPOL_BIND, &local_nodemask, maxnode, 0);
    if (unlikely(ret)) {
        int mbind_errno = errno;
        libc_munmap(new_addr, length);
        errno = mbind_errno;
        return MAP_FAILED;
    }
    return new_addr;
}

/* mmap with binding to CXL node */
void *cxl_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    void *new_addr = libc_mmap(addr, length, prot, flags, fd, offset);
    if (unlikely(new_addr == MAP_FAILED))
        return MAP_FAILED;

    long ret = mbind(new_addr, length, MPOL_BIND, &cxl_nodemask, maxnode, 0);
    if (unlikely(ret)) {
        int mbind_errno = errno;
        libc_munmap(new_addr, length);
        errno = mbind_errno;
        return MAP_FAILED;
    }
    return new_addr;
}

int hmunmap(void *addr, size_t length) {
    return libc_munmap(addr, length);
}


/*
 * Interweave malloc: allocates memory and binds pages according to interleave ratio.
 * For example, with local_ratio=3, remote_ratio=2:
 *   Pages 0,1,2 -> local node
 *   Pages 3,4 -> CXL node
 *   Pages 5,6,7 -> local node
 *   Pages 8,9 -> CXL node
 *   ...
 */
void *interweave_malloc(size_t size, int local_ratio, int remote_ratio)
{
    if (!ensure_jemalloc_ready() || remote_ratio == 0 || local_ratio == 0) {
        CXL_LOG("jemalloc not ready, fallback to libc_malloc");
        return libc_malloc(size);
    }

    
    /* Allocate a contiguous virtual memory region */
    size_t aligned_size = (size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    void *base = libc_mmap(NULL, aligned_size, PROT_READ | PROT_WRITE, 
                           MAP_PRIVATE | MAP_ANON, -1, 0);
    if (base == MAP_FAILED) {
        return NULL;
    }
    
    /* Calculate total pages and bind each page according to ratio */
    size_t num_pages = aligned_size / PAGE_SIZE;
    int cycle_length = local_ratio + remote_ratio;
    
    for (size_t i = 0; i < num_pages; i++) {
        void *page_addr = (char *)base + (i * PAGE_SIZE);
        int pos_in_cycle = i % cycle_length;
        unsigned long *mask;
        
        if (pos_in_cycle < local_ratio) {
            /* This page goes to local node */
            mask = &local_nodemask;
        } else {
            /* This page goes to CXL node */
            mask = &cxl_nodemask;
        }
        
        long ret = mbind(page_addr, PAGE_SIZE, MPOL_BIND, mask, maxnode, MPOL_MF_MOVE);
        if (ret != 0) {
            CXL_LOG("mbind failed for page %zu: %s", i, strerror(errno));
            /* Continue anyway, the page will use default policy */
        }
    }
    
    record_seg((unsigned long)base, aligned_size, 1);
    return base;
}

void interweave_free(void *ptr)
{
    if (unlikely(ptr == NULL))
        return;

    if (!ensure_jemalloc_ready()) {
        libc_free(ptr);
        return;
    }

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

void *interweave_calloc(size_t nmemb, size_t size, int local_ratio, int remote_ratio)
{
    if (size != 0 && nmemb > SIZE_MAX / size) {
        errno = ENOMEM;
        return NULL;
    }
    
    size_t total = nmemb * size;
    void *ptr = interweave_malloc(total, local_ratio, remote_ratio);
    
    if (likely(ptr)) {
        memset(ptr, 0, total);
    }
    return ptr;
}

void *interweave_realloc(void *ptr, size_t size, int local_ratio, int remote_ratio)
{
    if (!ensure_jemalloc_ready())
        return libc_realloc(ptr, size);
    
    if (ptr == NULL)
        return interweave_malloc(size, local_ratio, remote_ratio);
    
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
    void *new_ptr = interweave_malloc(size, local_ratio, remote_ratio);
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

void *interweave_aligned_alloc(size_t alignment, size_t size, int local_ratio, int remote_ratio)
{
    if (!ensure_jemalloc_ready())
        return libc_memalign(alignment, size);
    
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
    
    /* Apply interleave policy */
    size_t num_pages = aligned_size / PAGE_SIZE;
    int cycle_length = local_ratio + remote_ratio;
    
    if (cycle_length > 0) {
        for (size_t i = 0; i < num_pages; i++) {
            void *page_addr = (char *)aligned_ptr + (i * PAGE_SIZE);
            int pos_in_cycle = i % cycle_length;
            unsigned long *mask = (pos_in_cycle < local_ratio) ? &local_nodemask : &cxl_nodemask;
            mbind(page_addr, PAGE_SIZE, MPOL_BIND, mask, maxnode, MPOL_MF_MOVE);
        }
    }
    
    record_seg((unsigned long)aligned_ptr, aligned_size, 1);
    return aligned_ptr;
}

int interweave_posix_memalign(void **memptr, size_t alignment, size_t sz, int local_ratio, int remote_ratio)
{
    if (unlikely(alignment == 0 || !is_pow2(alignment) || alignment % sizeof(void *) != 0)) {
        *memptr = NULL;
        return EINVAL;
    }
    
    *memptr = interweave_aligned_alloc(alignment, sz, local_ratio, remote_ratio);
    
    if (*memptr == NULL) {
        return ENOMEM;
    }
    return 0;
}

// size_t interweave_malloc_usable_size(void *ptr)
// {
//     if (unlikely(ptr == NULL))
//         return 0;
    
//     pthread_mutex_lock(&seg_lock);
//     struct addr_seg *seg = find_seg((unsigned long)ptr);
//     size_t size = seg ? seg_length(seg) : 0;
//     pthread_mutex_unlock(&seg_lock);
    
//     if (size > 0)
//         return size;

//     if (ensure_jemalloc_ready())
//         return je_sallocx(ptr, 0);
//     return 0;
// }

