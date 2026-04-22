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
#include <fcntl.h>
#include <sys/types.h>
#include <sys/syscall.h>

/*
 * New kernel interface: set_mempolicy2 / mbind2 (kernel >= 6.17-per_object_interleave).
 * These syscalls embed il_weights directly in the call, eliminating:
 *   - sysfs write-based weight configuration (global shared state, 2 syscalls)
 *   - the global interleave_cfg_lock mutex (serialization bottleneck)
 * Each mbind2 call is fully independent and thread-safe.
 */
#ifndef __NR_set_mempolicy2
#define __NR_set_mempolicy2  470
#endif
#ifndef __NR_mbind2
#define __NR_mbind2          471
#endif

/*
 * struct mempolicy_args: extensible argument struct for set_mempolicy2 / mbind2.
 * Mirrors the RUNNING kernel's uapi (6.17.0Soar-adv_migrate-per_object_interleave+).
 * NOTE: MEMPOLICY_ARGS_SIZE_VER0 == 40, so the struct must be exactly 40 bytes.
 * The kernel's binary includes il_phase_offset as an additional field beyond
 * what is visible in the installed mempolicy.h header.
 */
struct mempolicy_args {
    __u16   mode;
    __u16   mode_flags;
    __s32   home_node;
    __u64   pol_maxnode;
    __u64   pol_nodes;        /* __user pointer to unsigned long nodemask */
    __u64   il_weights;       /* __user pointer to __u8 per-node weight array, or 0 */
    __u64   il_phase_offset;  /* interleave phase offset (0 = default) */
};

#define MEMPOLICY_ARGS_SIZE_VER0 40  /* sizeof first published struct, matches kernel check */

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
static size_t interleave_min_size = 65536;  /* no-THP default: skip mmap/mbind2 for medium allocations */
static uint64_t obj_hashes[MAX_HASH_ENTRIES];
static int obj_place_flags[MAX_HASH_ENTRIES];          // 0: local, 1..4: interleave tiers, 5: CXL cold
static long long obj_heate_cnts[MAX_HASH_ENTRIES];     // optional
static double obj_bw_scores[MAX_HASH_ENTRIES];         // optional
static int obj_count = 0;

struct hash_entry {
    uint64_t hash;
    int place_flag;
};

static struct hash_entry sorted_entries[MAX_HASH_ENTRIES];
static int sorted_count = 0;

static int enable_madvise = 1;  /* Controls whether madvise is enabled (default: enabled) */

static int interleave_node0_weights[6] = {0, 5, 4, 3, 2, 0};
static int interleave_node1_weights[6] = {0, 1, 1, 1, 1, 0};
static int fixed_interleave_mode = 0;

/*
 * mbind2 availability probe.
 *   0  = not probed yet
 *   1  = available (use mbind2 fast path)
 *  -1  = ENOSYS, fall back to sysfs+mbind
 */
static int mbind2_available = 0;
static pthread_once_t mbind2_probe_once = PTHREAD_ONCE_INIT;

static int prearm_thread_policy = 0;
static __thread int tl_setpol2_ok = 1;  /* assume available until proven wrong */

/* Legacy sysfs state (used only when mbind2 is unavailable) */
static pthread_mutex_t interleave_cfg_lock = PTHREAD_MUTEX_INITIALIZER;
static const char *weighted_interleave_node0_path = "/sys/kernel/mm/mempolicy/weighted_interleave/node0";
static const char *weighted_interleave_node1_path = "/sys/kernel/mm/mempolicy/weighted_interleave/node1";
static int weighted_interleave_node0_fd = -1;
static int weighted_interleave_node1_fd = -1;
static int current_weight_node0 = -1;
static int current_weight_node1 = -1;

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

static int open_weight_sysfs_fd(const char *path)
{
    int fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    return fd;
}

static void init_weight_sysfs_fds(void)
{
    if (weighted_interleave_node0_fd < 0) {
        weighted_interleave_node0_fd = open_weight_sysfs_fd(weighted_interleave_node0_path);
    }
    if (weighted_interleave_node1_fd < 0) {
        weighted_interleave_node1_fd = open_weight_sysfs_fd(weighted_interleave_node1_path);
    }
}

static int write_int_to_fd(int fd, int value)
{
    if (fd < 0) return -1;

    char buf[16];
    int len = snprintf(buf, sizeof(buf), "%d", value);
    ssize_t written = pwrite(fd, buf, len, 0);
    return (written == len) ? 0 : -1;
}

static int write_int_to_sysfs(const char *path, int value)
{
    int fd = open_weight_sysfs_fd(path);
    if (fd < 0) return -1;
    int ret = write_int_to_fd(fd, value);
    close(fd);
    return ret;
}

static int write_weight_pair(int w0, int w1)
{
    init_weight_sysfs_fds();

    int ok0;
    int ok1;
    if (weighted_interleave_node0_fd >= 0 && weighted_interleave_node1_fd >= 0) {
        ok0 = write_int_to_fd(weighted_interleave_node0_fd, w0);
        ok1 = write_int_to_fd(weighted_interleave_node1_fd, w1);
    } else {
        ok0 = write_int_to_sysfs(weighted_interleave_node0_path, w0);
        ok1 = write_int_to_sysfs(weighted_interleave_node1_path, w1);
    }
    return (ok0 == 0 && ok1 == 0) ? 0 : -1;
}

static int hash_entry_cmp(const void *a, const void *b)
{
    const struct hash_entry *ea = (const struct hash_entry *)a;
    const struct hash_entry *eb = (const struct hash_entry *)b;
    return (ea->hash > eb->hash) - (ea->hash < eb->hash);
}

static int prepare_interleave_weights_for_flag_locked(int place_flag)
{
    if (place_flag < 1 || place_flag > 4) {
        return 0;
    }
    if (fixed_interleave_mode) {
        return 0;
    }

    int w0 = interleave_node0_weights[place_flag];
    int w1 = interleave_node1_weights[place_flag];
    if (current_weight_node0 == w0 && current_weight_node1 == w1) {
        return 0;
    }

    if (write_weight_pair(w0, w1) == 0) {
        current_weight_node0 = w0;
        current_weight_node1 = w1;
        return 0;
    }

    CXL_LOG("set weighted interleave failed for flag=%d (%d:%d), errno=%d", place_flag, w0, w1, errno);
    return -1;
}

/*
 * Legacy policy path: sysfs weight write + mbind (requires global mutex).
 * Used only when mbind2 is unavailable (ENOSYS on older kernels).
 */
static int create_weighted_interleave_policy_sysfs(void *addr, size_t len, int place_flag)
{
    unsigned long nodemask = (1UL << NUMA_LOCAL_NODE) | (1UL << NUMA_CXL_NODE);
    unsigned long maxnode = sizeof(nodemask) * 8;

    pthread_mutex_lock(&interleave_cfg_lock);
    int prep_rc = prepare_interleave_weights_for_flag_locked(place_flag);
    if (prep_rc != 0) {
        pthread_mutex_unlock(&interleave_cfg_lock);
        return -1;
    }

    if (mbind(addr, len, MPOL_WEIGHTED_INTERLEAVE, &nodemask, maxnode, 0) != 0) {
        CXL_LOG("mbind interleave failed: %s", strerror(errno));
        pthread_mutex_unlock(&interleave_cfg_lock);
        return -1;
    }
    pthread_mutex_unlock(&interleave_cfg_lock);

    if (place_flag >= 3) {
        if (madvise(addr, len, MADV_NOMIGRATE) != 0) {
            CXL_LOG("madvise MADV_NOMIGRATE failed: %s", strerror(errno));
            return -1;
        }
    }
    return 0;
}

/*
 * Probe thread: try a dummy mbind2 with an obviously-invalid address so it
 * fails with EFAULT (not ENOSYS) if the syscall exists.
 */
static void probe_mbind2(void)
{
    struct mempolicy_args probe_args = {
        .mode       = MPOL_WEIGHTED_INTERLEAVE,
        .mode_flags = 0,
        .home_node  = -1,
        .pol_maxnode = 0,
        .pol_nodes  = 0,
        .il_weights = 0,
    };
    long rc = syscall(__NR_mbind2, (unsigned long)0UL, (unsigned long)0UL,
                      &probe_args, MEMPOLICY_ARGS_SIZE_VER0, (unsigned int)0);
    if (rc == -1 && errno == ENOSYS) {
        mbind2_available = -1;
        CXL_LOG("mbind2 not available (ENOSYS), using legacy sysfs path");
    } else {
        /* Any other error (EINVAL, EFAULT, etc.) means the syscall exists. */
        mbind2_available = 1;
        CXL_LOG("mbind2 available: using lock-free inline-weight path");
    }
}

static int create_weighted_interleave_policy_mbind2(void *addr, size_t len, int place_flag)
{
    unsigned long nodemask = (1UL << NUMA_LOCAL_NODE) | (1UL << NUMA_CXL_NODE);
    uint8_t weights[2] = {
        (uint8_t)interleave_node0_weights[place_flag],
        (uint8_t)(fixed_interleave_mode ? 0 : interleave_node1_weights[place_flag]),
    };

    struct mempolicy_args args = {
        .mode           = MPOL_WEIGHTED_INTERLEAVE,
        .mode_flags     = 0,
        .home_node      = -1,
        .pol_maxnode    = (uint64_t)(sizeof(unsigned long) * 8),
        .pol_nodes      = (uint64_t)(uintptr_t)&nodemask,
        /* fixed_interleave_mode: pass NULL so kernel uses sysfs weights */
        .il_weights     = fixed_interleave_mode ? 0 :
                          (uint64_t)(uintptr_t)weights,
        .il_phase_offset = 0,
    };

    long rc = syscall(__NR_mbind2,
                      (unsigned long)(uintptr_t)addr,
                      (unsigned long)len,
                      &args, MEMPOLICY_ARGS_SIZE_VER0,
                      (unsigned int)0);
    if (rc != 0) {
        CXL_LOG("mbind2 failed for flag=%d: %s", place_flag, strerror(errno));
        return -1;
    }

    if (enable_madvise && place_flag == 3) {
        if (madvise(addr, len, MADV_NOMIGRATE) != 0) {
            CXL_LOG("madvise MADV_NOMIGRATE failed: %s", strerror(errno));
            return -1;
        }
    }
    return 0;
}

/*
 * Set the calling thread's mempolicy via set_mempolicy2 with inline weights.
 *
 * This path is intentionally opt-in. For the allocator's current plain mmap
 * path, mbind2 alone is sufficient because page placement is decided on later
 * faults after the VMA policy has already been installed.
 */
static void set_thread_interleave_policy(int place_flag)
{
     if (!prearm_thread_policy || !tl_setpol2_ok) return;

    unsigned long nodemask = (1UL << NUMA_LOCAL_NODE) | (1UL << NUMA_CXL_NODE);
    uint8_t weights[2] = {
        (uint8_t)interleave_node0_weights[place_flag],
        (uint8_t)interleave_node1_weights[place_flag],
    };
    struct mempolicy_args args = {
        .mode           = MPOL_WEIGHTED_INTERLEAVE,
        .mode_flags     = 0,
        .home_node      = -1,
        .pol_maxnode    = (uint64_t)(sizeof(unsigned long) * 8),
        .pol_nodes      = (uint64_t)(uintptr_t)&nodemask,
        .il_weights     = fixed_interleave_mode ? 0 :
                          (uint64_t)(uintptr_t)weights,
        .il_phase_offset = 0,
    };
    long rc = syscall(__NR_set_mempolicy2, &args, MEMPOLICY_ARGS_SIZE_VER0);
    if (rc != 0 && errno == ENOSYS) {
        tl_setpol2_ok = 0;  /* disable per-thread for this thread */
    }
}

/*
 * Restore calling thread's mempolicy to MPOL_DEFAULT after the mmap region
 * has been bound with mbind2.  This prevents the thread-local WI policy from
 * accidentally influencing subsequent non-interleave allocations.
 */
static void restore_thread_default_policy(void)
{
    if (!prearm_thread_policy || !tl_setpol2_ok) return;
    struct mempolicy_args def = {
        .mode           = MPOL_DEFAULT,
        .mode_flags     = 0,
        .home_node      = -1,
        .pol_maxnode    = 0,
        .pol_nodes      = 0,
        .il_weights     = 0,
        .il_phase_offset = 0,
    };
    (void)syscall(__NR_set_mempolicy2, &def, MEMPOLICY_ARGS_SIZE_VER0);
}

/*
 * Unified entry point: dispatches to the mbind2 fast path or the legacy
 * sysfs path depending on what the kernel supports.
 */
static int create_weighted_interleave_policy(void *addr, size_t len, int place_flag)
{
    if (place_flag < 1 || place_flag > 4) return 0;

    if (mbind2_available == 0) {
        pthread_once(&mbind2_probe_once, probe_mbind2);
    }

    if (mbind2_available == 1) {
        if (prearm_thread_policy) {
            set_thread_interleave_policy(place_flag);
        }
        int rc = create_weighted_interleave_policy_mbind2(addr, len, place_flag);
        if (prearm_thread_policy) {
            restore_thread_default_policy();
        }
        return rc;
    }

    /* Fallback: legacy sysfs + mbind (serialised via global mutex) */
    return create_weighted_interleave_policy_sysfs(addr, len, place_flag);
}

/* ---- Hash table for tracking interleave allocations (O(1) lookup) ---- */
#define SEG_HT_BITS     16
#define SEG_HT_SIZE     (1 << SEG_HT_BITS)   /* 65536 slots */
#define SEG_HT_MASK     (SEG_HT_SIZE - 1)
#define SEG_SHARD_BITS  8
#define SEG_SHARD_COUNT (1 << SEG_SHARD_BITS)
#define SEG_SHARD_SIZE  (SEG_HT_SIZE / SEG_SHARD_COUNT)
#define SEG_SHARD_MASK  (SEG_SHARD_COUNT - 1)
#define SEG_SHARD_SLOT_MASK (SEG_SHARD_SIZE - 1)
#define SEG_EMPTY       0UL
#define SEG_TOMBSTONE   1UL   /* 1 is never a valid mmap address */

struct seg_entry {
    unsigned long addr;   /* SEG_EMPTY=free, SEG_TOMBSTONE=deleted */
    size_t        size;
    int           place_flag;
};

static struct seg_entry seg_ht[SEG_HT_SIZE];
static int __thread _in_trace = 0;
static pthread_mutex_t seg_locks[SEG_SHARD_COUNT];

static inline unsigned int seg_hash(unsigned long a)
{
    /* mmap addresses are page-aligned; shift out the zero low bits */
    a >>= 12;
    a ^= a >> 16;
    a *= 0x45d9f3bU;
    a ^= a >> 16;
    return (unsigned int)(a & SEG_HT_MASK);
}

static inline unsigned int seg_shard_id(unsigned int hash)
{
    return hash >> (SEG_HT_BITS - SEG_SHARD_BITS);
}

static inline unsigned int seg_shard_base(unsigned int shard_id)
{
    return shard_id * SEG_SHARD_SIZE;
}

static inline unsigned int seg_shard_offset(unsigned int hash)
{
    return hash & SEG_SHARD_SLOT_MASK;
}

static inline pthread_mutex_t *seg_lock_for_hash(unsigned int hash)
{
    return &seg_locks[seg_shard_id(hash) & SEG_SHARD_MASK];
}
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
// Returns 0 when no main-binary frames found (library-only allocation).
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
            // Use offset relative to binary base (deterministic across runs)
            offsets[offset_count++] = (unsigned long)callchain[i] - (unsigned long)info.dli_fbase;
        }
    }

    // No main-binary frames → library-only allocation, skip hash lookup
    if (offset_count == 0) {
        return 0;
    }

    uint64_t hash = fnv1a_hash((const unsigned char *)offsets, offset_count * sizeof(unsigned long));
    hash ^= alloc_size;
    hash *= FNV_PRIME;
    return hash;
}

// Based on hash value, determine placement flag.
// Returns 0 (DRAM default) when hash is 0 or not found.
static inline int get_place_flag_by_hash(uint64_t hash)
{
    static uint64_t total_matched = 0;
    static uint64_t total_unmatched = 0;

    if (hash == 0) {
        return 0;  // library-only allocation, skip lookup
    }
    int lo = 0;
    int hi = sorted_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        uint64_t mid_hash = sorted_entries[mid].hash;
        if (mid_hash == hash) {
            total_matched++;
            return sorted_entries[mid].place_flag;
        }
        if (mid_hash < hash) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    total_unmatched++;
    if (total_unmatched <= 20) {
        CXL_LOG("not find var hash : 0x%016lx (matched: %lu, unmatched: %lu)",
                (unsigned long)hash, total_matched, total_unmatched);
        if (total_unmatched == 20) {
            CXL_LOG("(suppressing further 'not find' messages)");
        }
    }
    return 0;
}

static inline int should_try_interleave(size_t size)
{
    return size >= interleave_min_size;
}

/* Record allocation segment — O(1) average via hash table */
void record_seg(unsigned long addr, size_t size, int place_flag)
{
    unsigned int hash = seg_hash(addr);
    unsigned int shard_id = seg_shard_id(hash);
    unsigned int base = seg_shard_base(shard_id);
    unsigned int offset = seg_shard_offset(hash);

    pthread_mutex_t *seg_lock = seg_lock_for_hash(hash);
    pthread_mutex_lock(seg_lock);
    for (unsigned int i = 0; i < SEG_SHARD_SIZE; i++) {
        unsigned int pos = base + ((offset + i) & SEG_SHARD_SLOT_MASK);
        unsigned long a = seg_ht[pos].addr;
        if (a == SEG_EMPTY || a == SEG_TOMBSTONE) {
            seg_ht[pos].addr = addr;
            seg_ht[pos].size = size;
            seg_ht[pos].place_flag = place_flag;
            pthread_mutex_unlock(seg_lock);
            return;
        }
    }
    pthread_mutex_unlock(seg_lock);
    CXL_LOG("Warning: seg shard %u full, cannot record segment", shard_id);
}

/* Find entry by address — O(1) average. Caller must hold the shard lock. */
static struct seg_entry *find_seg(unsigned long addr)
{
    unsigned int hash = seg_hash(addr);
    unsigned int base = seg_shard_base(seg_shard_id(hash));
    unsigned int offset = seg_shard_offset(hash);

    for (unsigned int i = 0; i < SEG_SHARD_SIZE; i++) {
        unsigned int pos = base + ((offset + i) & SEG_SHARD_SLOT_MASK);
        unsigned long a = seg_ht[pos].addr;
        if (a == addr)
            return &seg_ht[pos];
        if (a == SEG_EMPTY)
            return NULL;   /* not found — empty slot ends the probe chain */
        /* skip tombstones */
    }
    return NULL;
}

static inline void clear_seg(struct seg_entry *seg)
{
    if (seg) {
        seg->addr = SEG_TOMBSTONE;
        seg->size = 0;
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
    unsigned int hash = seg_hash(addr);
    pthread_mutex_t *seg_lock = seg_lock_for_hash(hash);
    pthread_mutex_lock(seg_lock);
    struct seg_entry *seg = find_seg(addr);
    size_t size_to_free = 0;
    if (seg) {
        size_to_free = seg->size;
        if (place_flag) {
            *place_flag = seg->place_flag;
        }
        clear_seg(seg);
    } else if (place_flag) {
        *place_flag = 0;
    }
    pthread_mutex_unlock(seg_lock);
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

    unsigned int hash = seg_hash((unsigned long)ptr);
    pthread_mutex_t *seg_lock = seg_lock_for_hash(hash);
    pthread_mutex_lock(seg_lock);
    struct seg_entry *seg = find_seg((unsigned long)ptr);
    size_t sz = seg ? seg->size : 0;
    pthread_mutex_unlock(seg_lock);

    if (sz > 0) {
        return sz;
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

    if (sz > INTERLEAVE_THRESHOLD && should_try_interleave(sz)) {
        void *addr;
        size_t callchain_size = 0;
        void *callchain_strings[CALLCHAIN_SIZE];

        if (!_in_trace) {
            get_trace(&callchain_size, callchain_strings);
        }
        if (callchain_size >= 4) {
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings, callchain_size, sz);
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

    if (total > INTERLEAVE_THRESHOLD && should_try_interleave(total)) {
        size_t callchain_size = 0;
        void *callchain_strings[CALLCHAIN_SIZE];

        if (!_in_trace) {
            get_trace(&callchain_size, callchain_strings);
        }
        if (callchain_size >= 4) {
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings, callchain_size, total);
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

    unsigned int hash = seg_hash((unsigned long)ptr);
    pthread_mutex_t *seg_lock = seg_lock_for_hash(hash);
    pthread_mutex_lock(seg_lock);
    struct seg_entry *seg = find_seg((unsigned long)ptr);
    int pf = seg ? seg->place_flag : 0;
    pthread_mutex_unlock(seg_lock);
    
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
    if (sz > INTERLEAVE_THRESHOLD && should_try_interleave(sz)) {
        size_t callchain_size = 0;
        void *callchain_strings[CALLCHAIN_SIZE];

        if (!_in_trace) {
            get_trace(&callchain_size, callchain_strings);
        }
        if (callchain_size >= 4) {
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings, callchain_size, sz);
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
    if (sz > INTERLEAVE_THRESHOLD && should_try_interleave(sz)) {
        size_t callchain_size = 0;
        void *callchain_strings[CALLCHAIN_SIZE];

        if (!_in_trace) {
            get_trace(&callchain_size, callchain_strings);
        }
        if (callchain_size >= 4) {
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings, callchain_size, sz);
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
    const char *interleave_min_size_env = getenv("CXL_MALLOC_INTERLEAVE_MIN_SIZE");
    if (interleave_min_size_env) {
        char *endptr = NULL;
        unsigned long long parsed = strtoull(interleave_min_size_env, &endptr, 10);
        if (endptr != interleave_min_size_env && *endptr == '\0' && parsed >= INTERLEAVE_THRESHOLD) {
            interleave_min_size = (size_t)parsed;
        }
        CXL_LOG("CXL_MALLOC_INTERLEAVE_MIN_SIZE=%zu", interleave_min_size);
    }

    /* Read madvise enable flag from environment variable */
    const char *enable_madvise_env = getenv("CXL_MALLOC_ENABLE_MADVISE");
    if (enable_madvise_env) {
        enable_madvise = (atoi(enable_madvise_env) != 0) ? 1 : 0;
        CXL_LOG("CXL_MALLOC_ENABLE_MADVISE=%d", enable_madvise);
    }
    const char *prearm_thread_policy_env = getenv("CXL_MALLOC_PREARM_THREAD_POLICY");
    if (prearm_thread_policy_env) {
        prearm_thread_policy = (atoi(prearm_thread_policy_env) != 0) ? 1 : 0;
        CXL_LOG("CXL_MALLOC_PREARM_THREAD_POLICY=%d", prearm_thread_policy);
    }
    const char *fixed_interleave_env = getenv("CXL_MALLOC_FIXED_INTERLEAVE");
    if (fixed_interleave_env) {
        fixed_interleave_mode = (atoi(fixed_interleave_env) != 0) ? 1 : 0;
        CXL_LOG("CXL_MALLOC_FIXED_INTERLEAVE=%d", fixed_interleave_mode);
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

    for (int i = 0; i < SEG_SHARD_COUNT; i++) {
        pthread_mutex_init(&seg_locks[i], NULL);
    }

    /* Probe mbind2 availability now so the first hot-path call is free. */
    pthread_once(&mbind2_probe_once, probe_mbind2);

    /* Open sysfs weight fds only when falling back to the legacy path. */
    if (mbind2_available != 1) {
        init_weight_sysfs_fds();
    }

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
    CXL_LOG("Loaded %d entries from rank file: %s", obj_count, filename);
    for (int i = 0; i < obj_count; i++) {
        sorted_entries[i].hash = obj_hashes[i];
        sorted_entries[i].place_flag = obj_place_flags[i];
    }
    sorted_count = obj_count;
    qsort(sorted_entries, sorted_count, sizeof(sorted_entries[0]), hash_entry_cmp);
}

__attribute__((destructor)) static void m_fini(void)
{
    /* Close sysfs fds only when we actually opened them (legacy path). */
    if (mbind2_available != 1) {
        if (weighted_interleave_node0_fd >= 0) {
            close(weighted_interleave_node0_fd);
            weighted_interleave_node0_fd = -1;
        }
        if (weighted_interleave_node1_fd >= 0) {
            close(weighted_interleave_node1_fd);
            weighted_interleave_node1_fd = -1;
        }
    }
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



void *interleave_malloc_with_flag(size_t size, int place_flag)
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

    if (create_weighted_interleave_policy(base, aligned_size, place_flag) != 0) {
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

/*
 * interleave_aligned_alloc_with_flag: allocates aligned memory with interleave policy
 */
void *interleave_aligned_alloc_with_flag(size_t alignment, size_t size, int place_flag)
{
    ensure_mapping_funcs();
    if (unlikely(!libc_mmap || !libc_munmap)) {
        errno = ENOSYS;
        return NULL;
    }

    // Validate alignment (must be power of 2 and multiple of sizeof(void*))
    if (alignment < sizeof(void *) || (alignment & (alignment - 1)) != 0) {
        errno = EINVAL;
        return NULL;
    }

    /* Allocate a contiguous virtual memory region with alignment */
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
    if (aligned_size < size) { // Overflow check
        errno = ENOMEM;
        return NULL;
    }

    // Add padding for alignment
    size_t alloc_size = aligned_size + alignment;
    
    void *base = libc_mmap(NULL, alloc_size, PROT_READ | PROT_WRITE, 
                           MAP_PRIVATE | MAP_ANON, -1, 0);
    if (base == MAP_FAILED) {
        return NULL;
    }

    // Calculate aligned address
    uintptr_t base_addr = (uintptr_t)base;
    uintptr_t aligned_addr = (base_addr + alignment - 1) & ~(alignment - 1);
    void *ptr = (void *)aligned_addr;

    // Calculate offset and unmap unused regions
    size_t offset = aligned_addr - base_addr;
    if (offset > 0) {
        libc_munmap(base, offset);
    }
    
    size_t tail = alloc_size - offset - aligned_size;
    if (tail > 0) {
        libc_munmap((void *)(aligned_addr + aligned_size), tail);
    }

    if (create_weighted_interleave_policy(ptr, aligned_size, place_flag) != 0) {
        libc_munmap(ptr, aligned_size);
        return NULL;
    }

    record_seg((unsigned long)ptr, aligned_size, place_flag);
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
        unsigned int hash = seg_hash((unsigned long)ptr);
        pthread_mutex_t *seg_lock = seg_lock_for_hash(hash);
        pthread_mutex_lock(seg_lock);
        struct seg_entry *seg = find_seg((unsigned long)ptr);
        size_t old_size = seg ? seg->size : 0;
        pthread_mutex_unlock(seg_lock);
        if (old_size > 0) {
            check_seg((unsigned long)ptr, NULL);
            libc_munmap(ptr, old_size);
        }
        return NULL;
    }

    /* Find old size */
    unsigned int hash = seg_hash((unsigned long)ptr);
    pthread_mutex_t *seg_lock = seg_lock_for_hash(hash);
    pthread_mutex_lock(seg_lock);
    struct seg_entry *seg = find_seg((unsigned long)ptr);
    size_t old_size = seg ? seg->size : 0;
    int old_place_flag = seg ? seg->place_flag : 1;
    pthread_mutex_unlock(seg_lock);
    
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
    if (size > INTERLEAVE_THRESHOLD && should_try_interleave(size)) {
        size_t callchain_size = 0;
        void *callchain_strings[CALLCHAIN_SIZE];
        if (!_in_trace) {
            get_trace(&callchain_size, callchain_strings);
        }
        if (callchain_size >= 4) {
            uint64_t callstack_hash = compute_callstack_hash(callchain_strings, callchain_size, size);
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
