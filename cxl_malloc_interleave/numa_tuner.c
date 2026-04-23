/*
 * Runtime dynamic NUMA-balancing tuner.
 *
 * Samples local DRAM (node 0) bandwidth once per interval from uncore_imc
 * PMUs using perf_event_open(2) and rewrites the kernel NUMA-balancing
 * sysctls so that promotion/migration intensity follows the measured
 * bandwidth pressure.
 *
 * IMPORTANT: This file must NEVER call malloc (directly or indirectly).
 * All I/O uses raw open/read/write/close syscalls.  Logging uses
 * write(STDERR_FILENO, ...) via a stack-based snprintf.  Directory
 * enumeration uses the getdents64 syscall instead of opendir/readdir.
 *
 * All configuration is passed in via struct numa_tuner_cfg from the
 * caller (cxl_malloc.c).  This file does NOT read environment variables.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "numa_tuner.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>      /* snprintf only (no FILE* I/O) */
#include <stdlib.h>     /* strtol, strtoul (malloc-free) */
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <linux/perf_event.h>

/* ----- logging helper (write(2) to stderr, no malloc) ----- */
/* #define NTU_LOG(fmt, ...) \
    do { \
        char _buf[512]; \
        int _n = snprintf(_buf, sizeof(_buf), "[numa_tuner] " fmt "\n", ##__VA_ARGS__); \
        if (_n > 0) { if ((size_t)_n > sizeof(_buf)-1) _n = (int)sizeof(_buf)-1; (void)write(STDERR_FILENO, _buf, (size_t)_n); } \
    } while (0)
*/
#define NTU_LOG(fmt, ...)  

#define NTU_WARN(fmt, ...) \
    do { \
        char _buf[512]; \
        int _n = snprintf(_buf, sizeof(_buf), "[numa_tuner] WARN: " fmt "\n", ##__VA_ARGS__); \
        if (_n > 0) { if ((size_t)_n > sizeof(_buf)-1) _n = (int)sizeof(_buf)-1; (void)write(STDERR_FILENO, _buf, (size_t)_n); } \
    } while (0)

/* ----- tunable defaults / clamps ----- */
#define DEFAULT_INTERVAL_MS      1000
#define DEFAULT_BW_HIGH_MBPS     60000
#define DEFAULT_BW_LOW_MBPS      30000
#define DEFAULT_EMA_ALPHA_PCT    40

/* hot_threshold_ms search range */
#define HOT_THRESH_MIN_MS        100
#define HOT_THRESH_MAX_MS        10000
#define HOT_THRESH_DEFAULT_MS    1000

/* promote_rate_limit_MBps search range */
#define PROMOTE_RATE_MIN_MBPS    256
#define PROMOTE_RATE_MAX_MBPS    131072
#define PROMOTE_RATE_DEFAULT     65536

/* scan period range */
#define SCAN_PERIOD_MIN_FLOOR    200
#define SCAN_PERIOD_MIN_CEIL     5000
#define SCAN_PERIOD_MAX_FLOOR    5000
#define SCAN_PERIOD_MAX_CEIL     60000
#define SCAN_PERIOD_MIN_DEFAULT  1000
#define SCAN_PERIOD_MAX_DEFAULT  60000

/* Each CAS transaction on modern Intel IMC = 64 bytes */
#define CAS_BYTES                64ULL

/* ----- sysctl file paths ----- */
static const char SYSCTL_HOT_THRESH[]   = "/proc/sys/kernel/numa_balancing_hot_threshold_ms";
static const char SYSCTL_PROMOTE_RATE[] = "/proc/sys/kernel/numa_balancing_promote_rate_limit_MBps";
static const char SYSCTL_SCAN_MIN[]     = "/proc/sys/kernel/numa_balancing_scan_period_min_ms";
static const char SYSCTL_SCAN_MAX[]     = "/proc/sys/kernel/numa_balancing_scan_period_max_ms";
static const char SYSCTL_ENABLE[]       = "/proc/sys/kernel/numa_balancing";

/* ----- module state ----- */
static pthread_t     g_thread;
static atomic_int    g_stop = 0;
static int           g_started = 0;

static int g_interval_ms   = DEFAULT_INTERVAL_MS;
static int g_bw_high_mbps  = DEFAULT_BW_HIGH_MBPS;
static int g_bw_low_mbps   = DEFAULT_BW_LOW_MBPS;
static int g_ema_alpha_pct = DEFAULT_EMA_ALPHA_PCT;

/* current sysctl values tracked by the tuner (to avoid redundant writes) */
static int g_cur_hot_thresh    = HOT_THRESH_DEFAULT_MS;
static int g_cur_promote_rate  = PROMOTE_RATE_DEFAULT;
static int g_cur_scan_min      = SCAN_PERIOD_MIN_DEFAULT;
static int g_cur_scan_max      = SCAN_PERIOD_MAX_DEFAULT;

/* saved initial values for restore on shutdown */
static int g_init_hot_thresh   = -1;
static int g_init_promote_rate = -1;
static int g_init_scan_min     = -1;
static int g_init_scan_max     = -1;

/* ----- perf counter bookkeeping ----- */
#define MAX_IMC_FDS 32
static int   g_imc_fds[MAX_IMC_FDS];
static int   g_imc_fd_count = 0;
static int   g_sysctl_writable = 1;  /* becomes 0 on first EACCES */

/* ===================================================================== */
/* small helpers  (all malloc-free)                                       */
/* ===================================================================== */

static int clamp_int(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* read a small text file into a stack buffer using raw open/read/close */
static int read_small_file(const char *path, char *buf, size_t bufsz)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    ssize_t n = read(fd, buf, bufsz - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = '\0';
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == ' ' || buf[n - 1] == '\t')) {
        buf[--n] = '\0';
    }
    return 0;
}

/* parse a decimal integer from a file (raw fd, no FILE*) */
static int read_int_file(const char *path, int *out)
{
    char buf[64];
    if (read_small_file(path, buf, sizeof(buf)) != 0) return -1;
    char *end = NULL;
    long v = strtol(buf, &end, 10);
    if (end == buf) return -1;
    *out = (int)v;
    return 0;
}

/* write a decimal integer to a file (raw fd, no FILE*) */
static int write_int_file(const char *path, int value)
{
    if (!g_sysctl_writable) return -1;
    int fd = open(path, O_WRONLY | O_TRUNC);
    if (fd < 0) {
        if (errno == EACCES || errno == EPERM) {
            g_sysctl_writable = 0;
            NTU_WARN("cannot write %s (permission denied); tuner observe-only.", path);
        } else if (errno == ENOENT) {
            NTU_LOG("skip %s (not present on this kernel)", path);
        }
        return -1;
    }
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%d\n", value);
    ssize_t w = write(fd, buf, (size_t)n);
    close(fd);
    return (w == n) ? 0 : -1;
}

static long long monotonic_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* simple atoi without locale (no malloc) */
static int simple_atoi(const char *s)
{
    int neg = 0, v = 0;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '-') { neg = 1; s++; }
    else if (*s == '+') { s++; }
    while (*s >= '0' && *s <= '9') { v = v * 10 + (*s - '0'); s++; }
    return neg ? -v : v;
}

/* ===================================================================== */
/* getdents64-based directory enumeration (no opendir/readdir/malloc)     */
/* ===================================================================== */

struct linux_dirent64 {
    uint64_t        d_ino;
    int64_t         d_off;
    unsigned short  d_reclen;
    unsigned char   d_type;
    char            d_name[];
};

/* ===================================================================== */
/* uncore_imc perf counter discovery                                      */
/* ===================================================================== */

static int parse_pmu_event_string(const char *s, struct perf_event_attr *attr)
{
    unsigned long event = 0, umask = 0;
    const char *p = s;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\n') p++;
        if (!*p) break;
        char key[32];
        size_t ki = 0;
        while (*p && *p != '=' && ki < sizeof(key) - 1) key[ki++] = *p++;
        key[ki] = '\0';
        if (*p != '=') return -1;
        p++;
        unsigned long val = strtoul(p, (char **)&p, 0);
        if (strcmp(key, "event") == 0) event = val;
        else if (strcmp(key, "umask") == 0) umask = val;
    }
    attr->config = (event & 0xff) | ((umask & 0xff) << 8);
    return 0;
}

static int parse_cpumask_first_cpu(const char *path)
{
    char buf[128];
    if (read_small_file(path, buf, sizeof(buf)) != 0) return -1;
    int cpu = -1;
    sscanf(buf, "%d", &cpu);
    return cpu;
}

static int cpu_is_on_node0(int cpu)
{
    if (cpu < 0) return 0;
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/node0", cpu);
    return access(path, F_OK) == 0;
}

static long perf_event_open_sys(struct perf_event_attr *attr, pid_t pid,
                                int cpu, int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static int open_imc_counter(int pmu_type, unsigned long config, int cpu)
{
    struct perf_event_attr pea;
    memset(&pea, 0, sizeof(pea));
    pea.type         = pmu_type;
    pea.size         = sizeof(pea);
    pea.config       = config;
    pea.disabled     = 0;
    pea.exclude_kernel = 0;
    pea.exclude_hv   = 0;

    int fd = (int)perf_event_open_sys(&pea, -1, cpu, -1, 0);
    return fd;
}

/*
 * Open all uncore_imc_* CAS read+write counters whose representative CPU
 * lives on NUMA node 0.  Uses getdents64 to avoid opendir (which mallocs).
 */
static int discover_imc_counters(void)
{
    int dirfd = open("/sys/bus/event_source/devices", O_RDONLY | O_DIRECTORY);
    if (dirfd < 0) {
        NTU_WARN("cannot open /sys/bus/event_source/devices");
        return 0;
    }

    g_imc_fd_count = 0;
    char dentbuf[4096];

    for (;;) {
        int nread = (int)syscall(SYS_getdents64, dirfd, dentbuf, sizeof(dentbuf));
        if (nread <= 0) break;

        int bpos = 0;
        while (bpos < nread) {
            struct linux_dirent64 *de = (struct linux_dirent64 *)(dentbuf + bpos);
            bpos += de->d_reclen;

            if (strncmp(de->d_name, "uncore_imc", 10) != 0) continue;
            if (g_imc_fd_count + 2 > MAX_IMC_FDS) goto done;

            char path[512];
            char buf[256];

            /* PMU type id */
            snprintf(path, sizeof(path),
                     "/sys/bus/event_source/devices/%s/type", de->d_name);
            if (read_small_file(path, buf, sizeof(buf)) != 0) continue;
            int pmu_type = simple_atoi(buf);

            /* cpumask */
            snprintf(path, sizeof(path),
                     "/sys/bus/event_source/devices/%s/cpumask", de->d_name);
            int cpu = parse_cpumask_first_cpu(path);
            if (cpu < 0) continue;
            if (!cpu_is_on_node0(cpu)) continue;

            static const char *evnames[2] = { "cas_count_read", "cas_count_write" };
            for (int k = 0; k < 2; k++) {
                snprintf(path, sizeof(path),
                         "/sys/bus/event_source/devices/%s/events/%s",
                         de->d_name, evnames[k]);
                if (read_small_file(path, buf, sizeof(buf)) != 0) continue;

                struct perf_event_attr pea;
                memset(&pea, 0, sizeof(pea));
                if (parse_pmu_event_string(buf, &pea) != 0) continue;

                int fd = open_imc_counter(pmu_type, pea.config, cpu);
                if (fd < 0) {
                    if (g_imc_fd_count == 0) {
                        NTU_WARN("perf_event_open failed for %s/%s on cpu %d "
                                 "(paranoid level too high or missing CAP_PERFMON)",
                                 de->d_name, evnames[k], cpu);
                    }
                    continue;
                }
                if (g_imc_fd_count < MAX_IMC_FDS) {
                    g_imc_fds[g_imc_fd_count++] = fd;
                } else {
                    close(fd);
                }
            }
        }
    }
done:
    close(dirfd);
    return g_imc_fd_count;
}

static void close_imc_counters(void)
{
    for (int i = 0; i < g_imc_fd_count; i++) {
        if (g_imc_fds[i] >= 0) close(g_imc_fds[i]);
    }
    g_imc_fd_count = 0;
}

static uint64_t read_imc_total(void)
{
    uint64_t sum = 0;
    for (int i = 0; i < g_imc_fd_count; i++) {
        uint64_t v = 0;
        if (read(g_imc_fds[i], &v, sizeof(v)) == (ssize_t)sizeof(v)) {
            sum += v;
        }
    }
    return sum;
}

/* ===================================================================== */
/* control policy                                                         */
/* ===================================================================== */

/*
 * Apply a single "adjustment pressure" in range [-1.0, +1.0]:
 *   +1.0 = maximum throttle (BW is saturated; slow migration down)
 *   -1.0 = maximum acceleration (plenty of headroom; migrate aggressively)
 *   0.0  = inside the deadband, hold current values.
 *
 * We map pressure linearly (on a log-ish scale) to each sysctl's search
 * range.  Values are only written when they actually change.
 */
static void apply_pressure(double pressure)
{
    /* hot_threshold_ms: throttle -> larger, accelerate -> smaller */
    int hot;
    if (pressure >= 0) {
        hot = (int)(HOT_THRESH_DEFAULT_MS +
                    pressure * (HOT_THRESH_MAX_MS - HOT_THRESH_DEFAULT_MS));
    } else {
        hot = (int)(HOT_THRESH_DEFAULT_MS +
                    pressure * (HOT_THRESH_DEFAULT_MS - HOT_THRESH_MIN_MS));
    }
    hot = clamp_int(hot, HOT_THRESH_MIN_MS, HOT_THRESH_MAX_MS);

    /* promote_rate_limit: throttle -> smaller, accelerate -> larger */
    int rate;
    if (pressure >= 0) {
        rate = (int)(PROMOTE_RATE_DEFAULT -
                     pressure * (PROMOTE_RATE_DEFAULT - PROMOTE_RATE_MIN_MBPS));
    } else {
        rate = (int)(PROMOTE_RATE_DEFAULT -
                     pressure * (PROMOTE_RATE_MAX_MBPS - PROMOTE_RATE_DEFAULT));
    }
    rate = clamp_int(rate, PROMOTE_RATE_MIN_MBPS, PROMOTE_RATE_MAX_MBPS);

    /* scan periods: throttle -> wider, accelerate -> narrower */
    int scan_min, scan_max;
    if (pressure >= 0) {
        scan_min = (int)(SCAN_PERIOD_MIN_DEFAULT +
                         pressure * (SCAN_PERIOD_MIN_CEIL - SCAN_PERIOD_MIN_DEFAULT));
        scan_max = (int)(SCAN_PERIOD_MAX_DEFAULT);
    } else {
        scan_min = (int)(SCAN_PERIOD_MIN_DEFAULT +
                         pressure * (SCAN_PERIOD_MIN_DEFAULT - SCAN_PERIOD_MIN_FLOOR));
        scan_max = (int)(SCAN_PERIOD_MAX_DEFAULT +
                         pressure * (SCAN_PERIOD_MAX_DEFAULT - SCAN_PERIOD_MAX_FLOOR));
    }
    scan_min = clamp_int(scan_min, SCAN_PERIOD_MIN_FLOOR, SCAN_PERIOD_MIN_CEIL);
    scan_max = clamp_int(scan_max, SCAN_PERIOD_MAX_FLOOR, SCAN_PERIOD_MAX_CEIL);
    if (scan_max < scan_min) scan_max = scan_min;

    if (g_init_hot_thresh > 0 && hot != g_cur_hot_thresh) {
        if (write_int_file(SYSCTL_HOT_THRESH, hot) == 0) g_cur_hot_thresh = hot;
    }
    if (g_init_promote_rate > 0 && rate != g_cur_promote_rate) {
        if (write_int_file(SYSCTL_PROMOTE_RATE, rate) == 0) g_cur_promote_rate = rate;
    }
    if (g_init_scan_min > 0 && scan_min != g_cur_scan_min) {
        if (write_int_file(SYSCTL_SCAN_MIN, scan_min) == 0) g_cur_scan_min = scan_min;
    }
    if (g_init_scan_max > 0 && scan_max != g_cur_scan_max) {
        if (write_int_file(SYSCTL_SCAN_MAX, scan_max) == 0) g_cur_scan_max = scan_max;
    }
}

/*
 * Turn a bandwidth reading (MB/s) into a [-1,+1] pressure value.
 * Below LOW            -> -1.0   (accelerate)
 * Above HIGH           -> +1.0   (throttle)
 * Between LOW and HIGH -> linear interpolation.
 */
static double bw_to_pressure(double bw_mbps)
{
    if (g_bw_high_mbps <= g_bw_low_mbps) return 0.0;
    if (bw_mbps <= g_bw_low_mbps)  return -1.0;
    if (bw_mbps >= g_bw_high_mbps) return +1.0;
    double mid   = 0.5 * (g_bw_high_mbps + g_bw_low_mbps);
    double halfw = 0.5 * (g_bw_high_mbps - g_bw_low_mbps);
    return (bw_mbps - mid) / halfw;  /* -1..+1 */
}

/* ===================================================================== */
/* monitor thread                                                         */
/* ===================================================================== */

static void *tuner_thread(void *arg)
{
    (void)arg;

    if (discover_imc_counters() == 0) {
        NTU_WARN("no usable uncore_imc counters on node 0; tuner disabled.");
        return NULL;
    }
    NTU_LOG("using %d uncore_imc counters on node 0", g_imc_fd_count);

    uint64_t last = read_imc_total();
    long long last_ns = monotonic_ns();
    double ema_mbps = -1.0;

    while (!atomic_load(&g_stop)) {
        struct timespec req = {
            .tv_sec  = g_interval_ms / 1000,
            .tv_nsec = (long)(g_interval_ms % 1000) * 1000000L,
        };
        nanosleep(&req, NULL);
        if (atomic_load(&g_stop)) break;

        uint64_t cur    = read_imc_total();
        long long cur_ns = monotonic_ns();
        double dt_s    = (cur_ns - last_ns) / 1e9;
        if (dt_s <= 0.0) { last = cur; last_ns = cur_ns; continue; }

        uint64_t delta_cas = cur - last;  /* wraparound on 64-bit is fine */
        double bytes_per_s = (double)delta_cas * (double)CAS_BYTES / dt_s;
        double mbps        = bytes_per_s / 1.0e6;
        last    = cur;
        last_ns = cur_ns;

        if (ema_mbps < 0.0) {
            ema_mbps = mbps;
        } else {
            double a = g_ema_alpha_pct / 100.0;
            ema_mbps = a * mbps + (1.0 - a) * ema_mbps;
        }

        double pressure = bw_to_pressure(ema_mbps);
        apply_pressure(pressure);

        NTU_LOG("bw=%.0f MB/s ema=%.0f MB/s pressure=%+.2f "
                "hot=%d rate=%d scan_min=%d scan_max=%d",
                mbps, ema_mbps, pressure,
                g_cur_hot_thresh, g_cur_promote_rate,
                g_cur_scan_min, g_cur_scan_max);
    }

    return NULL;
}

/* ===================================================================== */
/* public api                                                             */
/* ===================================================================== */

void numa_tuner_init(const struct numa_tuner_cfg *cfg)
{
    if (!cfg) return;

    g_interval_ms   = cfg->interval_ms;
    if (g_interval_ms < 50) g_interval_ms = 50;
    g_bw_high_mbps  = cfg->bw_high_mbps;
    g_bw_low_mbps   = cfg->bw_low_mbps;
    g_ema_alpha_pct = clamp_int(cfg->ema_alpha_pct, 1, 100);

    if (g_bw_high_mbps <= g_bw_low_mbps) {
        NTU_WARN("BW_HIGH(%d) <= BW_LOW(%d); tuner disabled",
                 g_bw_high_mbps, g_bw_low_mbps);
        return;
    }

    /* Warn (don't fail) if NUMA balancing itself is off. */
    int nb = 0;
    if (read_int_file(SYSCTL_ENABLE, &nb) == 0 && nb == 0) {
        NTU_WARN("numa_balancing=0; tuner will still run but has no effect "
                 "until numa_balancing is enabled.");
    }

    /* Snapshot initial sysctls so we can restore them on shutdown. */
    read_int_file(SYSCTL_HOT_THRESH,   &g_init_hot_thresh);
    read_int_file(SYSCTL_PROMOTE_RATE, &g_init_promote_rate);
    read_int_file(SYSCTL_SCAN_MIN,     &g_init_scan_min);
    read_int_file(SYSCTL_SCAN_MAX,     &g_init_scan_max);
    if (g_init_hot_thresh   > 0) g_cur_hot_thresh   = g_init_hot_thresh;
    if (g_init_promote_rate > 0) g_cur_promote_rate = g_init_promote_rate;
    if (g_init_scan_min     > 0) g_cur_scan_min     = g_init_scan_min;
    if (g_init_scan_max     > 0) g_cur_scan_max     = g_init_scan_max;

    NTU_LOG("init: interval=%dms bw_low=%d bw_high=%d ema_alpha=%d%% "
            "initial[hot=%d rate=%d scan_min=%d scan_max=%d]",
            g_interval_ms, g_bw_low_mbps, g_bw_high_mbps, g_ema_alpha_pct,
            g_cur_hot_thresh, g_cur_promote_rate,
            g_cur_scan_min, g_cur_scan_max);

    atomic_store(&g_stop, 0);
    if (pthread_create(&g_thread, NULL, tuner_thread, NULL) != 0) {
        NTU_WARN("pthread_create failed (errno=%d)", errno);
        return;
    }
    g_started = 1;
}

void numa_tuner_shutdown(void)
{
    if (!g_started) return;
    atomic_store(&g_stop, 1);
    pthread_join(g_thread, NULL);
    close_imc_counters();

    /* Best-effort restore of initial sysctl values. */
    if (g_init_hot_thresh   > 0) write_int_file(SYSCTL_HOT_THRESH,   g_init_hot_thresh);
    if (g_init_promote_rate > 0) write_int_file(SYSCTL_PROMOTE_RATE, g_init_promote_rate);
    if (g_init_scan_min     > 0) write_int_file(SYSCTL_SCAN_MIN,     g_init_scan_min);
    if (g_init_scan_max     > 0) write_int_file(SYSCTL_SCAN_MAX,     g_init_scan_max);

    NTU_LOG("shutdown complete");
    g_started = 0;
}
