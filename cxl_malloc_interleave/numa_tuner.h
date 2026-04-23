#ifndef CXL_NUMA_TUNER_H
#define CXL_NUMA_TUNER_H

/*
 * Runtime dynamic NUMA-balancing tuner.
 *
 * Monitors local DRAM (node 0) memory bandwidth via uncore IMC perf events
 * and adaptively adjusts kernel NUMA-balancing sysctls so that page
 * promotion/migration intensity tracks the real-time bandwidth pressure:
 *
 *   high DRAM bandwidth (saturated)  -> slow migration down
 *       (raise hot_threshold_ms, lower promote_rate_limit_MBps,
 *        widen scan periods)
 *
 *   low DRAM bandwidth (headroom)    -> speed migration up
 *       (lower hot_threshold_ms, raise promote_rate_limit_MBps,
 *        narrow scan periods)
 *
 * The caller (cxl_malloc.c) reads environment variables and decides whether
 * to start the tuner.  numa_tuner.c itself never calls malloc (directly or
 * indirectly) so it is safe to use inside an LD_PRELOAD malloc wrapper.
 */

struct numa_tuner_cfg {
    int interval_ms;      /* sampling period (ms), min 50        */
    int bw_high_mbps;     /* BW above which promotion is throttled */
    int bw_low_mbps;      /* BW below which promotion is accelerated */
    int ema_alpha_pct;    /* EMA smoothing 1..100                */
};

void numa_tuner_init(const struct numa_tuner_cfg *cfg);
void numa_tuner_shutdown(void);

#endif /* CXL_NUMA_TUNER_H */
