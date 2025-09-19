#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <dlfcn.h>
#include <limits.h>
#include <inttypes.h>
#include <float.h>
#include <sys/mman.h>

typedef struct MemoryBlock {
    void* ptr;
    uint64_t size;
    uint64_t id;
    char location[64];
    double alloc_ts;
    double free_ts;
} MemoryBlock;

typedef struct MemNode {
    uintptr_t addr;
    MemoryBlock block;
    struct MemNode* next;
} MemNode;

static MemNode* memory_list = NULL;
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static FILE* log_file = NULL;
static __thread bool in_hook = false;
static uint64_t global_id = 1;

static void* (*real_malloc)(size_t) = NULL;
static void  (*real_free)(void*) = NULL;
static void* (*real_calloc)(size_t, size_t) = NULL;
static void* (*real_mmap)(void*, size_t, int, int, int, off_t) = NULL;
static int   (*real_munmap)(void*, size_t) = NULL;

double start_time = 0.0;
double end_time = 0.0;

static double get_timestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void open_log_file() {
    const char* prof_env = getenv("HEAP_PROF_PATH");
    const char* prof_path = prof_env ? prof_env : "heap.prof";
    log_file = fopen(prof_path, "w");
    if (log_file) {
        fprintf(log_file, "   id       |       ptr       |  location  | size(Byte) | alloc_ts | free_ts\n");
        fflush(log_file);
    }
}


static void __attribute__((destructor)) output_final_stats() {
    pthread_mutex_lock(&mtx);
    MemNode* curr = memory_list;
    while (curr) {
        MemoryBlock* block = &curr->block;
        if (log_file) {
            fprintf(log_file, "%-10" PRIu64 " %14p %10s %-10" PRIu64 " %.6f %.6f\n",
                    block->id, block->ptr, block->location, block->size,
                    block->alloc_ts, block->free_ts);
        }

        MemNode* to_free = curr;
        curr = curr->next;
        real_free(to_free); 
    }
    memory_list = NULL;
    if (log_file) {
        end_time = get_timestamp();
        fprintf(log_file,"start_time: %.6f, end_time: %.6f", start_time, end_time);
        fflush(log_file);
        in_hook = true;
        fclose(log_file);//free的时候可能导致死锁
        log_file = NULL;
        in_hook = false;
    }
    pthread_mutex_unlock(&mtx);
}

static void register_alloc(void* ptr, size_t size, const char* location) {
    if (!ptr) return;
    if (in_hook) return;
    in_hook = true;

    MemNode* node = (MemNode*)real_malloc(sizeof(MemNode));
    if (!node) {
        in_hook = false;
        return;
    }

    node->addr = (uintptr_t)ptr;
    node->block.ptr = ptr;
    node->block.size = size;
    node->block.id = global_id++;
    strncpy(node->block.location, location ? location : "unknown", sizeof(node->block.location) - 1);
    node->block.location[sizeof(node->block.location) - 1] = '\0';
    node->block.alloc_ts = get_timestamp();
    node->block.free_ts = DBL_MAX ; 

    pthread_mutex_lock(&mtx);
    node->next = memory_list;
    memory_list = node;
    pthread_mutex_unlock(&mtx);

    in_hook = false;
}


static void unregister_alloc(void* ptr) {
    if (!ptr) return;
    if (in_hook) return;
    in_hook = true;

    pthread_mutex_lock(&mtx);
    MemNode** curr = &memory_list;
    while (*curr) {
        if ((*curr)->addr == (uintptr_t)ptr) {
            MemoryBlock* block = &(*curr)->block;
            block->free_ts = get_timestamp();

            if (log_file) {
                fprintf(log_file, "%-10" PRIu64 " %14p %10s %-10" PRIu64 " %.6f %.6f\n",
                        block->id, block->ptr, block->location, block->size,
                        block->alloc_ts, block->free_ts);
                fflush(log_file);
            }
            
            MemNode* to_free = *curr;
            *curr = (*curr)->next;
            real_free(to_free);
            break;
        }
        curr = &((*curr)->next);
    }
    pthread_mutex_unlock(&mtx);

    in_hook = false;
}

static pthread_once_t hook_init_once = PTHREAD_ONCE_INIT;
static bool hook_init_started = false;
static void init_hooks_once() {
    hook_init_started = true;
    real_malloc = dlsym(RTLD_NEXT, "malloc");
    real_free   = dlsym(RTLD_NEXT, "free");
    real_calloc = dlsym(RTLD_NEXT, "calloc");
    real_mmap   = dlsym(RTLD_NEXT, "mmap");
    real_munmap = dlsym(RTLD_NEXT, "munmap");
    if (!real_malloc || !real_free || !real_calloc || !real_mmap || !real_munmap) {
        exit(1);
    }
    start_time = get_timestamp();
    open_log_file();
}


void* malloc(size_t size) {
    if (!hook_init_started) {
        pthread_once(&hook_init_once, init_hooks_once);
    }
    if (!real_malloc) {
        return NULL; 
    }

    void* ptr = real_malloc(size);
    register_alloc(ptr, size, "malloc");
    return ptr;
}

void free(void* ptr) {
    if (!hook_init_started) {
        pthread_once(&hook_init_once, init_hooks_once);
    }
    if (!real_malloc) {
        return ; 
    }
    unregister_alloc(ptr);
    real_free(ptr);
}

void* calloc(size_t nmemb, size_t size) {
    if (!hook_init_started) {
        pthread_once(&hook_init_once, init_hooks_once);
    }
    if (!real_malloc) {
        return NULL; 
    }

    void* ptr = real_calloc(nmemb, size);
    register_alloc(ptr, nmemb * size, "calloc");
    return ptr;
}

void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    if (!hook_init_started) {
        pthread_once(&hook_init_once, init_hooks_once);
    }
    if (!real_malloc) {
        return NULL; 
    }

    void* ptr = real_mmap(addr, length, prot, flags, fd, offset);
    if (fd == -1 && (flags & MAP_ANONYMOUS)) {
        register_alloc(ptr, length, "mmap");
    }
    return ptr;
}

int munmap(void* addr, size_t length) {
    if (!hook_init_started) {
        pthread_once(&hook_init_once, init_hooks_once);
    }
    if (!real_malloc) {
        return -1; 
    }

    unregister_alloc(addr);
    return real_munmap(addr, length);
}