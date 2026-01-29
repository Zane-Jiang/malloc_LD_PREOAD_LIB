#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

#define ENV_RESERVE_PAGES "RESERVE_PAGES"
#define DEFAULT_RESERVE_PAGES 0

static void *reserved_memory = NULL;
static size_t reserved_size = 0;

__attribute__((constructor))
static void reserve_space_init(void) {
    const char *env_pages = getenv(ENV_RESERVE_PAGES);
    long num_pages = DEFAULT_RESERVE_PAGES;
    
    if (env_pages != NULL) {
        char *endptr;
        num_pages = strtol(env_pages, &endptr, 10);
        if (*endptr != '\0' || num_pages < 0) {
            fprintf(stderr, "[reserve_space] Invalid %s value: %s\n", 
                    ENV_RESERVE_PAGES, env_pages);
            return;
        }
    }
    
    if (num_pages == 0) {
        return;
    }
    
    // long page_size = sysconf(_SC_PAGESIZE);
    long page_size = 4 * 1024;
    if (page_size == -1) {
        perror("[reserve_space] Failed to get page size");
        return;
    }
    
    reserved_size = (size_t)num_pages * (size_t)page_size;
    
    // Use MAP_POPULATE to pre-fault pages during mmap (avoids later page faults)
    reserved_memory = mmap(NULL, reserved_size, 
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
                           -1, 0);
    
    if (reserved_memory == MAP_FAILED) {
        fprintf(stderr, "[reserve_space] Failed to reserve %zu bytes (%ld pages): %s\n",
                reserved_size, num_pages, strerror(errno));
        reserved_memory = NULL;
        reserved_size = 0;
        return;
    }
    
    // // Use mlock to ensure physical memory is allocated without accessing it
    // if (mlock(reserved_memory, reserved_size) == -1) {
    //     fprintf(stderr, "[reserve_space] mlock failed: %s (continuing anyway)\n",
    //             strerror(errno));
    //     // Don't return - MAP_POPULATE should have already allocated the pages
    // }
    
    fprintf(stderr, "[reserve_space] Reserved %zu bytes (%ld pages) at %p\n",
            reserved_size, num_pages, reserved_memory);
}

__attribute__((destructor))
static void reserve_space_fini(void) {
    if (reserved_memory != NULL && reserved_size > 0) {
        // munlock before munmap (optional, munmap will do it automatically)
        // munlock(reserved_memory, reserved_size);
        
        if (munmap(reserved_memory, reserved_size) == -1) {
            perror("[reserve_space] Failed to unmap reserved memory");
        } else {
            fprintf(stderr, "[reserve_space] Released %zu bytes at %p\n",
                    reserved_size, reserved_memory);
        }
        reserved_memory = NULL;
        reserved_size = 0;
    }
}
