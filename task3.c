/*
 * Task 3: False sharing demonstration.
 *
 * Build:  gcc -O2 -pthread task3.c -o task3          (Linux / macOS)
 * Run:    ./task3 [num_threads]                      (default = physical cores)
 *
 * Three variants, N threads each doing ITERS increments on "their" counter:
 *   1. Unpadded shared : counts[i] are adjacent 8-byte slots -> same 64-byte cache line
 *   2. Padded          : counts[i] are 64 bytes apart        -> one cache line per thread
 *   3. Local variable  : thread keeps a private register, writes counts[i] once at the end
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define ITERS 50000000L
#define CACHE_LINE 64
#define MAX_THREADS 256

typedef struct { int id; } arg_t;

/* Variant 1: contiguous array (all threads hit the same cache line(s)). */
static volatile long shared_counts[MAX_THREADS] __attribute__((aligned(CACHE_LINE)));

/* Variant 2: each counter owns a full 64-byte cache line. */
typedef struct { volatile long v; char pad[CACHE_LINE - sizeof(long)]; } padded_t;
static padded_t padded_counts[MAX_THREADS] __attribute__((aligned(CACHE_LINE)));

/* Variant 3 result slots (written once per thread at the end). */
static long local_result[MAX_THREADS];

static void *worker_unpadded(void *p) {
    int i = ((arg_t *)p)->id;
    for (long k = 0; k < ITERS; k++) shared_counts[i]++;
    return NULL;
}

static void *worker_padded(void *p) {
    int i = ((arg_t *)p)->id;
    for (long k = 0; k < ITERS; k++) padded_counts[i].v++;
    return NULL;
}

static void *worker_local(void *p) {
    int i = ((arg_t *)p)->id;
    long local = 0;
    for (long k = 0; k < ITERS; k++) {
        local++;
        __asm__ volatile("" : "+r"(local));   /* stop compiler folding the loop */
    }
    local_result[i] = local;                  /* single write at the end */
    return NULL;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double run(void *(*fn)(void *), int n) {
    pthread_t th[MAX_THREADS];
    arg_t args[MAX_THREADS];
    double t0 = now_ms();
    for (int i = 0; i < n; i++) { args[i].id = i; pthread_create(&th[i], NULL, fn, &args[i]); }
    for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
    return now_ms() - t0;
}

static int physical_cores(void) {
    int logical = (int)sysconf(_SC_NPROCESSORS_ONLN);
#ifdef __linux__
    /* count unique (physical id, core id) pairs in /proc/cpuinfo */
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256]; int phys = 0, core = 0, have_core = 0, n = 0;
        int seen[1024][2];
        while (fgets(line, sizeof line, f)) {
            if (sscanf(line, "physical id : %d", &phys) == 1) continue;
            if (sscanf(line, "core id : %d", &core) == 1) have_core = 1;
            if (line[0] == '\n' && have_core) {
                int dup = 0;
                for (int j = 0; j < n; j++) if (seen[j][0] == phys && seen[j][1] == core) dup = 1;
                if (!dup && n < 1024) { seen[n][0] = phys; seen[n][1] = core; n++; }
                have_core = 0;
            }
        }
        fclose(f);
        if (n > 0) return n;
    }
#elif defined(__APPLE__)
    FILE *p = popen("sysctl -n hw.physicalcpu", "r");
    if (p) { int n = 0; if (fscanf(p, "%d", &n) == 1 && n > 0) { pclose(p); return n; } pclose(p); }
#endif
    return logical;   /* fallback */
}

int main(int argc, char **argv) {
    int logical = (int)sysconf(_SC_NPROCESSORS_ONLN);
    int n = physical_cores();
    if (argc > 1) n = atoi(argv[1]);
    if (n < 1) n = 1;
    if (n > MAX_THREADS) n = MAX_THREADS;

    printf("Logical cores  : %d\n", logical);
    printf("Threads (N)    : %d (physical cores unless overridden)\n", n);
    printf("Increments/thr : %ld\n", ITERS);
    printf("Cache line     : %d bytes\n\n", CACHE_LINE);

    memset((void *)shared_counts, 0, sizeof shared_counts);
    memset((void *)padded_counts, 0, sizeof padded_counts);

    double t_unpadded = run(worker_unpadded, n);
    double t_padded   = run(worker_padded, n);
    double t_local    = run(worker_local, n);

    /* correctness check: every counter must equal ITERS */
    int ok = 1;
    for (int i = 0; i < n; i++)
        if (shared_counts[i] != ITERS || padded_counts[i].v != ITERS || local_result[i] != ITERS) ok = 0;

    printf("%-24s | %12s | %s\n", "Variant", "Wall Time", "Slowdown vs Unpadded");
    printf("-------------------------+--------------+---------------------\n");
    printf("%-24s | %9.1f ms | %s\n", "Unpadded shared", t_unpadded, "1.00x (baseline)");
    printf("%-24s | %9.1f ms | %.2fx faster\n", "Padded (64-byte)", t_padded, t_unpadded / t_padded);
    printf("%-24s | %9.1f ms | %.2fx faster\n", "Local variable", t_local, t_unpadded / t_local);

    printf("\nExecution ratio (Unpadded / Padded) : %.2f\n", t_unpadded / t_padded);
    printf("Execution ratio (Unpadded / Local)  : %.2f\n", t_unpadded / t_local);
    printf("Correctness check                   : %s\n", ok ? "PASS" : "FAIL");
    return 0;
}