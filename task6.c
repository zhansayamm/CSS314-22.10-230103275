/*
 * Task 6: Race condition vs. synchronization.
 *
 * Build:  gcc -O2 -pthread task6.c -o task6
 * Run:    ./task6
 *
 * Three variants, 4 threads x 2,000,000 increments each:
 *   1. Unsynchronized  : plain counter++ (read-modify-write, updates get lost)
 *   2. Atomic          : hardware atomic fetch-and-add (lock-prefixed instruction on x86)
 *   3. Mutex           : pthread_mutex around each increment
 */
#include <pthread.h>
#include <stdio.h>
#include <time.h>

#define THREADS 4
#define INCREMENTS 2000000L
#define EXPECTED ((long)THREADS * INCREMENTS)

static volatile long counter_race = 0;   /* volatile: force a real load/add/store every iteration */
static long counter_atomic = 0;
static long counter_mutex = 0;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static void *worker_race(void *arg) {
    (void)arg;
    for (long i = 0; i < INCREMENTS; i++) counter_race++;       /* NOT atomic */
    return NULL;
}

static void *worker_atomic(void *arg) {
    (void)arg;
    for (long i = 0; i < INCREMENTS; i++)
        __atomic_fetch_add(&counter_atomic, 1, __ATOMIC_SEQ_CST);
    return NULL;
}

static void *worker_mutex(void *arg) {
    (void)arg;
    for (long i = 0; i < INCREMENTS; i++) {
        pthread_mutex_lock(&lock);
        counter_mutex++;
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double run(void *(*fn)(void *)) {
    pthread_t th[THREADS];
    double t0 = now_ms();
    for (int i = 0; i < THREADS; i++) pthread_create(&th[i], NULL, fn, NULL);
    for (int i = 0; i < THREADS; i++) pthread_join(th[i], NULL);
    return now_ms() - t0;
}

int main(void) {
    printf("Threads            : %d\n", THREADS);
    printf("Increments/thread  : %ld\n", INCREMENTS);
    printf("Expected final     : %ld\n\n", EXPECTED);

    double t_race   = run(worker_race);
    double t_atomic = run(worker_atomic);
    double t_mutex  = run(worker_mutex);

    long lost = EXPECTED - counter_race;
    printf("--- 1. Unsynchronized (race condition) ---\n");
    printf("Expected final value : %ld\n", EXPECTED);
    printf("Actual final value   : %ld\n", (long)counter_race);
    printf("Lost updates         : %ld (%.2f%%)\n", lost, 100.0 * lost / EXPECTED);
    printf("Wall time            : %.1f ms\n\n", t_race);

    printf("--- 2. Atomic fetch-and-add ---\n");
    printf("Final value          : %ld  [%s]\n", counter_atomic, counter_atomic == EXPECTED ? "CORRECT" : "WRONG");
    printf("Wall time            : %.1f ms\n", t_atomic);
    printf("Penalty vs. race     : +%.1f ms (%.2fx slower)\n\n", t_atomic - t_race, t_atomic / t_race);

    printf("--- 3. Mutex lock ---\n");
    printf("Final value          : %ld  [%s]\n", counter_mutex, counter_mutex == EXPECTED ? "CORRECT" : "WRONG");
    printf("Wall time            : %.1f ms\n", t_mutex);
    printf("Penalty vs. race     : +%.1f ms (%.2fx slower)\n", t_mutex - t_race, t_mutex / t_race);
    return 0;
}