import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

A, B = 2.5, 1.0
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
WORKERS = os.cpu_count()
REFINE_POINTS = 14


def reps_for(n):
    """More repetitions for tiny arrays (timings are microseconds, noisy)."""
    if n <= 1_000:
        return 3000
    if n <= 10_000:
        return 1500
    if n <= 100_000:
        return 400
    if n <= 1_000_000:
        return 60
    return 12


def saxpy(x, y):
    """y = a*x + b computed in place (no temporary arrays)."""
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def run_serial(x, y):
    saxpy(x, y)


def run_parallel(pool, x, y, workers):
    """Split into `workers` slices; each thread processes its own slice.
    NumPy ufuncs release the GIL, so the slices run truly in parallel."""
    n = len(x)
    bounds = np.linspace(0, n, workers + 1, dtype=np.int64)
    futures = [pool.submit(saxpy, x[bounds[i]:bounds[i + 1]], y[bounds[i]:bounds[i + 1]])
               for i in range(workers)]
    for f in futures:
        f.result()


def median_time_us(fn, reps):
    for _ in range(max(3, reps // 20)):          # warm-up
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


def measure(pool, n):
    rng = np.random.default_rng(12345)
    x = rng.random(n)
    y_ser, y_par = np.empty(n), np.empty(n)
    w = min(WORKERS, n)
    reps = reps_for(n)
    t_ser = median_time_us(lambda: run_serial(x, y_ser), reps)
    t_par = median_time_us(lambda: run_parallel(pool, x, y_par, w), reps)
    assert np.array_equal(y_ser, y_par), f"result mismatch at N={n}"
    return t_ser, t_par


def print_table(rows):
    header = f"{'N':>12} | {'Serial (us)':>12} | {'Parallel (us)':>13} | {'Speedup':>8} | Winner"
    print(header)
    print("-" * len(header))
    for n, ts, tp in rows:
        s = ts / tp
        print(f"{n:>12,} | {ts:>12.1f} | {tp:>13.1f} | {s:>7.2f}x | {'PARALLEL' if s > 1 else 'serial'}")


def crossover(rows):
    """First N from which parallel stays faster for every larger measured N.
    Returns (last_slower_N, first_faster_N); None entries mean 'no bound'."""
    faster = [ts / tp > 1.0 for _, ts, tp in rows]
    if not any(faster):
        return ("never", None)
    k = len(rows)
    while k > 0 and faster[k - 1]:
        k -= 1                                   # k = index of first N in the trailing "all faster" run
    if k == len(rows):
        return ("never", None)                   # largest N is not faster
    lo = rows[k - 1][0] if k > 0 else None
    return (lo, rows[k][0])


def main():
    print(f"Platform : {platform.platform()}")
    print(f"Workers  : {WORKERS} threads (os.cpu_count())")
    print(f"Operation: y[i] = {A} * x[i] + {B}   (float64, in-place NumPy ufuncs)")
    print("Parallel timing includes task dispatch + join (thread pool is created once, outside timing).\n")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        run_parallel(pool, np.ones(WORKERS * 4), np.empty(WORKERS * 4), WORKERS)  # spin up threads

        print("--- Coarse sweep ---")
        rows = [(n, *measure(pool, n)) for n in SIZES]
        print_table(rows)

        lo, hi = crossover(rows)
        if lo not in (None, "never") and hi is not None:
            fine_sizes = sorted(set(int(v) for v in np.geomspace(lo, hi, REFINE_POINTS)) - {lo, hi})
            print(f"\n--- Refinement between N={lo:,} and N={hi:,} ---")
            fine = [(n, *measure(pool, n)) for n in fine_sizes]
            print_table(fine)
            rows = sorted(rows + fine)
            lo, hi = crossover(rows)

    print("\n=== RESULT ===")
    if lo == "never":
        print(f"Multi-threading was NOT faster than serial at any tested N on this machine "
              f"({WORKERS} worker(s)).")
    elif lo is None:
        print(f"Multi-threading was already faster at the smallest tested N ({hi:,}); "
              "the tipping point lies below it.")
    else:
        print(f"Multi-threading shifts from SLOWER to FASTER between N = {lo:,} and N = {hi:,}.")
        print(f"  Last N where serial wins          : {lo:,}")
        print(f"  First N from which parallel wins  : {hi:,}")


if __name__ == "__main__":
    main()