import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230103275"  
N = 20_000_000
P = 0.95         
REPEATS = 5       


def id_to_seed(student_id):
    digits = "".join(ch for ch in student_id if ch.isdigit())
    return int(digits) if digits else (sum(student_id.encode("utf-8")) or 1)


def parallel_sum(data, workers):
    bounds = np.linspace(0, len(data), workers + 1, dtype=np.int64)
    chunks = [data[bounds[i]:bounds[i + 1]] for i in range(workers)]
    if workers == 1:
        return float(np.sum(chunks[0]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        partials = list(pool.map(np.sum, chunks))
    return float(sum(partials))


def timed_run(data, workers):
    times, result = [], None
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = parallel_sum(data, workers)
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times), result


def amdahl(p_workers, parallel_fraction=P):
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / p_workers)


def main():
    max_cores = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, max_cores, max_cores * 2})

    print(f"Student ID : {STUDENT_ID}")
    print(f"Platform   : {platform.platform()}")
    print(f"Max cores  : {max_cores}  (os.cpu_count())")
    print(f"Array size : {N:,} float64  |  Amdahl P = {P}  |  median of {REPEATS} runs")

    seed = id_to_seed(STUDENT_ID)
    data = np.random.default_rng(seed).random(N, dtype=np.float64)
    reference = float(np.sum(data))
    print(f"Seed       : {seed}")
    print(f"Serial sum : {reference:.6f}\n")

    # warm-up so first-touch page faults don't distort T(1)
    parallel_sum(data, 1)

    rows, t1 = [], None
    for w in worker_counts:
        t_ms, result = timed_run(data, w)
        if w == 1:
            t1 = t_ms
        ok = abs(result - reference) <= 1e-6 * abs(reference)
        rows.append((w, t_ms, t1 / t_ms, amdahl(w), ok))

    header = f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup S(p)':>12} | {'Amdahl Upper Bound':>18}"
    print(header)
    print("-" * len(header))
    for w, t_ms, s, a, ok in rows:
        flag = "" if ok else "  <-- SUM MISMATCH"
        print(f"{w:>6} | {t_ms:>14.3f} | {s:>11.2f}x | {a:>17.2f}x{flag}")

    print(f"\nAmdahl limit as p -> infinity: {1.0 / (1.0 - P):.1f}x")


if __name__ == "__main__":
    main()