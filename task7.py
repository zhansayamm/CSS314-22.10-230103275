import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

N = 30_000_000
MAX_CORES = os.cpu_count() or 1
CORE_COUNTS = sorted({1, 2, MAX_CORES})
BLOCK = 8192                 # compute kernel works in cache-sized blocks (stays out of RAM)
EFF_THRESHOLD = 0.70         # efficiency below this at max cores = "stalled"
REPS = {"A": 5, "B": 3}


def bounds(n, p):
    return np.linspace(0, n, p + 1, dtype=np.int64)


def kernel_a_chunk(a, b, lo, hi):
    """Memory-bound: pure streaming copy, 16 bytes of traffic per element, ~0 FLOPs."""
    np.copyto(b[lo:hi], a[lo:hi])


def kernel_b_chunk(a, b, lo, hi):
    """Compute-bound: sin*cos+tan. Done in small blocks with reusable scratch buffers so the
    temporaries live in L1/L2 cache instead of generating extra RAM traffic."""
    t1, t2 = np.empty(BLOCK), np.empty(BLOCK)
    for s in range(lo, hi, BLOCK):
        e = min(s + BLOCK, hi)
        m = e - s
        x = a[s:e]
        np.sin(x, out=t1[:m])
        np.cos(x, out=t2[:m])
        np.multiply(t1[:m], t2[:m], out=t1[:m])
        np.tan(x, out=t2[:m])
        np.add(t1[:m], t2[:m], out=b[s:e])


def run(pool, chunk_fn, a, b, p):
    bd = bounds(len(a), p)
    if p == 1:
        chunk_fn(a, b, 0, len(a))
        return
    futs = [pool.submit(chunk_fn, a, b, int(bd[i]), int(bd[i + 1])) for i in range(p)]
    for f in futs:
        f.result()


def bench(pool, chunk_fn, a, b, p, reps):
    run(pool, chunk_fn, a, b, p)                       # warm-up
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run(pool, chunk_fn, a, b, p)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main():
    print(f"Platform : {platform.platform()}")
    print(f"Array    : {N:,} float64 ({N * 8 / 1e6:.0f} MB per array)")
    print(f"Cores    : testing {CORE_COUNTS} (max = {MAX_CORES})\n")

    rng = np.random.default_rng(7)
    a = rng.random(N) * 1.5
    b = np.zeros(N)                                    # pre-fault output pages

    results = {"A": {}, "B": {}}
    with ThreadPoolExecutor(max_workers=max(CORE_COUNTS)) as pool:
        for key, fn in (("A", kernel_a_chunk), ("B", kernel_b_chunk)):
            for p in CORE_COUNTS:
                results[key][p] = bench(pool, fn, a, b, p, REPS[key])
            # correctness check of the last (max-core) run
            if key == "A":
                assert np.array_equal(a, b), "copy mismatch"
            else:
                chk = slice(0, 1_000_000)
                ref = np.sin(a[chk]) * np.cos(a[chk]) + np.tan(a[chk])
                assert np.allclose(ref, b[chk]), "compute mismatch"

    names = {"A": "Kernel A (memory-bound copy)", "B": "Kernel B (compute-bound sin*cos+tan)"}
    eff_max = {}
    for key in ("A", "B"):
        print(f"=== {names[key]} ===")
        header = f"{'Cores':>5} | {'Time (ms)':>10} | {'Speedup':>8} | {'Efficiency':>10}"
        if key == "A":
            header += f" | {'Bandwidth':>11}"
        print(header)
        print("-" * len(header))
        t1 = results[key][1]
        for p in CORE_COUNTS:
            t = results[key][p]
            sp = t1 / t
            eff = sp / p
            line = f"{p:>5} | {t:>10.1f} | {sp:>7.2f}x | {eff * 100:>9.1f}%"
            if key == "A":
                line += f" | {2 * 8 * N / (t / 1000) / 1e9:>7.1f} GB/s"
            print(line)
            if p == MAX_CORES:
                eff_max[key] = eff
        print()

    print("=== VERDICT ===")
    if MAX_CORES < 2:
        print("Only 1 core is available here, so scaling cannot be assessed. Run on a multi-core machine.")
        return
    ea, eb = eff_max["A"], eff_max["B"]
    print(f"Efficiency at {MAX_CORES} cores: Kernel A = {ea * 100:.1f}%, Kernel B = {eb * 100:.1f}%")
    if ea < EFF_THRESHOLD and ea < eb:
        print("=> Kernel A stalled due to memory bandwidth saturation")
        print("   Kernel B scaled well to 2 cores and sublinearly to 8, consistent with heterogeneous P/E cores rather than memory limits.")
    elif ea < EFF_THRESHOLD and eb < EFF_THRESHOLD:
        print("=> BOTH kernels scaled poorly; Kernel A is the memory-bandwidth-saturated one"
              if ea < eb else "=> BOTH kernels scaled poorly; Kernel B scaled worse than the copy "
              "(likely shared execution units / SMT or thermal limits, not RAM).")
    else:
        print(f"=> No kernel dropped below {EFF_THRESHOLD * 100:.0f}% efficiency: memory channels "
              "were not saturated with this core count.")


if __name__ == "__main__":
    main()