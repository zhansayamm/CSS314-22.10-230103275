import threading
import time

N_ITEMS = 1000
WORKERS = 4


def cost_modulo(i):
    """Task spec: sleep(0.0001 * (i % 20)) -> irregular, repeating cost."""
    return 0.0001 * (i % 20)


def cost_skewed(i):
    """Bonus: cost grows linearly with i (0 .. ~2 ms) -> strongly unbalanced blocks."""
    return 0.000002 * i


def run_static(cost_fn, workers=WORKERS):
    """Strategy A: equal contiguous blocks, fixed in advance."""
    block = N_ITEMS // workers
    finish, counts = [0.0] * workers, [0] * workers

    def work(w, t0):
        lo, hi = w * block, (w + 1) * block if w < workers - 1 else N_ITEMS
        for i in range(lo, hi):
            time.sleep(cost_fn(i))
        counts[w] = hi - lo
        finish[w] = time.perf_counter() - t0

    return _launch(work, workers, finish, counts)


def run_dynamic(cost_fn, chunk, workers=WORKERS):
    """Strategy B: shared work queue; idle workers grab the next `chunk` items."""
    lock = threading.Lock()
    nxt = [0]
    finish, counts = [0.0] * workers, [0] * workers

    def work(w, t0):
        while True:
            with lock:                      # claim the next chunk
                lo = nxt[0]
                nxt[0] += chunk
            if lo >= N_ITEMS:
                break
            hi = min(lo + chunk, N_ITEMS)
            for i in range(lo, hi):
                time.sleep(cost_fn(i))
            counts[w] += hi - lo
        finish[w] = time.perf_counter() - t0

    return _launch(work, workers, finish, counts)


def _launch(work, workers, finish, counts):
    t0 = time.perf_counter()
    threads = [threading.Thread(target=work, args=(w, t0)) for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = (time.perf_counter() - t0) * 1000.0
    return total, [f * 1000.0 for f in finish], counts


def report(title, cost_fn):
    ideal = sum(cost_fn(i) for i in range(N_ITEMS)) / WORKERS * 1000.0
    print(f"\n=== {title} ===")
    print(f"Ideal balanced time (sum of costs / {WORKERS}): {ideal:.1f} ms")
    header = (f"{'Strategy':<22} | {'Total (ms)':>10} | {'First done':>10} | "
              f"{'Last done':>9} | {'Delta (ms)':>10} | Items per worker")
    print(header)
    print("-" * len(header))
    runs = [
        ("A: Static blocks", run_static(cost_fn)),
        ("B: Dynamic chunk=1", run_dynamic(cost_fn, 1)),
        ("B: Dynamic chunk=5", run_dynamic(cost_fn, 5)),
    ]
    for name, (total, finish, counts) in runs:
        delta = max(finish) - min(finish)
        print(f"{name:<22} | {total:>10.1f} | {min(finish):>10.1f} | "
              f"{max(finish):>9.1f} | {delta:>10.1f} | {counts}")


def main():
    print(f"Items: {N_ITEMS} | Workers: {WORKERS} | time.sleep() releases the GIL, so threads overlap")
    print("Delta = time between the first and the last worker finishing")
    report("Workload 1: sleep(0.0001 * (i % 20))  [task spec]", cost_modulo)
    report("Workload 2 (bonus): sleep(0.000002 * i)  [skewed]", cost_skewed)


if __name__ == "__main__":
    main()