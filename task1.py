import os
import platform
import subprocess
import time
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"
import numpy as np
STUDENT_ID = "230103275"
N = 20000000


#
def id_to_seed(student_id):
    digits = "".join(ch for ch in student_id if ch.isdigit())
    if digits:
        return int(digits)
    return sum(student_id.encode("utf-8")) or 1

#CPU model :
def get_cpu_model():
    """Return a readable CPU model string, with fallbacks per OS."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except Exception:
        pass
    # Windows and generic fallback
    return platform.processor() or platform.machine() or "Unknown"


def main():
    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS            : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Platform      : {platform.platform()}")
    print(f"Architecture  : {platform.machine()}")
    print(f"CPU model     : {get_cpu_model()}")
    print(f"Logical cores : {os.cpu_count()}")

     # ---- Part 2: generate data ----
    seed = id_to_seed(STUDENT_ID)
    print("\n=== Serial Computation ===")
    print(f"Seed          : {seed}")
    print(f"Array size    : {N:,} float64 elements")
 
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    data = rng.random(N, dtype=np.float64)
    gen_ms = (time.perf_counter() - t0) * 1000.0
 
    # ---- Part 3 & 4: serial sum + std, timed ----
    start = time.perf_counter()
    total = data.sum()
    std = data.std()  # population standard deviation (ddof=0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
 
    print(f"Sum           : {total:.6f}")
    print(f"Std deviation : {std:.10f}")
    print(f"Generation    : {gen_ms:.3f} ms (not included in compute time)")
    print(f"Compute time  : {elapsed_ms:.3f} ms (serial, wall-clock)")
 

if __name__ == "__main__":
    main()