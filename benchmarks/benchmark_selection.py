"""Empirical comparison of the selection algorithms (Part 1, step 3).

For each input size n and distribution we time selecting the median
(k = (n+1)//2) and report the median of several trials, so one slow outlier
(GC pause, OS hiccup) does not distort the result.

Distributions:
    random      - uniform random integers, essentially distinct
    sorted      - 0, 1, ..., n-1
    reverse     - n-1, ..., 0
    few_unique  - random integers from only 10 distinct values (heavy duplicates)
    all_equal   - n copies of the same key

Algorithms:
    deterministic  - median of medians (worst-case O(n))
    randomized     - randomized quickselect (expected O(n))
    naive          - last-element-pivot quickselect (O(n^2) on sorted input),
                     included as a baseline; capped at small n on bad inputs
    sort           - Python's built-in sorted()  (Timsort, O(n log n) in C)

Usage:
    python -m benchmarks.benchmark_selection            # full run (a few minutes)
    python -m benchmarks.benchmark_selection --quick    # smaller sizes
Outputs results/selection_results.csv and results/selection_*.png
"""

import argparse
import csv
import os
import random
import statistics
import sys
import time

sys.setrecursionlimit(10000)

from selection import deterministic_select, naive_quickselect, randomized_select

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def make_input(dist: str, n: int, rng: random.Random):
    if dist == "random":
        return [rng.randint(0, 10 * n) for _ in range(n)]
    if dist == "sorted":
        return list(range(n))
    if dist == "reverse":
        return list(range(n - 1, -1, -1))
    if dist == "few_unique":
        return [rng.randint(0, 9) for _ in range(n)]
    if dist == "all_equal":
        return [42] * n
    raise ValueError(dist)


def time_call(fn, arr, k, trials):
    times = []
    result = None
    for _ in range(trials):
        t0 = time.perf_counter()
        result = fn(arr, k)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000]
    if args.quick:
        sizes = [1_000, 5_000, 10_000, 50_000]
    dists = ["random", "sorted", "reverse", "few_unique", "all_equal"]
    naive_cap_bad = 5_000  # naive quickselect is quadratic on these; keep it short

    rng = random.Random(2026)
    algos = {
        "deterministic": deterministic_select,
        "randomized": lambda a, k: randomized_select(a, k, rng=rng),
        "naive": naive_quickselect,
        "sort": lambda a, k: sorted(a)[k - 1],
    }

    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for dist in dists:
        for n in sizes:
            arr = make_input(dist, n, rng)
            k = (n + 1) // 2
            truth = sorted(arr)[k - 1]
            for name, fn in algos.items():
                if name == "naive" and dist != "random" and n > naive_cap_bad:
                    continue
                trials = args.trials if n <= 200_000 else max(3, args.trials // 2)
                t, res = time_call(fn, arr, k, trials)
                assert res == truth, (name, dist, n)
                rows.append({"distribution": dist, "n": n, "algorithm": name,
                             "seconds": t, "us_per_element": 1e6 * t / n})
                print(f"{dist:11s} n={n:>7,d} {name:13s} {t*1000:10.2f} ms  "
                      f"({1e6*t/n:6.3f} us/elem)", flush=True)

    # Machine-independent work counts: shows *why* the timings differ.
    # elements_scanned / n should stay roughly constant as n grows (linear time).
    count_rows = []
    for dist in dists:
        for n in (10_000, 100_000, 500_000):
            arr = make_input(dist, n, rng)
            k = (n + 1) // 2
            d = {}
            deterministic_select(arr, k, stats=d)
            rp, rs = [], []
            for _ in range(20):
                r = {}
                randomized_select(arr, k, rng=rng, stats=r)
                rp.append(r["partitions"])
                rs.append(r["elements_scanned"] / n)
            count_rows.append({
                "distribution": dist, "n": n,
                "det_partitions": d["partitions"],
                "det_scanned_per_n": round(d["elements_scanned"] / n, 3),
                "rand_partitions_mean": round(statistics.mean(rp), 2),
                "rand_scanned_per_n_mean": round(statistics.mean(rs), 3),
                "rand_scanned_per_n_max": round(max(rs), 3),
            })

    with open(os.path.join(RESULTS, "selection_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RESULTS, "selection_partition_counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(count_rows[0]))
        w.writeheader()
        w.writerows(count_rows)
    print("\nWork counts (20 randomized runs each):")
    for r in count_rows:
        print(r)

    plot(rows, dists)


def plot(rows, dists):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    colors = {"deterministic": "#2a6fdb", "randomized": "#e07a1f",
              "naive": "#c0392b", "sort": "#7f8c8d"}
    styles = {"deterministic": "-o", "randomized": "-s", "naive": "--^", "sort": ":d"}

    fig, axes = plt.subplots(1, len(dists), figsize=(4.2 * len(dists), 4), sharey=True)
    for ax, dist in zip(axes, dists):
        for algo in colors:
            pts = sorted((r["n"], r["seconds"]) for r in rows
                         if r["distribution"] == dist and r["algorithm"] == algo)
            if pts:
                ax.loglog(*zip(*pts), styles[algo], color=colors[algo], label=algo, ms=4)
        ax.set_title(dist)
        ax.set_xlabel("n")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("seconds (median of trials)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Median selection time vs n (log-log)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "selection_time_loglog.png"), dpi=130)

    # Time per element: flat line == linear time.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, algo in zip(axes, ["deterministic", "randomized"]):
        for dist in dists:
            pts = sorted((r["n"], r["us_per_element"]) for r in rows
                         if r["distribution"] == dist and r["algorithm"] == algo)
            ax.semilogx(*zip(*pts), "-o", ms=4, label=dist)
        ax.set_title(f"{algo}: microseconds per element")
        ax.set_xlabel("n")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("us / element  (flat = linear time)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "selection_time_per_element.png"), dpi=130)
    print("plots written to", RESULTS)


if __name__ == "__main__":
    main()
