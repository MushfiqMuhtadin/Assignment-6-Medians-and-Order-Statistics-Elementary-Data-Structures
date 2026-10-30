"""Empirical comparison of the elementary data structures (Part 2).

Scenarios (each measures the *total* time for n operations, so O(1)-per-op
structures grow linearly and O(n)-per-op structures grow quadratically):

  1. stack      - n pushes then n pops:     ArrayStack vs LinkedStack
  2. queue      - n enqueues then n dequeues:
                  ArrayQueue (circular) vs LinkedQueue vs NaiveArrayQueue (shifting)
  3. front_insert - n insertions at the front:
                  DynamicArray.insert(0, x)  vs  SinglyLinkedList.insert_head
  4. random_access - n reads at random indices on a size-n structure:
                  DynamicArray[i]  vs  SinglyLinkedList.get(i)
  5. memory     - bytes used to hold n integers (container + nodes, excluding
                  the integer objects themselves)

Usage:  python -m benchmarks.benchmark_data_structures
Outputs results/ds_results.csv, results/ds_memory.csv and results/ds_*.png
"""

import csv
import os
import random
import statistics
import sys
import time

from data_structures import (ArrayQueue, ArrayStack, DynamicArray, LinkedQueue, LinkedStack,
                             NaiveArrayQueue, SinglyLinkedList)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TRIALS = 3


def timed(fn):
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def stack_workload(cls, n):
    def run():
        s = cls()
        for i in range(n):
            s.push(i)
        for _ in range(n):
            s.pop()
    return run


def queue_workload(cls, n):
    def run():
        q = cls()
        for i in range(n):
            q.enqueue(i)
        for _ in range(n):
            q.dequeue()
    return run


def front_insert_array(n):
    def run():
        a = DynamicArray()
        for i in range(n):
            a.insert(0, i)
    return run


def front_insert_list(n):
    def run():
        ll = SinglyLinkedList()
        for i in range(n):
            ll.insert_head(i)
    return run


def random_access(struct, getter, n, idxs):
    def run():
        for i in idxs:
            getter(struct, i)
    return run


def deep_size_array(a):
    return sys.getsizeof(a) + sys.getsizeof(a._data)


def deep_size_linked(ll):
    total = sys.getsizeof(ll)
    node = ll.head
    while node is not None:
        total += sys.getsizeof(node)
        node = node.next
    return total


def main():
    rows = []
    rng = random.Random(7)

    def record(scenario, impl, n, secs):
        rows.append({"scenario": scenario, "implementation": impl, "n": n,
                     "seconds": secs, "us_per_op": 1e6 * secs / n})
        print(f"{scenario:14s} {impl:22s} n={n:>7,d} {secs*1000:10.2f} ms "
              f"({1e6*secs/n:7.3f} us/op)", flush=True)

    big = [1_000, 10_000, 100_000, 300_000]
    for n in big:
        record("stack", "ArrayStack", n, timed(stack_workload(ArrayStack, n)))
        record("stack", "LinkedStack", n, timed(stack_workload(LinkedStack, n)))
        record("queue", "ArrayQueue (circular)", n, timed(queue_workload(ArrayQueue, n)))
        record("queue", "LinkedQueue", n, timed(queue_workload(LinkedQueue, n)))

    # Quadratic contenders: keep n small enough to finish.
    small = [1_000, 2_000, 5_000, 10_000, 20_000]
    for n in small:
        record("queue", "NaiveArrayQueue (shift)", n, timed(queue_workload(NaiveArrayQueue, n)))
        record("front_insert", "DynamicArray", n, timed(front_insert_array(n)))
        record("front_insert", "SinglyLinkedList", n, timed(front_insert_list(n)))
        arr = DynamicArray(range(n))
        ll = SinglyLinkedList(range(n))
        idxs = [rng.randrange(n) for _ in range(n)]
        record("random_access", "DynamicArray", n,
               timed(random_access(arr, lambda s, i: s[i], n, idxs)))
        record("random_access", "SinglyLinkedList", n,
               timed(random_access(ll, lambda s, i: s.get(i), n, idxs)))

    mem_rows = []
    for n in (1_000, 10_000, 100_000):
        arr = DynamicArray(range(n))
        ll = SinglyLinkedList(range(n))
        mem_rows.append({"n": n,
                         "DynamicArray_bytes": deep_size_array(arr),
                         "DynamicArray_capacity": arr.capacity,
                         "SinglyLinkedList_bytes": deep_size_linked(ll),
                         "array_bytes_per_item": round(deep_size_array(arr) / n, 1),
                         "list_bytes_per_item": round(deep_size_linked(ll) / n, 1)})
    print("\nMemory:")
    for r in mem_rows:
        print(r)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "ds_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RESULTS, "ds_memory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mem_rows[0]))
        w.writeheader()
        w.writerows(mem_rows)
    plot(rows)


def plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return
    scenarios = ["stack", "queue", "front_insert", "random_access"]
    titles = {"stack": "Stack: n push + n pop", "queue": "Queue: n enqueue + n dequeue",
              "front_insert": "n inserts at the front", "random_access": "n reads at random index"}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, sc in zip(axes, scenarios):
        impls = sorted({r["implementation"] for r in rows if r["scenario"] == sc})
        for impl in impls:
            pts = sorted((r["n"], r["seconds"]) for r in rows
                         if r["scenario"] == sc and r["implementation"] == impl)
            ax.loglog(*zip(*pts), "-o", ms=4, label=impl)
        ax.set_title(titles[sc])
        ax.set_xlabel("n (number of operations)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("total seconds")
    fig.suptitle("Elementary data structures: total time for n operations (log-log; "
                 "slope 1 = O(1)/op, slope 2 = O(n)/op)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "ds_operations_loglog.png"), dpi=130)
    print("plots written to", RESULTS)


if __name__ == "__main__":
    main()
