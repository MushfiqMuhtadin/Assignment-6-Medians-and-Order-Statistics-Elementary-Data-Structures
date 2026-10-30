"""Randomized selection in expected linear time (Randomized Quickselect).

Algorithm (Hoare, 1961; CLRS section 9.2):

    RANDOMIZED-SELECT(A, k):
      1. Pick a pivot x uniformly at random from A.
      2. Partition A around x into  L (< x),  E (== x),  G (> x).
      3. If k <= |L|           -> recurse into L
         elif k <= |L| + |E|   -> return x
         else                  -> recurse into G with k - |L| - |E|

Expected cost: call a pivot "good" if it lands in the middle half of the
window (rank between n/4 and 3n/4). That happens with probability 1/2, so on
average 2 partitions are needed before a good pivot, and a good pivot shrinks
the window to at most 3n/4. Summing over phases:

    E[T(n)] <= sum_j  2 * c * (3/4)^j * n  =  8cn  =  O(n).

Worst case is still Theta(n^2) (every pivot is the min or max), but that
has probability about 2^-n for distinct keys and does not depend on the input
order. Sorted or reverse-sorted input is *not* a bad case, unlike a
deterministic "always pick the last element" quickselect.

Implementation notes
--------------------
* In place on a copy; step 3 is a loop, so extra space is O(1) besides the copy.
* Three-way partitioning makes duplicate-heavy input linear as well (with a
  two-way partition, n equal keys would cost Theta(n^2)).
* ``rng`` can be injected for reproducible experiments.
"""

import random
from typing import Optional, Sequence, TypeVar

from ._common import partition3, validate

T = TypeVar("T")


def randomized_select(
    arr: Sequence[T],
    k: int,
    rng: Optional[random.Random] = None,
    stats: Optional[dict] = None,
) -> T:
    """Return the k-th smallest element of ``arr`` (k is 1-based).

    Expected O(n) time for every input; worst case O(n^2) with vanishing
    probability. O(n) extra space for the working copy, O(1) otherwise.

    Args:
        arr:   any sequence of mutually comparable items (duplicates allowed).
        k:     rank to select, 1 <= k <= len(arr).
        rng:   optional ``random.Random`` instance (for reproducibility).
        stats: optional dict; ``stats["partitions"]`` counts partition passes and
               ``stats["elements_scanned"]`` sums their window sizes.

    >>> randomized_select([7, 2, 9, 4, 4, 1], 3, rng=random.Random(0))
    4
    """
    validate(arr, k)
    rand = (rng or random).randint
    a = list(arr)
    lo, hi, idx = 0, len(a) - 1, k - 1
    while True:
        if lo == hi:
            return a[lo]
        pivot = a[rand(lo, hi)]
        lt, gt = partition3(a, lo, hi, pivot)
        if stats is not None:
            stats["partitions"] = stats.get("partitions", 0) + 1
            stats["elements_scanned"] = stats.get("elements_scanned", 0) + hi - lo + 1
        if idx < lt:
            hi = lt - 1
        elif idx > gt:
            lo = gt + 1
        else:
            return pivot


def randomized_median(arr: Sequence[T], rng: Optional[random.Random] = None) -> T:
    """Lower median of ``arr`` in expected linear time."""
    return randomized_select(arr, (len(arr) + 1) // 2, rng=rng)


def naive_quickselect(arr: Sequence[T], k: int) -> T:
    """Baseline for the experiments only: quickselect that always uses the
    *last* element as pivot (Lomuto-style, two-way partition).

    Linear on random data, but Theta(n^2) on sorted / reverse-sorted input.
    It shows why the pivot rule matters; don't use it in real code.
    """
    validate(arr, k)
    a = list(arr)
    lo, hi, idx = 0, len(a) - 1, k - 1
    while lo < hi:
        pivot = a[hi]
        i = lo
        for j in range(lo, hi):
            if a[j] <= pivot:
                a[i], a[j] = a[j], a[i]
                i += 1
        a[i], a[hi] = a[hi], a[i]
        if idx == i:
            return a[i]
        if idx < i:
            hi = i - 1
        else:
            lo = i + 1
    return a[lo]
