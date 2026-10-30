"""Deterministic selection in worst-case linear time (Median of Medians).

Algorithm (Blum, Floyd, Pratt, Rivest & Tarjan, 1973; CLRS section 9.3):

    SELECT(A, k):
      1. If |A| <= 5, sort it and return the k-th element.
      2. Split A into ceil(n/5) groups of 5 (the last may be smaller).
      3. Sort each group (O(1) per group) and take its median.
      4. Recursively SELECT the median x of those ceil(n/5) medians.
      5. Partition A around x into  L (< x),  E (== x),  G (> x).
      6. If k <= |L|           -> recurse into L
         elif k <= |L| + |E|   -> return x
         else                  -> recurse into G with k - |L| - |E|

Why the pivot is good: at least half of the ceil(n/5) groups have a median
<= x, and each such group (except possibly x's own group and the short last
group) contributes 3 elements <= x. So at least 3n/10 - 6 elements are <= x
and, symmetrically, >= x. Step 6 therefore recurses on at most 7n/10 + 6
elements, giving

    T(n) <= T(ceil(n/5)) + T(7n/10 + 6) + O(n)  =>  T(n) = O(n),

because 1/5 + 7/10 = 9/10 < 1 (the work shrinks geometrically).

Implementation notes
--------------------
* Works in place on a copy of the input, so the caller's list is untouched.
* Step 6 is a loop rather than a recursive call (tail-call elimination), so
  the only real recursion is step 4, whose depth is O(log_5 n).
* Medians are swapped to the front of the current window, so step 4 recurses
  on a contiguous sub-window and needs no extra list.
* Step 5 is a three-way partition, which keeps the algorithm linear even on
  inputs with many duplicates (e.g. all keys equal).
"""

from typing import List, Optional, Sequence, TypeVar

from ._common import insertion_sort, partition3, validate

T = TypeVar("T")

GROUP_SIZE = 5


def deterministic_select(arr: Sequence[T], k: int, stats: Optional[dict] = None) -> T:
    """Return the k-th smallest element of ``arr`` (k is 1-based).

    Worst-case O(n) time, O(n) extra space for the working copy plus
    O(log n) recursion stack.

    Args:
        arr:   any sequence of mutually comparable items (duplicates allowed).
        k:     rank to select, 1 <= k <= len(arr). k=1 is the minimum,
               k=len(arr) the maximum, k=(len(arr)+1)//2 the lower median.
        stats: optional dict; if given, ``stats["partitions"]`` counts partition
               passes (including those inside the recursive median-of-medians
               calls) and ``stats["elements_scanned"]`` sums their window sizes,
               a machine-independent measure of the work done.

    Raises:
        ValueError: on an empty sequence or out-of-range k.
        TypeError:  if k is not an int.

    >>> deterministic_select([7, 2, 9, 4, 4, 1], 3)
    4
    """
    validate(arr, k)
    a = list(arr)
    return _select(a, 0, len(a) - 1, k - 1, stats)


def _select(a: List[T], lo: int, hi: int, idx: int, stats: Optional[dict]) -> T:
    """Return the element that would sit at absolute index ``idx`` if
    ``a[lo..hi]`` were sorted. Requires lo <= idx <= hi."""
    while True:
        n = hi - lo + 1
        if n <= GROUP_SIZE:
            insertion_sort(a, lo, hi)
            return a[idx]

        # Steps 2-3: sort each group of 5 and move its median to a[lo + g].
        m = lo  # next free slot in the "medians" prefix
        for g_lo in range(lo, hi + 1, GROUP_SIZE):
            g_hi = min(g_lo + GROUP_SIZE - 1, hi)
            insertion_sort(a, g_lo, g_hi)
            mid = (g_lo + g_hi) // 2
            a[m], a[mid] = a[mid], a[m]
            m += 1

        # Step 4: median of the medians, which now occupy a[lo .. m-1].
        pivot = _select(a, lo, m - 1, lo + (m - 1 - lo) // 2, stats)

        # Step 5: three-way partition of the whole window around the pivot.
        lt, gt = partition3(a, lo, hi, pivot)
        if stats is not None:
            stats["partitions"] = stats.get("partitions", 0) + 1
            stats["elements_scanned"] = stats.get("elements_scanned", 0) + n

        # Step 6: keep only the side that contains idx.
        if idx < lt:
            hi = lt - 1
        elif idx > gt:
            lo = gt + 1
        else:
            return pivot


def deterministic_median(arr: Sequence[T]) -> T:
    """Lower median of ``arr`` in worst-case linear time."""
    return deterministic_select(arr, (len(arr) + 1) // 2)
