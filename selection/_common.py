"""Shared helpers for the selection algorithms.

Both selection algorithms work *in place* on a private copy of the input,
restricted to an index window ``a[lo..hi]`` (inclusive). The helpers here
operate on such windows.
"""

from typing import List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def validate(arr: Sequence, k: int) -> None:
    """Raise a helpful error if ``k`` is not a valid 1-based rank for ``arr``."""
    if len(arr) == 0:
        raise ValueError("cannot select from an empty sequence")
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if not 1 <= k <= len(arr):
        raise ValueError(f"k must be in [1, {len(arr)}], got {k}")


def insertion_sort(a: List[T], lo: int, hi: int) -> None:
    """Sort ``a[lo..hi]`` in place. Used only on tiny windows (<= 5 items),
    where it costs at most 10 comparisons, i.e. O(1)."""
    for i in range(lo + 1, hi + 1):
        key = a[i]
        j = i - 1
        while j >= lo and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key


def partition3(a: List[T], lo: int, hi: int, pivot: T) -> Tuple[int, int]:
    """Three-way (Dutch national flag) partition of ``a[lo..hi]`` around ``pivot``.

    After the call, with returned ``(lt, gt)``:

        a[lo   .. lt-1] <  pivot
        a[lt   .. gt  ] == pivot
        a[gt+1 .. hi  ] >  pivot

    Grouping *all* copies of the pivot together is what makes the selection
    algorithms robust to duplicates: an array of n equal keys is finished
    after one O(n) pass instead of degrading to O(n^2).
    One pass, n comparisons-ish, O(1) extra space.
    """
    lt, i, gt = lo, lo, hi
    while i <= gt:
        v = a[i]
        if v < pivot:
            a[lt], a[i] = v, a[lt]
            lt += 1
            i += 1
        elif v > pivot:
            a[i], a[gt] = a[gt], v
            gt -= 1
        else:
            i += 1
    return lt, gt
