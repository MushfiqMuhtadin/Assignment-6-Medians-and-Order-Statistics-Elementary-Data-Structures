"""Dynamic array built on a fixed-size block of memory.

"From scratch" rule used throughout this package: the only thing we borrow
from Python is a *fixed-length* block of slots, created once by ``_alloc``
(``[None] * capacity``). We never call ``list.append``, ``insert``, ``pop``
or slicing on it, so it behaves like a raw C array. Growing, shifting and
shrinking are all done by hand, which is exactly what we want to analyse.

Complexities (n = number of stored items):

    access / update  a[i]          O(1)
    append                          O(1) amortized, O(n) worst (resize)
    insert(i, x)                    O(n - i)   shifts the tail right
    delete(i) / pop(i)              O(n - i)   shifts the tail left
    pop() from the end              O(1) amortized
    search (index_of / contains)    O(n)

Resizing policy: double when full, halve when only 1/4 full. Shrinking at
1/4 (not 1/2) avoids thrashing when appends and pops alternate at a boundary
and keeps both operations O(1) amortized (potential-method argument, CLRS 17.4).
"""

from typing import Any, Iterable, Iterator, List, Optional


def _alloc(capacity: int) -> List[Any]:
    """Allocate a fixed-size block of ``capacity`` empty slots."""
    return [None] * capacity


class DynamicArray:
    """A resizable array with explicit capacity management."""

    _MIN_CAPACITY = 4

    def __init__(self, items: Optional[Iterable[Any]] = None) -> None:
        self._capacity = self._MIN_CAPACITY
        self._data = _alloc(self._capacity)
        self._size = 0
        if items is not None:
            for x in items:
                self.append(x)

    # ---- size / capacity -------------------------------------------------
    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def is_empty(self) -> bool:
        return self._size == 0

    # ---- access ----------------------------------------------------------
    def _check_index(self, i: int) -> int:
        if i < 0:
            i += self._size
        if not 0 <= i < self._size:
            raise IndexError("array index out of range")
        return i

    def __getitem__(self, i: int) -> Any:
        """O(1): address arithmetic on a contiguous block."""
        return self._data[self._check_index(i)]

    def __setitem__(self, i: int, value: Any) -> None:
        """O(1)."""
        self._data[self._check_index(i)] = value

    def __iter__(self) -> Iterator[Any]:
        for i in range(self._size):
            yield self._data[i]

    def __repr__(self) -> str:
        return f"DynamicArray([{', '.join(repr(x) for x in self)}])"

    def to_list(self) -> List[Any]:
        return [self._data[i] for i in range(self._size)]

    # ---- insertion -------------------------------------------------------
    def _resize(self, new_capacity: int) -> None:
        """Copy every item into a fresh block: O(n)."""
        new_data = _alloc(new_capacity)
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data
        self._capacity = new_capacity

    def append(self, value: Any) -> None:
        """Add at the end. O(1) amortized: a resize costing n happens only
        after n/2 cheap appends since the previous resize."""
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        self._data[self._size] = value
        self._size += 1

    def insert(self, i: int, value: Any) -> None:
        """Insert before position ``i`` (0 <= i <= len). O(n - i)."""
        if i < 0:
            i += self._size
        if not 0 <= i <= self._size:
            raise IndexError("insert index out of range")
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        for j in range(self._size, i, -1):  # shift tail right by one
            self._data[j] = self._data[j - 1]
        self._data[i] = value
        self._size += 1

    # ---- deletion --------------------------------------------------------
    def _maybe_shrink(self) -> None:
        if self._capacity > self._MIN_CAPACITY and self._size <= self._capacity // 4:
            self._resize(max(self._MIN_CAPACITY, self._capacity // 2))

    def pop(self, i: int = -1) -> Any:
        """Remove and return the item at ``i`` (default: last). O(n - i)."""
        i = self._check_index(i)
        value = self._data[i]
        for j in range(i, self._size - 1):  # shift tail left by one
            self._data[j] = self._data[j + 1]
        self._size -= 1
        self._data[self._size] = None  # drop the reference for the GC
        self._maybe_shrink()
        return value

    def delete(self, i: int) -> None:
        """Remove the item at ``i``. O(n - i)."""
        self.pop(i)

    def remove(self, value: Any) -> None:
        """Remove the first occurrence of ``value``. O(n)."""
        self.pop(self.index_of(value))

    # ---- search ----------------------------------------------------------
    def index_of(self, value: Any) -> int:
        """Linear search. O(n). Raises ValueError if absent."""
        for i in range(self._size):
            if self._data[i] == value:
                return i
        raise ValueError(f"{value!r} not in array")

    def __contains__(self, value: Any) -> bool:
        try:
            self.index_of(value)
            return True
        except ValueError:
            return False
