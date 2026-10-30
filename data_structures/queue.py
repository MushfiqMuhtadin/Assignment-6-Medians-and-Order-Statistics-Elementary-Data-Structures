"""Queues (FIFO).

ArrayQueue  - circular buffer on a fixed block (head index wraps around).
LinkedQueue - singly linked list with head (front) and tail (back) pointers.
NaiveArrayQueue - *anti-pattern kept for the benchmark*: dequeue shifts every
                  remaining item left, so it costs O(n).

                     ArrayQueue (circular)   LinkedQueue      NaiveArrayQueue
    enqueue          O(1) amortized          O(1)             O(1) amortized
    dequeue          O(1) amortized          O(1)             O(n)
    peek / is_empty  O(1)                    O(1)             O(1)

Why circular: with a plain array, the front of the queue drifts right as
items leave. Either we shift everything back (O(n) per dequeue) or we waste
the vacated prefix forever. Treating the block as a ring, slot
``(head + i) % capacity`` holds the i-th item, so both ends move in O(1) and
freed slots are reused.
"""

from typing import Any, Iterator, Optional

from .array import _alloc


class EmptyQueueError(IndexError):
    pass


class ArrayQueue:
    """FIFO queue on a circular buffer with doubling / halving."""

    def __init__(self, capacity: int = 4) -> None:
        self._data = _alloc(max(1, capacity))
        self._head = 0  # index of the front item
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def _resize(self, cap: int) -> None:
        """Unroll the ring into a fresh block starting at index 0. O(n)."""
        new = _alloc(cap)
        old_cap = len(self._data)
        for i in range(self._size):
            new[i] = self._data[(self._head + i) % old_cap]
        self._data = new
        self._head = 0

    def enqueue(self, x: Any) -> None:
        cap = len(self._data)
        if self._size == cap:
            self._resize(2 * cap)
            cap *= 2
        self._data[(self._head + self._size) % cap] = x
        self._size += 1

    def dequeue(self) -> Any:
        if self._size == 0:
            raise EmptyQueueError("dequeue from empty queue")
        x = self._data[self._head]
        self._data[self._head] = None
        self._head = (self._head + 1) % len(self._data)
        self._size -= 1
        cap = len(self._data)
        if cap >= 4 and self._size <= cap // 4:
            self._resize(cap // 2)
        return x

    def peek(self) -> Any:
        if self._size == 0:
            raise EmptyQueueError("peek at empty queue")
        return self._data[self._head]

    def __iter__(self) -> Iterator[Any]:
        """Front to back."""
        cap = len(self._data)
        for i in range(self._size):
            yield self._data[(self._head + i) % cap]


class NaiveArrayQueue:
    """Queue on a non-circular array: dequeue shifts all items. O(n) dequeue.
    Exists only to show, in the benchmark, why the circular buffer matters."""

    def __init__(self) -> None:
        self._data = _alloc(4)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def enqueue(self, x: Any) -> None:
        if self._size == len(self._data):
            new = _alloc(2 * len(self._data))
            for i in range(self._size):
                new[i] = self._data[i]
            self._data = new
        self._data[self._size] = x
        self._size += 1

    def dequeue(self) -> Any:
        if self._size == 0:
            raise EmptyQueueError("dequeue from empty queue")
        x = self._data[0]
        for i in range(1, self._size):
            self._data[i - 1] = self._data[i]
        self._size -= 1
        self._data[self._size] = None
        return x


class _Node:
    __slots__ = ("value", "next")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.next: Optional[_Node] = None


class LinkedQueue:
    """FIFO queue on a singly linked list: dequeue at head, enqueue at tail."""

    def __init__(self) -> None:
        self._head: Optional[_Node] = None
        self._tail: Optional[_Node] = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def enqueue(self, x: Any) -> None:
        node = _Node(x)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._size += 1

    def dequeue(self) -> Any:
        if self._head is None:
            raise EmptyQueueError("dequeue from empty queue")
        node = self._head
        self._head = node.next
        if self._head is None:
            self._tail = None
        self._size -= 1
        return node.value

    def peek(self) -> Any:
        if self._head is None:
            raise EmptyQueueError("peek at empty queue")
        return self._head.value

    def __iter__(self) -> Iterator[Any]:
        node = self._head
        while node is not None:
            yield node.value
            node = node.next
