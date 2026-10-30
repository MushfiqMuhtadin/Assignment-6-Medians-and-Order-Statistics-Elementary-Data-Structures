"""Stacks (LIFO), two implementations for comparison.

ArrayStack  - top of stack is the end of a growable fixed-block array.
LinkedStack - top of stack is the head of a singly linked chain.

                     ArrayStack              LinkedStack
    push             O(1) amortized          O(1) worst case
    pop              O(1) amortized          O(1) worst case
    peek / is_empty  O(1)                    O(1)
    memory / item    1 slot (+ up to 4x      1 node object: value + next
                     slack after shrink)     pointer + object header
"""

from typing import Any, Iterator, Optional

from .array import _alloc


class EmptyStackError(IndexError):
    pass


class ArrayStack:
    """Stack backed by a manually-resized fixed block."""

    def __init__(self, capacity: int = 4) -> None:
        self._data = _alloc(max(1, capacity))
        self._top = 0  # number of items == index of next free slot

    def __len__(self) -> int:
        return self._top

    def is_empty(self) -> bool:
        return self._top == 0

    def _resize(self, cap: int) -> None:
        new = _alloc(cap)
        for i in range(self._top):
            new[i] = self._data[i]
        self._data = new

    def push(self, x: Any) -> None:
        if self._top == len(self._data):
            self._resize(2 * len(self._data))
        self._data[self._top] = x
        self._top += 1

    def pop(self) -> Any:
        if self._top == 0:
            raise EmptyStackError("pop from empty stack")
        self._top -= 1
        x = self._data[self._top]
        self._data[self._top] = None
        if 4 <= len(self._data) and self._top <= len(self._data) // 4:
            self._resize(len(self._data) // 2)
        return x

    def peek(self) -> Any:
        if self._top == 0:
            raise EmptyStackError("peek at empty stack")
        return self._data[self._top - 1]

    def __iter__(self) -> Iterator[Any]:
        """Top to bottom."""
        for i in range(self._top - 1, -1, -1):
            yield self._data[i]


class _Node:
    __slots__ = ("value", "next")

    def __init__(self, value: Any, nxt: "Optional[_Node]" = None) -> None:
        self.value = value
        self.next = nxt


class LinkedStack:
    """Stack backed by a singly linked list (push/pop at the head)."""

    def __init__(self) -> None:
        self._head: Optional[_Node] = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._head is None

    def push(self, x: Any) -> None:
        self._head = _Node(x, self._head)
        self._size += 1

    def pop(self) -> Any:
        if self._head is None:
            raise EmptyStackError("pop from empty stack")
        node = self._head
        self._head = node.next
        self._size -= 1
        return node.value

    def peek(self) -> Any:
        if self._head is None:
            raise EmptyStackError("peek at empty stack")
        return self._head.value

    def __iter__(self) -> Iterator[Any]:
        node = self._head
        while node is not None:
            yield node.value
            node = node.next
