"""Singly linked list with head and tail pointers.

Each node stores a value and a reference to the next node. Keeping a tail
pointer makes appending O(1); without it, append would need an O(n) walk.

Complexities (n = length):

    insert_head (prepend)          O(1)
    insert_tail (append)           O(1)    thanks to the tail pointer
    insert_at(i, x)                O(i)    walk to node i-1, then O(1) splice
    insert_after(node, x)          O(1)    when you already hold the node
    delete_head                    O(1)
    delete_tail                    O(n)    must find the predecessor (singly linked!)
    delete_at(i)                   O(i)
    delete_value(x)                O(n)
    get(i) / search(x)             O(n)
    traverse / iterate             O(n)
    reverse                        O(n), O(1) extra space
"""

from typing import Any, Callable, Iterable, Iterator, List, Optional


class Node:
    __slots__ = ("value", "next")

    def __init__(self, value: Any, nxt: "Optional[Node]" = None) -> None:
        self.value = value
        self.next = nxt

    def __repr__(self) -> str:
        return f"Node({self.value!r})"


class SinglyLinkedList:
    def __init__(self, items: Optional[Iterable[Any]] = None) -> None:
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self._size = 0
        if items is not None:
            for x in items:
                self.insert_tail(x)

    # ---- basics ----------------------------------------------------------
    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self.head is None

    def __iter__(self) -> Iterator[Any]:
        node = self.head
        while node is not None:
            yield node.value
            node = node.next

    def traverse(self, visit: Callable[[Any], None]) -> None:
        """Call ``visit(value)`` on every element, front to back. O(n)."""
        node = self.head
        while node is not None:
            visit(node.value)
            node = node.next

    def to_list(self) -> List[Any]:
        return list(self)

    def __repr__(self) -> str:
        return " -> ".join(repr(v) for v in self) + (" -> None" if self.head else "None")

    def _node_at(self, i: int) -> Node:
        if not 0 <= i < self._size:
            raise IndexError("list index out of range")
        node = self.head
        for _ in range(i):
            node = node.next  # type: ignore[union-attr]
        return node  # type: ignore[return-value]

    def get(self, i: int) -> Any:
        """Value at position ``i``. O(i): no random access in a linked list."""
        return self._node_at(i).value

    def search(self, value: Any) -> int:
        """Index of the first node equal to ``value``, or -1. O(n)."""
        for i, v in enumerate(self):
            if v == value:
                return i
        return -1

    def __contains__(self, value: Any) -> bool:
        return self.search(value) != -1

    # ---- insertion -------------------------------------------------------
    def insert_head(self, value: Any) -> Node:
        node = Node(value, self.head)
        self.head = node
        if self.tail is None:
            self.tail = node
        self._size += 1
        return node

    def insert_tail(self, value: Any) -> Node:
        node = Node(value)
        if self.tail is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self._size += 1
        return node

    def insert_after(self, node: Node, value: Any) -> Node:
        """Splice a new node right after ``node``. O(1)."""
        new = Node(value, node.next)
        node.next = new
        if node is self.tail:
            self.tail = new
        self._size += 1
        return new

    def insert_at(self, i: int, value: Any) -> Node:
        """Insert so the new value ends up at index ``i`` (0 <= i <= len). O(i)."""
        if not 0 <= i <= self._size:
            raise IndexError("insert index out of range")
        if i == 0:
            return self.insert_head(value)
        if i == self._size:
            return self.insert_tail(value)
        return self.insert_after(self._node_at(i - 1), value)

    # ---- deletion --------------------------------------------------------
    def delete_head(self) -> Any:
        if self.head is None:
            raise IndexError("delete from empty list")
        node = self.head
        self.head = node.next
        if self.head is None:
            self.tail = None
        self._size -= 1
        return node.value

    def _unlink_after(self, prev: Optional[Node]) -> Any:
        """Remove the node after ``prev`` (or the head if prev is None)."""
        if prev is None:
            return self.delete_head()
        target = prev.next
        assert target is not None
        prev.next = target.next
        if target is self.tail:
            self.tail = prev
        self._size -= 1
        return target.value

    def delete_tail(self) -> Any:
        """O(n): a singly linked node does not know its predecessor."""
        if self.head is None:
            raise IndexError("delete from empty list")
        if self.head is self.tail:
            return self.delete_head()
        return self._unlink_after(self._node_at(self._size - 2))

    def delete_at(self, i: int) -> Any:
        if not 0 <= i < self._size:
            raise IndexError("list index out of range")
        return self._unlink_after(None if i == 0 else self._node_at(i - 1))

    def delete_value(self, value: Any) -> bool:
        """Remove the first node equal to ``value``. Returns True if found. O(n)."""
        prev, node = None, self.head
        while node is not None:
            if node.value == value:
                self._unlink_after(prev)
                return True
            prev, node = node, node.next
        return False

    # ---- extras ----------------------------------------------------------
    def reverse(self) -> None:
        """Reverse in place by re-pointing links. O(n) time, O(1) space."""
        prev, node = None, self.head
        self.tail = self.head
        while node is not None:
            nxt = node.next
            node.next = prev
            prev, node = node, nxt
        self.head = prev
