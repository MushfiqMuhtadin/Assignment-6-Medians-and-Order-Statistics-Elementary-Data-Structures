"""Rooted trees with linked nodes, the left-child / right-sibling representation.

(Optional part of the assignment, CLRS section 10.4.)

A node with arbitrarily many children needs only three pointers:

    parent        -> its parent (None for the root)
    left_child    -> its first (leftmost) child
    right_sibling -> the next child of the same parent

So the children of a node form a singly linked list that starts at
``left_child`` and follows ``right_sibling``. Every node has the same fixed
size no matter how many children it has: O(n) space for n nodes.

Complexities (n nodes, d = number of children of a node, h = height):

    add_child (as first child)      O(1)
    add_child (as last child)       O(d)    walk the sibling list
    children(node)                  O(d)
    parent / depth(node)            O(1) / O(depth)
    remove_subtree(node)            O(d_parent)  unlink from sibling list
    preorder / postorder / BFS      O(n)
    height                          O(n)
    find(value)                     O(n)
"""

from typing import Any, Callable, Iterator, List, Optional

from .queue import LinkedQueue
from .stack import ArrayStack


class TreeNode:
    __slots__ = ("value", "parent", "left_child", "right_sibling")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.parent: Optional[TreeNode] = None
        self.left_child: Optional[TreeNode] = None
        self.right_sibling: Optional[TreeNode] = None

    def children(self) -> Iterator["TreeNode"]:
        c = self.left_child
        while c is not None:
            yield c
            c = c.right_sibling

    def is_leaf(self) -> bool:
        return self.left_child is None

    def __repr__(self) -> str:
        return f"TreeNode({self.value!r})"


class RootedTree:
    def __init__(self, root_value: Any) -> None:
        self.root = TreeNode(root_value)
        self._size = 1

    def __len__(self) -> int:
        return self._size

    # ---- building --------------------------------------------------------
    def add_child(self, parent: TreeNode, value: Any, first: bool = False) -> TreeNode:
        """Attach a new child to ``parent``. Appended as the last child by
        default (O(d)); pass ``first=True`` for O(1) prepend."""
        node = TreeNode(value)
        node.parent = parent
        if first or parent.left_child is None:
            node.right_sibling = parent.left_child
            parent.left_child = node
        else:
            c = parent.left_child
            while c.right_sibling is not None:
                c = c.right_sibling
            c.right_sibling = node
        self._size += 1
        return node

    def remove_subtree(self, node: TreeNode) -> None:
        """Detach ``node`` and all its descendants."""
        if node is self.root:
            raise ValueError("cannot remove the root")
        parent = node.parent
        assert parent is not None
        if parent.left_child is node:
            parent.left_child = node.right_sibling
        else:
            c = parent.left_child
            while c is not None and c.right_sibling is not node:
                c = c.right_sibling
            if c is None:
                raise ValueError("node is not in this tree")
            c.right_sibling = node.right_sibling
        self._size -= sum(1 for _ in self._preorder_nodes(node))
        node.parent = node.right_sibling = None

    # ---- queries ---------------------------------------------------------
    @staticmethod
    def depth(node: TreeNode) -> int:
        d = 0
        while node.parent is not None:
            node = node.parent
            d += 1
        return d

    def height(self, node: Optional[TreeNode] = None) -> int:
        """Edges on the longest root-to-leaf path. Iterative BFS, O(n)."""
        start = node or self.root
        h = 0
        q = LinkedQueue()
        q.enqueue((start, 0))
        while not q.is_empty():
            n, d = q.dequeue()
            h = max(h, d)
            for c in n.children():
                q.enqueue((c, d + 1))
        return h

    def find(self, value: Any) -> Optional[TreeNode]:
        for n in self._preorder_nodes(self.root):
            if n.value == value:
                return n
        return None

    # ---- traversals ------------------------------------------------------
    @staticmethod
    def _preorder_nodes(start: TreeNode) -> Iterator[TreeNode]:
        """Iterative preorder using an explicit stack (no recursion limit)."""
        stack = ArrayStack()
        stack.push(start)
        while not stack.is_empty():
            n = stack.pop()
            yield n
            kids = list(n.children())
            for c in reversed(kids):
                stack.push(c)

    def preorder(self) -> List[Any]:
        return [n.value for n in self._preorder_nodes(self.root)]

    def postorder(self) -> List[Any]:
        out: List[Any] = []

        def visit(n: TreeNode) -> None:
            for c in n.children():
                visit(c)
            out.append(n.value)

        visit(self.root)
        return out

    def level_order(self) -> List[Any]:
        out: List[Any] = []
        q = LinkedQueue()
        q.enqueue(self.root)
        while not q.is_empty():
            n = q.dequeue()
            out.append(n.value)
            for c in n.children():
                q.enqueue(c)
        return out

    def traverse(self, visit: Callable[[Any], None]) -> None:
        for n in self._preorder_nodes(self.root):
            visit(n.value)

    def pretty(self) -> str:
        lines: List[str] = []

        def walk(n: TreeNode, prefix: str, last: bool, is_root: bool) -> None:
            lines.append(("" if is_root else prefix + ("└── " if last else "├── ")) + str(n.value))
            kids = list(n.children())
            for i, c in enumerate(kids):
                walk(c, "" if is_root else prefix + ("    " if last else "│   "), i == len(kids) - 1, False)

        walk(self.root, "", True, True)
        return "\n".join(lines)
