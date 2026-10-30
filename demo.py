"""A short tour of everything in the repository. Run: python demo.py"""

import random

from data_structures import (ArrayQueue, ArrayStack, DynamicArray, LinkedQueue, LinkedStack,
                             Matrix, RootedTree, SinglyLinkedList)
from selection import deterministic_select, randomized_select


def main():
    print("=== Part 1: selection ===")
    rng = random.Random(1)
    data = [rng.randint(0, 99) for _ in range(15)]
    print("data:          ", data)
    print("sorted:        ", sorted(data))
    for k in (1, 8, 15):
        stats = {}
        d = deterministic_select(data, k)
        r = randomized_select(data, k, rng=rng, stats=stats)
        print(f"k={k:2d}: deterministic={d:2d}  randomized={r:2d}  "
              f"(randomized used {stats['partitions']} partition passes)")
    print("median of 10 equal values:", deterministic_select([5] * 10, 5))

    print("\n=== Part 2: data structures ===")
    a = DynamicArray([10, 20, 30])
    a.insert(1, 15)
    a.pop(0)
    print("DynamicArray:", a, "capacity", a.capacity)

    m = Matrix.from_rows([[1, 2], [3, 4]])
    m.insert_row(2, [5, 6])
    m.insert_col(0, [0, 0, 0])
    print("Matrix after inserting a row and a column:", m.to_lists())
    print("Transposed:", m.transpose().to_lists())

    for cls in (ArrayStack, LinkedStack):
        s = cls()
        for x in "abc":
            s.push(x)
        print(f"{cls.__name__}: pop order", [s.pop() for _ in range(3)])

    for cls in (ArrayQueue, LinkedQueue):
        q = cls()
        for x in "abc":
            q.enqueue(x)
        print(f"{cls.__name__}: dequeue order", [q.dequeue() for _ in range(3)])

    ll = SinglyLinkedList([1, 2, 3])
    ll.insert_head(0)
    ll.insert_at(2, 1.5)
    ll.delete_value(3)
    print("SinglyLinkedList:", ll)
    ll.reverse()
    print("reversed:        ", ll)

    t = RootedTree("root")
    a1 = t.add_child(t.root, "a")
    b1 = t.add_child(t.root, "b")
    t.add_child(a1, "a1")
    t.add_child(a1, "a2")
    t.add_child(b1, "b1")
    print("RootedTree (height", t.height(), "):")
    print(t.pretty())
    print("level order:", t.level_order())


if __name__ == "__main__":
    main()
