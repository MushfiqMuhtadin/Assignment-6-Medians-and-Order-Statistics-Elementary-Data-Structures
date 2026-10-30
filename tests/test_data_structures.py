"""Correctness tests for Part 2."""

import random
import unittest
from collections import deque

from data_structures import (ArrayQueue, ArrayStack, DynamicArray, EmptyQueueError,
                             EmptyStackError, LinkedQueue, LinkedStack, Matrix,
                             NaiveArrayQueue, RootedTree, SinglyLinkedList)


class TestDynamicArray(unittest.TestCase):
    def test_append_access(self):
        a = DynamicArray()
        for i in range(100):
            a.append(i)
        self.assertEqual(len(a), 100)
        self.assertEqual(a[0], 0)
        self.assertEqual(a[-1], 99)
        a[5] = "x"
        self.assertEqual(a[5], "x")
        self.assertGreaterEqual(a.capacity, 100)

    def test_insert_delete(self):
        a = DynamicArray([1, 2, 3])
        a.insert(0, 0)
        a.insert(4, 4)
        a.insert(2, 1.5)
        self.assertEqual(a.to_list(), [0, 1, 1.5, 2, 3, 4])
        self.assertEqual(a.pop(), 4)
        self.assertEqual(a.pop(0), 0)
        a.delete(1)
        a.remove(3)
        self.assertEqual(a.to_list(), [1, 2])
        self.assertIn(2, a)
        self.assertNotIn(9, a)

    def test_shrinks(self):
        a = DynamicArray(range(1000))
        big = a.capacity
        while len(a) > 3:
            a.pop()
        self.assertLess(a.capacity, big)
        self.assertEqual(a.to_list(), [0, 1, 2])

    def test_errors(self):
        a = DynamicArray()
        with self.assertRaises(IndexError):
            a[0]
        with self.assertRaises(IndexError):
            a.pop()
        with self.assertRaises(IndexError):
            a.insert(2, 1)
        with self.assertRaises(ValueError):
            a.index_of(1)

    def test_random_against_list(self):
        rng = random.Random(3)
        a, ref = DynamicArray(), []
        for _ in range(2000):
            op = rng.random()
            if op < 0.5 or not ref:
                i = rng.randint(0, len(ref))
                v = rng.randint(0, 99)
                a.insert(i, v)
                ref.insert(i, v)
            else:
                i = rng.randrange(len(ref))
                self.assertEqual(a.pop(i), ref.pop(i))
        self.assertEqual(a.to_list(), ref)


class TestMatrix(unittest.TestCase):
    def test_access_and_shape(self):
        m = Matrix(2, 3)
        m[1, 2] = 5
        self.assertEqual(m[1, 2], 5)
        self.assertEqual(m.shape, (2, 3))
        with self.assertRaises(IndexError):
            m[2, 0]

    def test_row_col_ops(self):
        m = Matrix.from_rows([[1, 2], [3, 4]])
        m.insert_row(1, [9, 9])
        self.assertEqual(m.to_lists(), [[1, 2], [9, 9], [3, 4]])
        m.insert_col(0, [0, 0, 0])
        self.assertEqual(m.to_lists(), [[0, 1, 2], [0, 9, 9], [0, 3, 4]])
        m.delete_row(1)
        m.delete_col(0)
        self.assertEqual(m.to_lists(), [[1, 2], [3, 4]])
        m.insert_row(2, [5, 6])
        m.insert_col(2, [7, 8, 9])
        self.assertEqual(m.to_lists(), [[1, 2, 7], [3, 4, 8], [5, 6, 9]])

    def test_arithmetic(self):
        a = Matrix.from_rows([[1, 2, 3], [4, 5, 6]])
        b = Matrix.from_rows([[7, 8], [9, 10], [11, 12]])
        self.assertEqual((a @ b).to_lists(), [[58, 64], [139, 154]])
        self.assertEqual(a.transpose().to_lists(), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual((a + a).to_lists(), [[2, 4, 6], [8, 10, 12]])
        with self.assertRaises(ValueError):
            a @ a


class StackContract:
    cls = None

    def test_lifo(self):
        s = self.cls()
        for i in range(500):
            s.push(i)
        self.assertEqual(s.peek(), 499)
        self.assertEqual(list(s)[:3], [499, 498, 497])
        for i in reversed(range(500)):
            self.assertEqual(s.pop(), i)
        self.assertTrue(s.is_empty())
        with self.assertRaises(EmptyStackError):
            s.pop()
        with self.assertRaises(EmptyStackError):
            s.peek()

    def test_interleaved(self):
        s, ref = self.cls(), []
        rng = random.Random(5)
        for _ in range(3000):
            if ref and rng.random() < 0.45:
                self.assertEqual(s.pop(), ref.pop())
            else:
                v = rng.random()
                s.push(v)
                ref.append(v)
            self.assertEqual(len(s), len(ref))


class TestArrayStack(StackContract, unittest.TestCase):
    cls = ArrayStack


class TestLinkedStack(StackContract, unittest.TestCase):
    cls = LinkedStack


class QueueContract:
    cls = None

    def test_fifo(self):
        q = self.cls()
        for i in range(500):
            q.enqueue(i)
        for i in range(500):
            self.assertEqual(q.dequeue(), i)
        self.assertTrue(q.is_empty())
        with self.assertRaises(EmptyQueueError):
            q.dequeue()

    def test_interleaved_wraparound(self):
        q, ref = self.cls(), deque()
        rng = random.Random(6)
        for _ in range(4000):
            if ref and rng.random() < 0.48:
                self.assertEqual(q.dequeue(), ref.popleft())
            else:
                v = rng.randint(0, 10**6)
                q.enqueue(v)
                ref.append(v)
            self.assertEqual(len(q), len(ref))
        while ref:
            self.assertEqual(q.dequeue(), ref.popleft())


class TestArrayQueue(QueueContract, unittest.TestCase):
    cls = ArrayQueue

    def test_iter_and_peek(self):
        q = ArrayQueue(capacity=4)
        for i in range(3):
            q.enqueue(i)
        q.dequeue()
        q.enqueue(3)
        q.enqueue(4)  # wraps around the ring
        self.assertEqual(list(q), [1, 2, 3, 4])
        self.assertEqual(q.peek(), 1)


class TestLinkedQueue(QueueContract, unittest.TestCase):
    cls = LinkedQueue


class TestNaiveQueue(QueueContract, unittest.TestCase):
    cls = NaiveArrayQueue


class TestSinglyLinkedList(unittest.TestCase):
    def test_insertions(self):
        ll = SinglyLinkedList()
        ll.insert_tail(2)
        ll.insert_head(1)
        ll.insert_tail(4)
        ll.insert_at(2, 3)
        ll.insert_at(0, 0)
        ll.insert_at(5, 5)
        self.assertEqual(ll.to_list(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(ll.tail.value, 5)
        self.assertEqual(ll.get(3), 3)

    def test_deletions(self):
        ll = SinglyLinkedList(range(6))
        self.assertEqual(ll.delete_head(), 0)
        self.assertEqual(ll.delete_tail(), 5)
        self.assertEqual(ll.tail.value, 4)
        self.assertEqual(ll.delete_at(1), 2)
        self.assertTrue(ll.delete_value(4))
        self.assertFalse(ll.delete_value(99))
        self.assertEqual(ll.to_list(), [1, 3])
        self.assertEqual(ll.tail.value, 3)
        ll.delete_tail()
        ll.delete_tail()
        self.assertTrue(ll.is_empty())
        self.assertIsNone(ll.tail)
        with self.assertRaises(IndexError):
            ll.delete_head()

    def test_search_traverse_reverse(self):
        ll = SinglyLinkedList("abcd")
        self.assertEqual(ll.search("c"), 2)
        self.assertEqual(ll.search("z"), -1)
        seen = []
        ll.traverse(seen.append)
        self.assertEqual(seen, list("abcd"))
        ll.reverse()
        self.assertEqual(ll.to_list(), list("dcba"))
        ll.insert_tail("e")
        self.assertEqual(ll.to_list(), list("dcbae"))

    def test_random_against_list(self):
        rng = random.Random(7)
        ll, ref = SinglyLinkedList(), []
        for _ in range(2000):
            r = rng.random()
            if r < 0.4 or not ref:
                i = rng.randint(0, len(ref))
                v = rng.randint(0, 50)
                ll.insert_at(i, v)
                ref.insert(i, v)
            elif r < 0.7:
                i = rng.randrange(len(ref))
                self.assertEqual(ll.delete_at(i), ref.pop(i))
            else:
                v = rng.randint(0, 50)
                found = v in ref
                if found:
                    ref.remove(v)
                self.assertEqual(ll.delete_value(v), found)
            self.assertEqual(len(ll), len(ref))
            if ref:
                self.assertEqual(ll.tail.value, ref[-1])
        self.assertEqual(ll.to_list(), ref)


class TestRootedTree(unittest.TestCase):
    def build(self):
        t = RootedTree("A")
        b = t.add_child(t.root, "B")
        c = t.add_child(t.root, "C")
        t.add_child(t.root, "D")
        t.add_child(b, "E")
        t.add_child(b, "F")
        g = t.add_child(c, "G")
        t.add_child(g, "H")
        return t

    def test_traversals(self):
        t = self.build()
        self.assertEqual(len(t), 8)
        self.assertEqual(t.preorder(), list("ABEFCGHD"))
        self.assertEqual(t.postorder(), list("EFBHGCDA"))
        self.assertEqual(t.level_order(), list("ABCDEFGH"))
        self.assertEqual(t.height(), 3)
        self.assertEqual(RootedTree.depth(t.find("H")), 3)

    def test_remove_and_first_child(self):
        t = self.build()
        t.remove_subtree(t.find("C"))
        self.assertEqual(t.preorder(), list("ABEFD"))
        self.assertEqual(len(t), 5)
        t.add_child(t.root, "Z", first=True)
        self.assertEqual(t.level_order(), list("AZBDEF"))
        with self.assertRaises(ValueError):
            t.remove_subtree(t.root)
        self.assertIn("A", t.pretty())


if __name__ == "__main__":
    unittest.main()
