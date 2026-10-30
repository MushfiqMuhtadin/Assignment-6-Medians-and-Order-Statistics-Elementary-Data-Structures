"""Correctness tests for Part 1. Every result is cross-checked against sorted()."""

import random
import unittest

from selection import (deterministic_median, deterministic_select, naive_quickselect,
                       randomized_median, randomized_select)

ALGOS = {
    "deterministic": deterministic_select,
    "randomized": lambda a, k: randomized_select(a, k, rng=random.Random(42)),
    "naive": naive_quickselect,
}


class TestSelection(unittest.TestCase):
    def check_all_ranks(self, arr):
        expected = sorted(arr)
        for name, fn in ALGOS.items():
            for k in range(1, len(arr) + 1):
                with self.subTest(algo=name, k=k, n=len(arr)):
                    self.assertEqual(fn(arr, k), expected[k - 1])

    def test_single_element(self):
        self.check_all_ranks([5])

    def test_small_fixed(self):
        self.check_all_ranks([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])

    def test_sorted_and_reverse(self):
        self.check_all_ranks(list(range(53)))
        self.check_all_ranks(list(range(53, 0, -1)))

    def test_all_equal(self):
        self.check_all_ranks([7] * 40)

    def test_two_values(self):
        self.check_all_ranks([0, 1] * 25)

    def test_negative_and_float(self):
        self.check_all_ranks([-2.5, 3.0, -100, 0, 0.0, 1e9, -1e-9])

    def test_strings(self):
        self.check_all_ranks(list("thequickbrownfoxjumpsoverthelazydog"))

    def test_random_many(self):
        rng = random.Random(0)
        for _ in range(200):
            n = rng.randint(1, 120)
            arr = [rng.randint(-20, 20) for _ in range(n)]  # lots of duplicates
            k = rng.randint(1, n)
            exp = sorted(arr)[k - 1]
            self.assertEqual(deterministic_select(arr, k), exp)
            self.assertEqual(randomized_select(arr, k), exp)

    def test_large(self):
        rng = random.Random(1)
        arr = [rng.random() for _ in range(20001)]
        s = sorted(arr)
        for k in (1, 7, 10001, 19999, 20001):
            self.assertEqual(deterministic_select(arr, k), s[k - 1])
            self.assertEqual(randomized_select(arr, k), s[k - 1])

    def test_large_all_equal_is_fast(self):
        # Three-way partition: finishes in a single pass.
        stats = {}
        self.assertEqual(randomized_select([1] * 100000, 50000, stats=stats), 1)
        self.assertEqual(stats["partitions"], 1)

    def test_median_helpers(self):
        arr = [9, 1, 8, 2, 7, 3]
        self.assertEqual(deterministic_median(arr), 3)
        self.assertEqual(randomized_median(arr), 3)

    def test_input_not_mutated(self):
        arr = [5, 3, 1, 4, 2]
        copy = list(arr)
        deterministic_select(arr, 2)
        randomized_select(arr, 2)
        self.assertEqual(arr, copy)

    def test_tuple_input(self):
        self.assertEqual(deterministic_select((4, 2, 6), 2), 4)

    def test_errors(self):
        for fn in (deterministic_select, randomized_select, naive_quickselect):
            with self.assertRaises(ValueError):
                fn([], 1)
            with self.assertRaises(ValueError):
                fn([1, 2], 0)
            with self.assertRaises(ValueError):
                fn([1, 2], 3)
            with self.assertRaises(TypeError):
                fn([1, 2], 1.0)


if __name__ == "__main__":
    unittest.main()
