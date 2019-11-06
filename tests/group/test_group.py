#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import unittest
import numpy as np
from group.group import Group


class TestGroup(unittest.TestCase):
    def test_1(self):
        Cayley_table = [
            [0, 1],
            [1, 0],
        ]
        group = Group(Cayley_table)

        orders_of_conjugacy_classes = group.get_orders_of_conjugacy_classes()
        orders_of_conjugacy_classes_expected = [1, 1]
        self.assertListEqual(
            list(orders_of_conjugacy_classes),
            orders_of_conjugacy_classes_expected)


def main():
    Cayley_table = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ]
    run(Cayley_table)
    Cayley_table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]
    run(Cayley_table)
    Cayley_table = [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
    ]
    run(Cayley_table)
    # Symmorphic group: S3
    Cayley_table = [
        [0, 1, 2, 3, 4, 5],
        [1, 0, 4, 5, 2, 3],
        [2, 5, 0, 4, 3, 1],
        [3, 4, 5, 0, 1, 2],
        [4, 3, 1, 2, 5, 0],
        [5, 2, 3, 1, 0, 4],
    ]
    run(Cayley_table, subset=np.array([4]))

    print("Quaternion group")
    Cayley_table = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 3, 2, 5, 4, 7, 6],
        [2, 3, 1, 0, 6, 7, 5, 4],
        [3, 2, 0, 1, 7, 6, 4, 5],
        [4, 5, 7, 6, 1, 0, 2, 3],
        [5, 4, 6, 7, 0, 1, 3, 2],
        [6, 7, 4, 5, 3, 2, 1, 0],
        [7, 6, 5, 4, 2, 3, 0, 1],
    ]
    run(Cayley_table)

    # Dihedral group: D8
    Cayley_table = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 0, 5, 6, 7, 4],
        [2, 3, 0, 1, 6, 7, 4, 5],
        [3, 0, 1, 2, 7, 4, 5, 6],
        [4, 7, 6, 5, 0, 3, 2, 1],
        [5, 4, 7, 6, 1, 0, 3, 2],
        [6, 5, 4, 7, 2, 1, 0, 3],
        [7, 6, 5, 4, 3, 2, 1, 0],
    ]
    run(Cayley_table)


def run(Cayley_table, subset=None):
    Cayley_table = np.array(Cayley_table)
    group = Group(Cayley_table)
    print("identity:")
    print(group.get_identity())
    print("center:")
    print(group.get_center())
    print("orders_of_elements:")
    print(group.get_orders_of_elements())
    print("exponent:")
    print(group.get_exponent())
    print("conjugacy_classes:")
    print(group.get_conjugacy_classes())
    print("commutator_subgroup:")
    print(group.get_commutator_subgroup())
    print("crst:")
    print(group.get_crst())
    if subset is not None:
        print("subset:")
        print(subset)
        group.create_centralizer(subset)
        group.create_normalizer(subset)
        print("centralizer:")
        print(group.get_centralizer())
        print("normalizer:")
        print(group.get_normalizer())


if __name__ == "__main__":
    main()
    unittest.main()
