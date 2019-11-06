#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__author__ = "Yuji Ikeda"

import unittest
import numpy as np
from upho.analysis.mappings_modifier import MappingsModifier


class TestMappingsInverter(unittest.TestCase):
    def test_invert_mappings(self):
        mappings = [
            [0, 1, 2, 3],
            [0, 2, 1, 3],
            [1, 2, 3, 0],
        ]
        mappings_inv_expected = [
            [0, 1, 2, 3],
            [0, 2, 1, 3],
            [3, 0, 1, 2],
        ] 

        mappings_inv = MappingsModifier(mappings).invert_mappings()
        self.assertTrue(np.all(mappings_inv == mappings_inv_expected))

    def test_expand_mappings(self):
        mappings = [
            [0, 1, 2, 3],
            [0, 2, 1, 3],
            [1, 2, 3, 0],
        ]
        n = 3
        expanded_mappings_expected = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11],
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2],  # Good
        ]

        expanded_mappings = MappingsModifier(mappings).expand_mappings(n)
        self.assertTrue(
            np.all(expanded_mappings == expanded_mappings_expected))

    def test_expand_mappings_inv(self):
        mappings = [
            [0, 1, 2, 3],
            [0, 2, 1, 3],
            [1, 2, 3, 0],
        ]
        n = 3
        expanded_mappings_inv_expected = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11],
            [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        ]

        mappings_modifier = MappingsModifier(mappings)
        expanded_mappings_inv = mappings_modifier.expand_mappings(
            n, is_inverse=True)
        self.assertTrue(
            np.all(expanded_mappings_inv == expanded_mappings_inv_expected))

    def test_expand_mappings2(self):
        mappings = [
            [
                [0, 1, 2, 3],
                [0, 2, 1, 3],
            ],
            [
                [1, 2, 3, 0],
                [2, 3, 0, 1],
            ],
        ]
        n = 3
        expanded_mappings_expected = [
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11],
            ],
            [
                [3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2],  # Good
                [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5],  # Good
            ],
        ]

        expanded_mappings = MappingsModifier(mappings).expand_mappings(n)
        self.assertTrue(
            np.all(expanded_mappings == expanded_mappings_expected))


if __name__ == "__main__":
    unittest.main()
