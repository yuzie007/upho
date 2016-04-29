#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

from mathtools import gcd, lcm
import unittest


class TestMathtools(unittest.TestCase):
    def setUp(self):
        self._integers = [36, 120, 24]
        self._gcms = [
            [ 36,  12,  12],
            [ 12, 120,  24],
            [ 12,  24,  24],
        ]
        self._lcms = [
            [ 36, 360,  72],
            [360, 120, 120],
            [ 72, 120,  24],
        ]

    # def test_0(self):
    #     self.assertEqual(gcd(), 0)
    #     self.assertEqual(lcm(), 0)

    def test_1(self):
        for i in self._integers:
            self.assertEqual(gcd(i), i)
            self.assertEqual(lcm(i), i)

    def test_2(self):
        for i, a in enumerate(self._integers):
            for j, b in enumerate(self._integers):
                self.assertEqual(gcd(a, b), self._gcms[i][j])
                self.assertEqual(lcm(a, b), self._lcms[i][j])
                self.assertEqual(gcd(a, b) * lcm(a, b), a * b)

    def test_3(self):
        a, b, c = self._integers
        self.assertEqual(gcd(a, b, c), 12)
        self.assertEqual(lcm(a, b, c), 360)


if __name__ == "__main__":
    unittest.main()
