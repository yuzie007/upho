#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__author__ = "Yuji Ikeda"

import unittest
import numpy as np
from phonopy.interface.vasp import read_vasp
from ph_unfolder.structure.unfolder_symmetry import UnfolderSymmetry


class TestUnfolderSymmetry(unittest.TestCase):
    def setUp(self):
        atoms = read_vasp("POSCAR_sc")
        self._symmetry = UnfolderSymmetry(atoms)

    def test_000(self):
        kpoint = np.array([0.0, 0.0, 0.0])
        (rotations_kpoint,
         translations_kpoint) = self._symmetry.get_group_of_kpoint(kpoint)

        self.assertEqual(len(rotations_kpoint), 48)

    def test_100(self):
        kpoint = np.array([0.5, 0.0, 0.0])  # X: D_4h
        (rotations_kpoint,
         translations_kpoint) = self._symmetry.get_group_of_kpoint(kpoint)

        self.assertNotEqual(len(rotations_kpoint), 48)
        self.assertEqual(len(rotations_kpoint), 16)

    def test_110(self):
        kpoint = np.array([0.5, 0.5, 0.0])  # M: D_4h
        (rotations_kpoint,
         translations_kpoint) = self._symmetry.get_group_of_kpoint(kpoint)

        self.assertNotEqual(len(rotations_kpoint), 48)
        self.assertEqual(len(rotations_kpoint), 16)

    def test_111(self):
        kpoint = np.array([0.5, 0.5, 0.5])  # R: O_h
        (rotations_kpoint,
         translations_kpoint) = self._symmetry.get_group_of_kpoint(kpoint)

        self.assertEqual(len(rotations_kpoint), 48)


if __name__ == "__main__":
    unittest.main()
