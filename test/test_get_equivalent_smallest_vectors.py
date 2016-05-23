#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import time
import unittest
import numpy as np
from phonopy.interface.vasp import read_vasp
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors
from ph_unfolder.harmonic.dynamical_matrix import get_equivalent_smallest_vectors_np


class TestRotationalProjector(unittest.TestCase):
    def setUp(self):
        self._atoms = read_vasp('poscars/POSCAR_fcc_2x2x2')
        self._primitive_matrix = np.array([
            [0.00, 0.25, 0.25],
            [0.25, 0.00, 0.25],
            [0.25, 0.25, 0.00],
        ])

    def test(self):
        natoms = self._atoms.get_number_of_atoms()
        symprec = 1e-6

        dt_old = 0.0
        dt_new = 0.0
        for i in range(natoms):
            for j in range(natoms):
                t0 = time.time()
                tmp0 = get_equivalent_smallest_vectors(
                    i, j, self._atoms, self._primitive_matrix, symprec)
                t1 = time.time()
                dt_old += t1 - t0

                t0 = time.time()
                tmp1 = get_equivalent_smallest_vectors_np(
                    i, j, self._atoms, self._primitive_matrix, symprec)
                t1 = time.time()
                dt_new += t1 - t0

                print(tmp0)
                print(tmp1)
                self.assertTrue(np.array_equal(tmp0, tmp1))
        print(dt_old)
        print(dt_new)

if __name__ == "__main__":
    unittest.main()
