#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import unittest
import numpy as np
from upho.phonon.rotational_projector import RotationalProjector
from phonopy.interface.vasp import read_vasp

__author__ = "Yuji Ikeda"


class TestRotationalProjector(unittest.TestCase):
    def setUp(self):
        self._vectors = np.random.rand(3, 100) + 1.0j * np.random.rand(3, 100)
        self._vectors = self._vectors[None]  # Tests for arbitrary number of dimensions

    def load_sc(self):
        atoms = read_vasp("tests/poscars/POSCAR_A_h")
        self._rotational_projector = RotationalProjector(atoms)

    def load_fcc(self):
        atoms = read_vasp("tests/poscars/POSCAR_fcc_prim_test")
        # atoms = read_vasp("tests/poscars/POSCAR_fcc_prim")
        self._rotational_projector = RotationalProjector(atoms)

    def test_0(self):
        self.load_sc()
        self._kpoint = np.array([0.00, 0.00, 0.00])
        self.check()

    def test_1(self):
        self.load_sc()
        self._kpoint = np.array([0.00, 0.25, 0.25])
        self.check()

    def test_0_fcc(self):
        self.load_fcc()
        self._kpoint = np.array([0.00, 0.00, 0.00])
        self.check()

    def test_1_fcc(self):
        self.load_fcc()
        self._kpoint = np.array([0.00, 0.25, 0.25])
        self.check()

    def test_2_fcc(self):
        self.load_fcc()
        self._kpoint = np.array([0.50, 0.25, 0.25])
        self.check()

    def test_3_fcc(self):
        self.load_fcc()
        self._kpoint = np.array([0.25, 0.25, 0.25])
        self.check()

    def test_5_fcc(self):
        self.load_fcc()
        self._kpoint = np.array([0.00, 0.05, 0.05])
        self.check()

    # def test_4_fcc(self):
    #     self.load_fcc()
    #     self._kpoint = np.array([0.75, 0.50, 0.25])
    #     self.check()

    def check(self):
        prec = 1e-6
        kpoint = self._kpoint
        vectors = self._vectors
        rotational_projector = self._rotational_projector
        rotational_projector.create_standard_rotations(kpoint)
        r_proj_vectors = rotational_projector.project_vectors(
            vectors, kpoint, np.eye(3, dtype=int))
        ir_labels = rotational_projector.get_ir_labels()
        print(ir_labels)

        sum_r_proj_vectors = np.sum(r_proj_vectors, axis=0)
        self.assertTrue(np.all(np.abs(sum_r_proj_vectors - vectors) < prec))


if __name__ == "__main__":
    unittest.main()
