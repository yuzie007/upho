#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import unittest
import numpy as np
from phonopy.structure.cells import get_primitive
from phonopy.interface.vasp import read_vasp
from ph_unfolder.phonon.vectors_adjuster import VectorsAdjuster


class TestVectorsAdjuster(unittest.TestCase):
    def setUp(self):
        scaled_positions = np.array(
            [[0.00, 0.0, 0.0],
             [0.25, 0.0, 0.0],
             [0.50, 0.0, 0.0],
             [0.75, 0.0, 0.0]])
        self._vectors_adjuster = VectorsAdjuster(scaled_positions)
        self._prec = 1e-12

    def test_0_0(self):
        eigvecs = self.get_eigvec_0().reshape((-1, 1))
        q = self.get_q_0()

        vectors_adjuster = self._vectors_adjuster
        vectors_adjuster.set_q(q)

        recovered_vecs = vectors_adjuster.recover_Bloch(eigvecs)

        recovered_vecs_expected = np.array([
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(recovered_vecs - recovered_vecs_expected) < self._prec).all()
        self.assertTrue(is_same)
        
    def test_0_1(self):
        eigvecs = self.get_eigvec_0().reshape((-1, 1))
        q = self.get_q_1()

        vectors_adjuster = self._vectors_adjuster
        vectors_adjuster.set_q(q)

        print(eigvecs)
        recovered_vecs = vectors_adjuster.recover_Bloch(eigvecs)
        print(recovered_vecs)

        recovered_vecs_expected = np.array([
            0.5, 0.0, 0.0,
           -0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
           -0.5, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(recovered_vecs - recovered_vecs_expected) < self._prec).all()
        self.assertTrue(is_same)
        
    def test_1_1(self):
        eigvecs = self.get_eigvec_1().reshape((-1, 1))
        q = self.get_q_1()

        vectors_adjuster = self._vectors_adjuster
        vectors_adjuster.set_q(q)

        recovered_vecs = vectors_adjuster.recover_Bloch(eigvecs)
        print(recovered_vecs)

        recovered_vecs_expected = np.array([
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(recovered_vecs - recovered_vecs_expected) < self._prec).all()
        self.assertTrue(is_same)

    def test_reduce_vectors_to_primitive(self):
        vectors_adjuster = self._vectors_adjuster
        atoms = read_vasp("poscars/POSCAR_fcc")
        primitive_matrix = [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
        primitive = get_primitive(atoms, primitive_matrix)
        vectors = self.get_eigvec_0()[:, None]
        reduced_vectors = vectors_adjuster.reduce_vectors_to_primitive(
            vectors, primitive)
        nexpansion = 4
        exp = np.array([
            [0.5],
            [0.0],
            [0.0],
        ])
        exp *= np.sqrt(nexpansion)
        self.assertTrue(np.all(np.abs(reduced_vectors - exp) < 1e-6))

    def get_eigvec_0(self):
        eigvec = np.array([
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
        ], dtype=complex)
        return eigvec
    
    def get_eigvec_1(self):
        eigvec = np.array([
            0.5, 0.0, 0.0,
           -0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
           -0.5, 0.0, 0.0,
        ], dtype=complex)
        return eigvec
    
    def get_eigvec_2(self):
        eigvec = np.array([
            0.5 , 0.0, 0.0,
            0.5j, 0.0, 0.0,
           -0.5 , 0.0, 0.0,
           -0.5j, 0.0, 0.0,
        ], dtype=complex)
        return eigvec
    
    def get_q_0(self):
        q = np.array([0.0, 0.0, 0.0])
        return q

    def get_q_1(self):
        q = np.array([2.0, 0.0, 0.0])
        return q

    def get_q_2(self):
        q = np.array([1.0, 0.0, 0.0])
        return q


if __name__ == "__main__":
    unittest.main()
