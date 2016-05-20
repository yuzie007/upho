#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import unittest
import numpy as np
from phonopy.interface.vasp import read_vasp
from phonopy.structure.cells import get_primitive
from ph_unfolder.phonon.element_weights_calculator import (
    ElementWeightsCalculator)


class TestElementWeightsCalculator(unittest.TestCase):
    def test_A4(self):
        unitcell = read_vasp("poscars/POSCAR_A4_conv")
        primitive_matrix = [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
        primitive = get_primitive(unitcell, primitive_matrix)
        self.check(unitcell, primitive)

    def test_B3(self):
        unitcell = read_vasp("poscars/POSCAR_B3_conv")
        primitive_matrix = [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
        primitive = get_primitive(unitcell, primitive_matrix)
        self.check(unitcell, primitive)

    def test_L1_2(self):
        unitcell = read_vasp("poscars/POSCAR_L1_2")
        primitive_matrix = [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
        unitcell_ideal = read_vasp("poscars/POSCAR_fcc")
        primitive = get_primitive(unitcell_ideal, primitive_matrix)
        self.check(unitcell, primitive)

    def check(self, unitcell, primitive):
        ews_calculator = ElementWeightsCalculator(
            unitcell, primitive)
        print(ews_calculator.get_map_elements())
        print(ews_calculator.get_map_atoms_u2p())

        natoms_u = unitcell.get_number_of_atoms()
        ndims = 3
        nbands = 30

        vectors  = np.random.rand(1, natoms_u * ndims, nbands).astype(complex)
        vectors += np.random.rand(1, natoms_u * ndims, nbands) * 1.0j

        weights = ews_calculator.run_star(vectors)
        weights_all = np.linalg.norm(vectors, axis=1) ** 2
        weights_sum = np.sum(weights, axis=(1, 2))

        prec = 1e-9
        self.assertTrue(np.all(np.abs(weights_sum - weights_all) < prec))
        print(weights)

if __name__ == "__main__":
    unittest.main()
