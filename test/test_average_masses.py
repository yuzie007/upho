#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import unittest
import numpy as np
from phonopy.interface.vasp import read_vasp
from phonopy.structure.symmetry import Symmetry
from ph_unfolder.api_unfolding import calculate_average_masses


__author__ = "Yuji Ikeda"


def create_msg(list1, list2):
    msg = ''
    for x1, x2 in zip(list1, list2):
        msg += '\n{:12.6f}{:12.6f}'.format(x1, x2)
    return msg


class TestAverageMasses(unittest.TestCase):
    def setUp(self):
        unitcell_ideal = read_vasp('poscars/POSCAR_omega_ideal')
        self._symmetry = Symmetry(unitcell_ideal)
        self._prec = 1e-12

    def test_1(self):
        unitcell = read_vasp('poscars/POSCAR_omega_disordered_1')
        masses_average = calculate_average_masses(unitcell, self._symmetry)

        masses_expected = np.array([47.867, 91.224, 91.224])
        is_same = (np.abs(masses_average - masses_expected) < self._prec).all()
        msg = create_msg(masses_average, masses_expected)
        self.assertTrue(is_same, msg=msg)
        
    def test_2(self):
        unitcell = read_vasp('poscars/POSCAR_omega_disordered_2')
        masses_average = calculate_average_masses(unitcell, self._symmetry)

        masses_expected = np.array([47.867, 69.5455, 69.5455])
        is_same = (np.abs(masses_average - masses_expected) < self._prec).all()
        msg = create_msg(masses_average, masses_expected)
        self.assertTrue(is_same, msg=msg)


if __name__ == "__main__":
    unittest.main()
