#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import unittest
from phonopy.interface.vasp import read_vasp
from upho.structure.unfolder_symmetry import UnfolderSymmetry
from upho.irreps.irreps import Irreps

__author__ = "Yuji Ikeda"


class TestIrreps(unittest.TestCase):
    def test_fcc_conv(self):
        filename = "../poscars/POSCAR_fcc"

        atoms = read_vasp(filename)
        symmetry = UnfolderSymmetry(atoms)

        rotations = symmetry.get_pointgroup_operations()
        self.check_irreps(rotations, 'm-3m')

        rotations = symmetry.get_group_of_wave_vector([0.00, 0.25, 0.25])[0]
        self.check_irreps(rotations, 'mm2')

        rotations = symmetry.get_group_of_wave_vector([0.25, 0.00, 0.25])[0]
        self.check_irreps(rotations, 'mm2')

        rotations = symmetry.get_group_of_wave_vector([0.25, 0.25, 0.00])[0]
        self.check_irreps(rotations, 'mm2')

    def test_fcc_prim(self):
        filename = "../poscars/POSCAR_fcc_prim"

        atoms = read_vasp(filename)
        symmetry = UnfolderSymmetry(atoms)

        rotations = symmetry.get_pointgroup_operations()
        self.check_irreps(rotations, 'm-3m')

        rotations = symmetry.get_group_of_wave_vector([0.00, 0.25, -0.25])[0]
        self.check_irreps(rotations, 'mm2')

        p = [0.50, 0.25, 0.75]  # W (-42m)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, '-42m')

        rotations = symmetry.get_group_of_wave_vector([0.50, 0.25, 0.25])[0]
        self.check_irreps(rotations, 'mm2')

        rotations = symmetry.get_group_of_wave_vector([0.25, 0.50, 0.25])[0]
        self.check_irreps(rotations, 'mm2')

        rotations = symmetry.get_group_of_wave_vector([0.25, 0.25, 0.50])[0]
        self.check_irreps(rotations, 'mm2')

        p = [0.5, 0.3, 0.7]  # Q (2)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, '2')

        p = [0.4, 0.4, 0.1]  # C (m)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, 'm')

        p = [0.4, 0.2, 0.1]  # GP (1)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, '1')

    def test_a3(self):
        filename = "../poscars/POSCAR_A3"

        atoms = read_vasp(filename)
        symmetry = UnfolderSymmetry(atoms)

        rotations = symmetry.get_pointgroup_operations()
        self.check_irreps(rotations, '6/mmm')

        p = [0.0, 0.0, 0.5]  # A (6/mmm)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, '6/mmm')

        p = [1.0 / 3.0, 1.0 / 3.0, 0.0]  # K (-6m2)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, '-6m2')

        p = [0.5, 0.0, 0.0]  # M (mmm)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, 'mmm')

        p = [0.5, 0.0, 0.5]  # L (mmm)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, 'mmm')

        p = [0.0, 0.0, 0.25]  # DT (6mm)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, '6mm')

    def test_a13(self):
        filename = "../poscars/POSCAR_A13"

        atoms = read_vasp(filename)
        symmetry = UnfolderSymmetry(atoms)  # 213

        rotations = symmetry.get_pointgroup_operations()
        self.check_irreps(rotations, '432')

        p = [0.0, 0.5, 0.5]  # X (422)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, '422')

    def test_b3(self):
        filename = "../poscars/POSCAR_B3_conv"

        atoms = read_vasp(filename)
        symmetry = UnfolderSymmetry(atoms)  # 216

        rotations = symmetry.get_pointgroup_operations()
        self.check_irreps(rotations, '-43m')

        p = [0.0, 0.5, 0.5]  # X (-42m)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, '-42m')

    def test_mgnh(self):
        filename = "../poscars/POSCAR_MgNH"

        atoms = read_vasp(filename)
        symmetry = UnfolderSymmetry(atoms)  # 216

        rotations = symmetry.get_pointgroup_operations()
        self.check_irreps(rotations, '6/m')

        p = [1.0 / 3.0, 1.0 / 3.0, 0.0]  # K (-6)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, '-6')

        p = [0.0, 0.0, 0.25]  # DT (6)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, '6')

        p = [1.0 / 3.0, 1.0 / 3.0, 0.25]  # P (3)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, '3')

    def test_sc(self):
        filename = "../poscars/POSCAR_Sc"

        atoms = read_vasp(filename)
        symmetry = UnfolderSymmetry(atoms)  # 178

        rotations = symmetry.get_pointgroup_operations()
        self.check_irreps(rotations, '622')

        p = [1.0 / 3.0, 1.0 / 3.0, 0.0]  # K (32)
        rotations = symmetry.get_group_of_wave_vector(p)[0]
        self.check_irreps(rotations, '32')

    def test_cl12pd6(self):
        filename = "../poscars/POSCAR_Cl12Pd6"

        atoms = read_vasp(filename)
        symmetry = UnfolderSymmetry(atoms)  # 148

        rotations = symmetry.get_pointgroup_operations()
        self.check_irreps(rotations, '-3')

    def test_thcl4(self):
        filename = "../poscars/POSCAR_ThCl4"

        atoms = read_vasp(filename)
        symmetry = UnfolderSymmetry(atoms)  # 88

        rotations = symmetry.get_pointgroup_operations()
        self.check_irreps(rotations, '4/m')

    def test_22(self):
        filename = "../poscars/POSCAR_H3S"

        atoms = read_vasp(filename)
        symmetry = UnfolderSymmetry(atoms)

        rotations = symmetry.get_pointgroup_operations()
        self.check_irreps(rotations, '222')

    def test_97(self):
        filename = "../poscars/POSCAR_NaGdCu2F8"

        atoms = read_vasp(filename)
        symmetry = UnfolderSymmetry(atoms)

        rotations = symmetry.get_pointgroup_operations()
        self.check_irreps(rotations, '422')

    def check_irreps(self, rotations, pointgroup_symbol):
        # multiplication_table = create_multiplication_table(rotations)
        # print(multiplication_table)

        # types_of_rotations = create_types_of_rotations(rotations)

        # group = Group(multiplication_table)
        # conjugacy_classes = group.get_conjugacy_classes()
        # orders_of_conjugacy_classes = group.get_orders_of_conjugacy_classes()

        # print(np.vstack((types_of_rotations, conjugacy_classes)))
        # print(orders_of_conjugacy_classes)

        if pointgroup_symbol == '222': print(rotations)
        irreps = Irreps(rotations)
        pg = irreps.get_pointgroup_symbol()

        print(pg, pointgroup_symbol)
        self.assertEqual(pg, pointgroup_symbol)


if __name__ == "__main__":
    unittest.main()
