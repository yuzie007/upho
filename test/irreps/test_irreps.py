#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import unittest
import numpy as np
from phonopy.interface.vasp import read_vasp
from phonopy.structure.symmetry import Symmetry
from upho.structure.unfolder_symmetry import UnfolderSymmetry
from upho.irreps.irreps import Irreps


class TestIrreps(unittest.TestCase):
    def test_fcc_conv(self):
        # np.set_printoptions(threshold=2304, linewidth=145)  # 48 * 48
        filename = "../poscars/POSCAR_fcc"
    
        atoms = read_vasp(filename)
        symmetry = UnfolderSymmetry(atoms)
    
        rotations = symmetry.get_pointgroup_operations()
        check_irreps(rotations)
    
        rotations = symmetry.get_group_of_wave_vector([0.00, 0.25, 0.25])[0]
        check_irreps(rotations)
    
        rotations = symmetry.get_group_of_wave_vector([0.25, 0.00, 0.25])[0]
        check_irreps(rotations)
    
        rotations = symmetry.get_group_of_wave_vector([0.25, 0.25, 0.00])[0]
        check_irreps(rotations)

    def test_fcc_prim(self):
        filename = "../poscars/POSCAR_fcc_prim"
    
        atoms = read_vasp(filename)
        symmetry = UnfolderSymmetry(atoms)
    
        rotations = symmetry.get_pointgroup_operations()
        check_irreps(rotations)
    
        rotations = symmetry.get_group_of_wave_vector([0.00, 0.25,-0.25])[0]
        check_irreps(rotations)
    
        rotations = symmetry.get_group_of_wave_vector([0.50, 0.25, 0.25])[0]
        check_irreps(rotations)
    
        rotations = symmetry.get_group_of_wave_vector([0.25, 0.50, 0.25])[0]
        check_irreps(rotations)
    
        rotations = symmetry.get_group_of_wave_vector([0.25, 0.25, 0.50])[0]
        check_irreps(rotations)


def check_irreps(rotations):
    # multiplication_table = create_multiplication_table(rotations)
    # print(multiplication_table)

    # types_of_rotations = create_types_of_rotations(rotations)

    # group = Group(multiplication_table)
    # conjugacy_classes = group.get_conjugacy_classes()
    # orders_of_conjugacy_classes = group.get_orders_of_conjugacy_classes()

    # print(np.vstack((types_of_rotations, conjugacy_classes)))
    # print(orders_of_conjugacy_classes)

    irreps = Irreps(rotations)

    print(irreps.get_pointgroup_symbol())
    print(irreps.get_transformation_matrix())
    print(irreps.get_rotation_labels())
    print(irreps.get_characters())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename",
                        default="POSCAR",
                        type=str,
                        help="POSCAR filename.")
    args = parser.parse_args()

    run(filename=args.filename)


if __name__ == "__main__":
    unittest.main()
