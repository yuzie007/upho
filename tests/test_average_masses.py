import unittest
import numpy as np
from phonopy.interface.vasp import read_vasp
from phonopy.structure.symmetry import Symmetry
from upho.api_unfolding import (
    calculate_average_masses, calculate_mass_variances)


def create_msg(list1, list2):
    msg = ''
    for x1, x2 in zip(list1, list2):
        msg += '\n{:12.6f}{:12.6f}'.format(x1, x2)
    return msg


class TestAverageMasses(unittest.TestCase):
    def setUp(self):
        unitcell_ideal = read_vasp('tests/poscars/POSCAR_omega_ideal')
        self._symmetry = Symmetry(unitcell_ideal)
        self._prec = 1e-12

    def test_1(self):
        unitcell = read_vasp('tests/poscars/POSCAR_omega_disordered_1')
        masses = unitcell.get_masses()

        masses_average = calculate_average_masses(masses, self._symmetry)
        masses_expected = np.array([47.867, 91.224, 91.224])
        is_same = (np.abs(masses_average - masses_expected) < self._prec).all()
        msg = create_msg(masses_average, masses_expected)
        self.assertTrue(is_same, msg=msg)

        mass_variances = calculate_mass_variances(masses, self._symmetry)
        mass_variances_expected = np.array([0.0, 0.0, 0.0])
        is_same = (np.abs(mass_variances - mass_variances_expected) < self._prec).all()
        msg = create_msg(mass_variances, mass_variances_expected)
        self.assertTrue(is_same, msg=msg)

    def test_2(self):
        unitcell = read_vasp('tests/poscars/POSCAR_omega_disordered_2')
        masses = unitcell.get_masses()

        masses_average = calculate_average_masses(masses, self._symmetry)
        masses_expected = np.array([47.867, 69.5455, 69.5455])
        is_same = (np.abs(masses_average - masses_expected) < self._prec).all()
        msg = create_msg(masses_average, masses_expected)
        self.assertTrue(is_same, msg=msg)

        mass_variances = calculate_mass_variances(masses, self._symmetry)
        mass_variances_expected = np.array([0.0, 469.95736225000013, 469.95736225000013])
        is_same = (np.abs(mass_variances - mass_variances_expected) < self._prec).all()
        msg = create_msg(mass_variances, mass_variances_expected)
        self.assertTrue(is_same, msg=msg)

        # mass_scattering_factor:
        # ((47.867 - 69.5455) ** 2 + (91.224 - 69.5455) ** 2) * 0.5 / (69.5455 ** 2)
        # 0.09716735699807359

if __name__ == "__main__":
    unittest.main()
