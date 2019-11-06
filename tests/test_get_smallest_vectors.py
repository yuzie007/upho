import time
import unittest
import numpy as np
from phonopy.interface.vasp import read_vasp
from phonopy.structure.cells import _get_smallest_vectors, get_primitive
from upho.harmonic.dynamical_matrix import (
    get_smallest_vectors as get_smallest_vectors_upho)


class TestRotationalProjector(unittest.TestCase):
    def setUp(self):
        self._atoms = read_vasp('tests/poscars/POSCAR_fcc_2x2x2')
        self._primitive_matrix = np.array([
            [0.00, 0.25, 0.25],
            [0.25, 0.00, 0.25],
            [0.25, 0.25, 0.00],
        ])

    def test(self):
        natoms = self._atoms.get_number_of_atoms()
        primitive = get_primitive(self._atoms, self._primitive_matrix)
        symprec = 1e-6

        smallest_vectors0, multiplicity0 = _get_smallest_vectors(
            self._atoms, primitive, symprec)
        smallest_vectors1, multiplicity1 = get_smallest_vectors_upho(
            self._atoms, primitive, symprec)

        dt_old = 0.0
        dt_new = 0.0
        for i in range(natoms):
            for j in range(primitive.get_number_of_atoms()):
                t0 = time.time()
                tmp0 = smallest_vectors0[i, j, :multiplicity0[i][j]]
                t1 = time.time()
                dt_old += t1 - t0

                t0 = time.time()
                tmp1 = smallest_vectors1[i, j, :multiplicity1[i][j]]
                t1 = time.time()
                dt_new += t1 - t0

                print(tmp0)
                print(tmp1)
                self.assertTrue(np.array_equal(tmp0, tmp1))
        print(dt_old)
        print(dt_new)


if __name__ == "__main__":
    unittest.main()
