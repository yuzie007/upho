import unittest
import numpy as np
from upho.phonon.translational_projector import TranslationalProjector


class DummyPrimitive(object):
    def get_primitive_matrix(self):
        primitive_matrix = np.array(
            [[0.25, 0.0, 0.0],
             [0.00, 1.0, 0.0],
             [0.00, 0.0, 1.0]])
        return primitive_matrix

    def get_primitive_to_supercell_map(self):
        p2s_map = [0]
        return p2s_map

class DummyAtoms(object):
    def get_scaled_positions(self):
        scaled_positions = np.array(
            [[0.00, 0.0, 0.0],
             [0.25, 0.0, 0.0],
             [0.50, 0.0, 0.0],
             [0.75, 0.0, 0.0]])
        return scaled_positions

    def get_chemical_symbols(self):
        chemical_symbols = ['H', 'H', 'H', 'H']
        return chemical_symbols

    def get_number_of_atoms(self):
        return 4


def get_eigvec_0():
    eigvec = np.array([
        0.5, 0.0, 0.0,
        0.5, 0.0, 0.0,
        0.5, 0.0, 0.0,
        0.5, 0.0, 0.0,
    ], dtype=complex).reshape(-1, 1)
    return eigvec


def get_eigvec_1():
    eigvec = np.array([
        0.5, 0.0, 0.0,
       -0.5, 0.0, 0.0,
        0.5, 0.0, 0.0,
       -0.5, 0.0, 0.0,
    ], dtype=complex).reshape(-1, 1)
    return eigvec


def get_eigvec_2():
    eigvec = np.array([
        0.5 , 0.0, 0.0,
        0.5j, 0.0, 0.0,
       -0.5 , 0.0, 0.0,
       -0.5j, 0.0, 0.0,
    ], dtype=complex).reshape(-1, 1)
    return eigvec


def get_q_0():
    q = np.array([0.0, 0.0, 0.0])
    return q


def get_q_1():
    q = np.array([2.0, 0.0, 0.0])
    return q


def get_q_2():
    q = np.array([1.0, 0.0, 0.0])
    return q


class TestTranslationalProjector(unittest.TestCase):
    """

    Bloch properties of eigenvectors are supposed to be recovered.
    """
    # TODO(ikeda): Test this test!
    def setUp(self):
        self._translational_projector = TranslationalProjector(
            DummyPrimitive(), DummyAtoms())
            # mappings, scaled_positions)
        self._prec = 1e-12

    def test_0_0(self):
        eigvec = get_eigvec_0()
        q = get_q_0()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors(eigvec, q)

        projected_eigvec_expected = np.array([
            1.0, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)

    def test_0_1(self):
        eigvec = get_eigvec_0()
        q = get_q_1()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors(eigvec, q)

        projected_eigvec_expected = np.array([
             1.0, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)
        
    def test_1_0(self):
        eigvec = get_eigvec_1()
        q = get_q_0()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors(eigvec, q)

        projected_eigvec_expected = np.array([
             0.0, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)
        
    def test_1_1(self):
        eigvec = get_eigvec_1()
        q = get_q_1()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors(eigvec, q)

        projected_eigvec_expected = np.array([
            0.0, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)
        
    def test_1_2(self):
        eigvec = get_eigvec_1()
        q = get_q_2()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors(eigvec, q)

        projected_eigvec_expected = np.array([
            0.0, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)
        
    def test_2_2(self):
        eigvec = get_eigvec_2()
        q = get_q_2()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors(eigvec, q)

        projected_eigvec_expected = np.array([
            0.0, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)

    def test_2_2_as_multidimensional_vectors(self):
        eigvec = get_eigvec_2()[None]
        q = get_q_2()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors(eigvec, q)

        projected_eigvec_expected = np.array([
            0.0, 0.0, 0.0,
        ], dtype=complex)[None, None, :]
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)


class TestTranslationalProjectorFull(unittest.TestCase):
    """

    Bloch properties of eigenvectors are supposed to be recovered.
    """
    # TODO(ikeda): Test this test!
    def setUp(self):
        self._translational_projector = TranslationalProjector(
            DummyPrimitive(), DummyAtoms())
            # mappings, scaled_positions)
        self._prec = 1e-12

    def test_0_0(self):
        eigvec = get_eigvec_0()
        q = get_q_0()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors_full(eigvec, q)

        projected_eigvec_expected = np.array([
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
            0.5, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)

    def test_0_1(self):
        eigvec = get_eigvec_0()
        q = get_q_1()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors_full(eigvec, q)

        projected_eigvec_expected = np.array([
             0.5, 0.0, 0.0,
             0.5, 0.0, 0.0,
             0.5, 0.0, 0.0,
             0.5, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)
        
    def test_1_0(self):
        eigvec = get_eigvec_1()
        q = get_q_0()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors_full(eigvec, q)

        projected_eigvec_expected = np.array([
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)
        
    def test_1_1(self):
        eigvec = get_eigvec_1()
        q = get_q_1()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors_full(eigvec, q)

        projected_eigvec_expected = np.array([
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)
        
    def test_1_2(self):
        eigvec = get_eigvec_1()
        q = get_q_2()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors_full(eigvec, q)

        projected_eigvec_expected = np.array([
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)
        
    def test_2_2(self):
        eigvec = get_eigvec_2()
        q = get_q_2()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors_full(eigvec, q)

        projected_eigvec_expected = np.array([
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ], dtype=complex).reshape(-1, 1)
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)

    def test_2_2_as_multidimensional_vectors(self):
        eigvec = get_eigvec_2()[None]
        q = get_q_2()

        translational_projector = self._translational_projector
        projected_eigvec = translational_projector.project_vectors_full(eigvec, q)

        projected_eigvec_expected = np.array([
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ], dtype=complex)[None, None, :]
        is_same = (
            np.abs(projected_eigvec - projected_eigvec_expected) < self._prec).all()
        self.assertTrue(is_same)


if __name__ == "__main__":
    unittest.main()
