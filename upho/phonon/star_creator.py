import numpy as np
from phonopy.structure.symmetry import Symmetry


class StarCreator:
    def __init__(self, is_overlapping=False, atoms=None, symprec=1e-6):
        """

        Parameters
        ----------
        atoms : Phonopy Atoms object
            Atoms for primitive cell.
        """
        self.set_is_overlapping(is_overlapping)
        self._atoms = atoms
        self._symprec = symprec
        self._create_symmetry()

    def set_is_overlapping(self, is_overlapping):
        """

        Parameters
        ----------
        is_overlapping : Bool
            If True, it allows the overlapping arms of the star.
            The number of the arms equals to that of the rotational
            operations.
        """
        self._is_overlapping = is_overlapping

    def _create_symmetry(self):
        symmetry = Symmetry(
            self._atoms, symprec=self._symprec, is_symmetry=True)
        self._symmetry = symmetry

    def get_rotations(self):
        return self._symmetry.get_dataset()["rotations"]

    def create_star(self, kpoint):
        """Create the star of the given kpoint

        Definition of the star of k follows that in ITB 2010 Chap. 1.5.

        Parameters
        ----------
        kpoint : Reciprocal space point

        Returns
        -------
        star : n x 3 array
            Star of the given kpoint.
        transformation_matrices : n x 3 x 3 array
            Matrices to obtain arms of the star from the given kpoint.
        """
        rotations = self._symmetry.get_dataset()["rotations"]
        lattice = self._atoms.get_cell()

        def get_dist(tmp, arm):
            diff = tmp - arm
            diff -= np.rint(diff)
            dist = np.linalg.norm(np.dot(np.linalg.inv(lattice), diff))
            return dist

        star = []
        transformation_matrices = []
        for r in rotations:
            tmp = np.dot(kpoint, r)
            if (self._is_overlapping or
                all(get_dist(tmp, arm) > self._symprec for arm in star)):
                star.append(tmp)
                transformation_matrices.append(r)

        return np.array(star), np.array(transformation_matrices)
