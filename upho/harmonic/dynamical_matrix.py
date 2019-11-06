import numpy as np
from phonopy.structure.cells import get_reduced_bases
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix


class UnfolderDynamicalMatrix(DynamicalMatrix):
    """Dynamical matrix class

    When prmitive and supercell lattices are L_p and L_s, respectively,
    frame F is defined by
    L_p = dot(F, L_s), then L_s = dot(F^-1, L_p).
    where lattice matrix is defined by axies a,b,c in Cartesian:
        [ a1 a2 a3 ]
    L = [ b1 b2 b3 ]
        [ c1 c2 c3 ]

    Phase difference in primitive cell unit
    between atoms 1 and 2 in supercell is calculated by, e.g.,
    1j * dot((x_s(2) - x_s(1)), F^-1) * 2pi
    where x_s is reduced atomic coordinate in supercell unit.
    """

    def __init__(self,
                 supercell,
                 primitive,
                 force_constants,
                 decimals=None,
                 symprec=1e-5):
        self._scell = supercell
        self._pcell = primitive
        self._force_constants = np.array(force_constants,
                                         dtype='double', order='C')
        self._decimals = decimals
        self._symprec = symprec

        self._p2s_map = primitive.get_primitive_to_supercell_map()
        self._s2p_map = primitive.get_supercell_to_primitive_map()
        p2p_map = primitive.get_primitive_to_primitive_map()
        self._p2p_map = [p2p_map[self._s2p_map[i]]
                         for i in range(len(self._s2p_map))]
        self._smallest_vectors, self._multiplicity = \
            get_smallest_vectors(supercell, primitive, symprec)
        self._mass = self._pcell.get_masses()
        # Non analytical term correction
        self._nac = False


# Helper methods
def get_equivalent_smallest_vectors_np(
        atom_number_supercell,
        atom_number_primitive,
        supercell,
        primitive_lattice,
        symprec):
    distances = []
    differences = []
    reduced_bases = get_reduced_bases(supercell.get_cell(), symprec)
    positions = np.dot(supercell.get_positions(), np.linalg.inv(reduced_bases))

    # Atomic positions are confined into the lattice made of reduced bases.
    positions -= np.rint(positions)

    p_pos = positions[atom_number_primitive]
    s_pos = positions[atom_number_supercell]
    sc_range_1 = np.array([-1, 0, 1])[:, None] * np.array([1, 0, 0])[None, :]
    sc_range_2 = np.array([-1, 0, 1])[:, None] * np.array([0, 1, 0])[None, :]
    sc_range_3 = np.array([-1, 0, 1])[:, None] * np.array([0, 0, 1])[None, :]
    # The vector arrow is from the atom in primitive to
    # the atom in supercell cell plus a supercell lattice
    # point. This is related to determine the phase
    # convension when building dynamical matrix.
    differences = (s_pos
                + sc_range_1[:, None, None]
                + sc_range_2[None, :, None]
                + sc_range_3[None, None, :]
                - p_pos)
    vecs = np.dot(differences, reduced_bases)
    distances = np.linalg.norm(vecs, axis=-1)

    relative_scale = np.dot(reduced_bases, np.linalg.inv(primitive_lattice))
    minimum = np.min(distances)
    indices = np.where(np.abs(minimum - distances) < symprec)
    smallest_vectors = np.dot(differences[indices], relative_scale)
    smallest_vectors = smallest_vectors.reshape(-1, 3)

    return smallest_vectors


def get_smallest_vectors(supercell, primitive, symprec):
    """
    shortest_vectors:

      Shortest vectors from an atom in primitive cell to an atom in
      supercell in the fractional coordinates. If an atom in supercell
      is on the border centered at an atom in primitive and there are
      multiple vectors that have the same distance and different
      directions, several shortest vectors are stored. The
      multiplicity is stored in another array, "multiplicity".
      [atom_super, atom_primitive, multiple-vectors, 3]

    multiplicity:
      Number of multiple shortest vectors (third index of "shortest_vectors")
      [atom_super, atom_primitive]
    """

    p2s_map = primitive.get_primitive_to_supercell_map()
    size_super = supercell.get_number_of_atoms()
    size_prim = primitive.get_number_of_atoms()
    shortest_vectors = np.zeros((size_super, size_prim, 27, 3), dtype='double')
    multiplicity = np.zeros((size_super, size_prim), dtype='intc')

    for i in range(size_super):  # run in supercell
        for j, s_j in enumerate(p2s_map):  # run in primitive
            vectors = get_equivalent_smallest_vectors_np(i,
                                                      s_j,
                                                      supercell,
                                                      primitive.get_cell(),
                                                      symprec)
            multiplicity[i][j] = len(vectors)
            for k, elem in enumerate(vectors):
                shortest_vectors[i][j][k] = elem

    return shortest_vectors, multiplicity
