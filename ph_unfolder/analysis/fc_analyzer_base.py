#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from phonopy.file_IO import write_FORCE_CONSTANTS
from phonopy.structure.cells import Supercell
from phonopy.harmonic.force_constants import symmetrize_force_constants

__author__ = "Yuji Ikeda"


class FCAnalyzerBase(object):
    def __init__(self,
                 force_constants=None,
                 atoms=None,
                 atoms_ideal=None,
                 supercell_matrix=None,
                 is_symmetrized=True):
        """

        Parameters
        ----------
        force_constants: (natoms, natoms, 3, 3) array
        atoms: The "Atoms" object
            This is used to extract chemical symbols.
        atoms_ideal: The "Atoms" object
            This is used to judge the expected crystallographic symmetry.
        supercell_matrix: (3, 3) array
        """
        if supercell_matrix is None:
            supercell_matrix = np.eye(3)

        print("supercell_matrix:")
        print(supercell_matrix)

        self.set_force_constants(force_constants)

        if atoms is not None:
            self.set_atoms(Supercell(atoms, supercell_matrix))
        if atoms_ideal is not None:
            self.set_atoms_ideal(Supercell(atoms_ideal, supercell_matrix))

        if is_symmetrized:
            self.symmetrize_force_constants()

        self._fc_distribution_analyzer = None

        self.check_consistency()

    def check_consistency(self):
        number_of_atoms = self.get_atoms().get_number_of_atoms()
        number_of_atoms_fc = self.get_force_constants().shape[0]
        if number_of_atoms != number_of_atoms_fc:
            print(number_of_atoms, number_of_atoms_fc)
            raise ValueError("Atoms, Dim, and FC are not consistent.")

    def symmetrize_force_constants(self, iteration=3):
        symmetrize_force_constants(self._force_constants, iteration)
        return self

    def set_force_constants(self, force_constants):
        self._force_constants = force_constants

    def get_force_constants(self):
        return self._force_constants

    def set_atoms(self, atoms):
        self._atoms = atoms

    def get_atoms(self):
        return self._atoms

    def set_atoms_ideal(self, atoms_ideal):
        self._atoms_ideal = atoms_ideal

    def get_atoms_ideal(self):
        return self._atoms_ideal

    def write_force_constants(self, filename_write):
        write_FORCE_CONSTANTS(self._force_constants, filename_write)
