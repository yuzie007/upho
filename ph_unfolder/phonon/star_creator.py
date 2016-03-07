#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

__author__ = "Yuji Ikeda"

import numpy as np
from phonopy.structure.symmetry import Symmetry


class StarCreator(object):
    def __init__(self, is_overlapping=False, atoms=None):
        """
        """
        self.set_is_overlapping(is_overlapping)
        if atoms is not None:
            self.build_reciprocal_operations(atoms)

    def set_is_overlapping(self, is_overlapping):
        """

        Args:
            is_overlapping:
                If True, it allows the overlapping arms of the star.
                The number of the arms equals to that of the reciprocal
                operations.
        """
        self._is_overlapping = is_overlapping

    def build_reciprocal_operations(self, atoms, prec=1e-6):
        """

        Args:
            atoms:
                The "Atoms" object.
                Note that the given q-point (in fractional cooridnates)
                must correspond to the atoms.

        Note:
            "time_reversal_symmetry" is considered in reciprocal_operations.
        """
        symmetry = Symmetry(atoms, symprec=prec, is_symmetry=True)
        self._reciprocal_operations = symmetry.get_reciprocal_operations()

    @property
    def reciprocal_operations(self):
        return self._reciprocal_operations

    def create_star_of_k(self, k, prec=1e-6):
        """Create the star of given k.

        Args:
            k: Reciprocal space point to be checked.

        Returns:
            star_k: n x 3 array of the star of k.
        """
        star_k = []
        for r in self._reciprocal_operations:
            is_identical = False
            tmp_k = np.dot(r, k)

            # Symmetry is considered.
            if not self._is_overlapping:
                for arm in star_k:
                    if (np.abs(tmp_k - arm) < prec).all():
                        is_identical = True
                        break

            if not is_identical:
                star_k.append(tmp_k)
        star_k = np.array(star_k)
        return star_k
