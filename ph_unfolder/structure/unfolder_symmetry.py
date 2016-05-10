#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np
from phonopy.structure.symmetry import Symmetry, get_pointgroup


class UnfolderSymmetry(Symmetry):
    def create_little_group(self, kpoint):
        rotations = self._symmetry_operations["rotations"]
        translations = self._symmetry_operations["translations"]
        lattice = self._cell.get_cell()

        rotations_kpoint = []
        translations_kpoint = []
        for r, t in zip(rotations, translations):
            diff = np.dot(kpoint, r) - kpoint
            diff -= np.rint(diff)
            dist = np.linalg.norm(np.dot(np.linalg.inv(lattice), diff))
            if dist < self._symprec:
                rotations_kpoint.append(r)
                translations_kpoint.append(t)

        return np.array(rotations_kpoint), np.array(translations_kpoint)

    get_group_of_wave_vector = create_little_group

    def create_star(self, kpoint):
        """
        Create the star of the given kpoint

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
        rotations = self._symmetry_operations["rotations"]
        lattice = self._cell.get_cell()

        def get_dist(tmp, arm):
            diff = tmp - arm
            # diff -= np.rint(diff)  # TODO(ikeda): Check definition.
            dist = np.linalg.norm(np.dot(np.linalg.inv(lattice), diff))
            return dist

        star = []
        transformation_matrices = []
        for r in rotations:
            tmp = np.dot(kpoint, r)
            if all(get_dist(tmp, arm) > self._symprec for arm in star):
                star.append(tmp)
                transformation_matrices.append(r)

        return np.array(star), np.array(transformation_matrices)
