#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

__author__ = "Yuji Ikeda"

import numpy as np


class VectorsProjector(object):
    """This class makes the projection of the given vectors.

    The given vectors could be both eigenvectors and displacements.
    This class can treat spaces with any numbers of dimensions.
    The number of dimensions is determined from the given k.
    """
    def __init__(self, mappings, scaled_positions):
        """

        Args:
            mappings:
            scaled_positions:
                Atomic positions in fractional coordinates for ideal unit cell.
        """
        self.set_mappings(mappings)
        self.set_scaled_positions(scaled_positions)

    def set_mappings(self, mappings):
        self._mappings = mappings

    def set_scaled_positions(self, scaled_positions):
        self._scaled_positions = scaled_positions

    def project_vectors_onto_k(self, vecs, k):
        """Project vectors onto k.

        Args:
            vecs: Vectors for SC at k. Each "column" vector is an eigenvector.
            k: Reciprocal space point in fractional coordinates for SC.

        Returns:
            projected_vecs: Projection of the given vectors.
        """
        scaled_positions_old = self._scaled_positions

        ndim = k.shape[0]  # The number of dimensions of space
        ne, nb = vecs.shape
        shape = (ne // ndim, ndim, nb)

        projected_vecs = np.zeros(shape, dtype=vecs.dtype)
        for mapping in self._mappings:
            scaled_positions_new = scaled_positions_old[mapping]
            diff = scaled_positions_new - scaled_positions_old
            phases_sc = np.exp(-2.0j * np.pi * np.dot(diff, k))

            tmp = phases_sc[:, None, None] * vecs.reshape(shape)[mapping, :, :]
            projected_vecs += tmp

        projected_vecs /= self._mappings.shape[0]
        projected_vecs = projected_vecs.reshape(vecs.shape)

        return projected_vecs
