#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np
from ph_unfolder.analysis.mappings_modifier import MappingsModifier


class TranslationalProjector(object):
    """This class makes the projection of the given vectors.

    The given vectors could be both eigenvectors and displacements.
    This class can treat spaces with any numbers of dimensions.
    The number of dimensions is determined from the given k.
    """
    def __init__(self, mappings, scaled_positions, ndim=3):
        """

        Parameters
        ----------
        mappings :
        scaled_positions :
            Atomic positions in fractional coordinates for ideal unit cell.
        ndim : Integer
            The number of dimensions of space
        """
        self._ndim = ndim
        self._ncells = mappings.shape[0]
        self._create_expanded_mappings(mappings, ndim)

    def _create_expanded_mappings(self, mappings, ndim):
        mappings_modifier = MappingsModifier(mappings)
        self._expanded_mappings = mappings_modifier.expand_mappings(ndim)

    def set_scaled_positions(self, scaled_positions):
        self._scaled_positions = scaled_positions

    def project_vectors(self, vectors, kpoint):
        """
        Project vectors onto kpoint

        Parameters
        ----------
        vectors : (natoms * ndim, nbands) array
            Vectors for SC at kpoint.  Each "column" vector is an eigenvector.
        kpoint : (ndim) array
            Reciprocal space point in fractional coordinates for SC.

        Returns
        -------
        projected_vectors : (natoms * ndim, nbands) array
            Projection of the given vectors.
        """
        ncells = self._ncells
        ndim = self._ndim

        expanded_mappings = self._expanded_mappings

        projected_vectors = np.zeros_like(vectors)
        for expanded_mapping in expanded_mappings:
            projected_vectors += vectors[expanded_mapping, :]

        projected_vectors /= ncells

        return projected_vectors
