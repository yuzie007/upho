#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np


class VectorsAdjuster(object):
    def __init__(self, scaled_positions):
        self.set_scaled_positions(scaled_positions)

    def set_scaled_positions(self, scaled_positions):
        """

        Args:
            scaled_positions:
                Scaled positions for the (disordered) cell.
        """
        self._scaled_positions = scaled_positions

    def set_q(self, q):
        """

        Args:
            q: Reciprocal space point in fractional coordinates for SC.
        """
        self._q = q

    def recover_Bloch(self, vecs):
        """Recorver the properties of Bloch's waves.

        Args:
            vecs: Vectors to be recovered.

        Returns:
            recovered_vecs: Vectors having the properties of Bloch's waves.
        """
        recovered_vecs = np.zeros_like(vecs) * np.nan
        for i, vec in enumerate(vecs):
            iatom = i // 3
            p = self._scaled_positions[iatom]
            phase = np.exp(2.0j * np.pi * np.dot(p, self._q))
            recovered_vecs[i] = vec * phase
        return recovered_vecs

    def remove_phase_factors(self, vectors, kpoint):
        """
        Remove phase factors from given vectors.

        Parameters
        ----------
        vectors : array
            Vectors whose phase factors are removed.
        kpiont :
            Reciprocal space point in fractional coordinates for SC.
        """
        phases = np.exp(-2.0j * np.pi * np.dot(self._scaled_positions, kpoint))
        phases = np.repeat(phases, 3)
        modified_vectors = phases[:, None] * vectors
        return modified_vectors

    def reduce_vectors_to_primitive(self, vectors, primitive):
        """
        Reduce size of vectors to primitive.

        Parameters
        ----------
        vectors : array
            Vectors which will be reduced.
            Phase factors must be removed in advance.
        primitive : Phonopy Primitive object.

        Returns
        -------
        reduced_vectors : array
            Reduced vectors.
        """
        ndim = 3
        p2s_map = primitive.get_primitive_to_supercell_map()
        num_atoms = vectors.shape[0] // ndim
        tmp = vectors.reshape(num_atoms, ndim, -1)
        tmp = tmp[p2s_map, :, :]
        reduced_vectors = tmp.reshape(ndim * len(p2s_map), -1)

        # Renormalization of the reduced vectors
        relative_size = num_atoms / len(p2s_map)
        reduced_vectors *= np.sqrt(relative_size)

        return reduced_vectors
