#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np
from ph_unfolder.analysis.mappings_modifier import MappingsModifier


class VectorsAdjuster(object):
    def __init__(self, atoms):
        """

        Parameters
        ----------
        atoms : Phonopy Atoms object
            Disordered supercell.
            Eigenvectors must correspond to the size of the "atoms".
        """
        self._scaled_positions = atoms.get_scaled_positions()
        self._masses = atoms.get_masses()

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
        vectors : (..., natoms_u * ndims, nbands) array
            Vectors which will be reduced.
            Phase factors must be removed in advance.
        primitive : Phonopy Primitive object.

        Returns
        -------
        reduced_vectors : (..., natoms_p, ndims, nbands) array
            Reduced vectors.
        """
        ndim = 3
        p2s_map = primitive.get_primitive_to_supercell_map()
        indices = MappingsModifier(p2s_map).expand_mappings(ndim)
        reduced_vectors = vectors[..., indices, :]

        # Renormalization of the reduced vectors
        relative_size = vectors.shape[-2] / reduced_vectors.shape[-2]
        reduced_vectors *= np.sqrt(relative_size)

        return reduced_vectors

    def apply_mass_weights(self, vectors, ndim=3):
        """Multiply vectors by mass weights

        Parameters
        ----------
        vectors : (ndim * natoms, nbands) array
        masses : (natoms) array

        Returns
        -------
        modified_vectors : (ndim * natoms, nbands) array
            mass weights are multiplied.
        """
        masses = self._masses
        modified_vectors = vectors / np.sqrt(np.repeat(masses, ndim))[:, None]
        return modified_vectors

    def get_masses(self):
        return self._masses
