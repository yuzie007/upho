#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np
from upho.analysis.mappings_modifier import MappingsModifier


class ElementWeightsCalculator(object):
    """Extract weights on elements from eigenvectors"""
    def __init__(self, unitcell, primitive):
        """

        Parameters
        ----------
        unitcell : Phonopy Atoms object
            This may have a disordered atomic configuration.
        primitive : Phonopy Primitive object
        """
        self._extract_map_elements(unitcell)
        self._extract_map_atoms_u2p(primitive)

    def _extract_map_elements(self, unitcell):
        natoms_u = unitcell.get_number_of_atoms()
        elements = unitcell.get_chemical_symbols()
        reduced_elements = sorted(set(elements), key=elements.index)

        map_elements = []
        for re in reduced_elements:
            map_elements.append(
                [i for i, v in enumerate(elements) if v == re])

        if sum(len(v) for v in map_elements) != natoms_u:
            raise ValueError("Mapping of elements is failed.")

        self._map_elements = map_elements
        self._reduced_elements = reduced_elements

    def _extract_map_atoms_u2p(self, primitive):
        p2s_map = primitive.get_primitive_to_supercell_map()
        s2p_map = primitive.get_supercell_to_primitive_map()
        natoms_u = len(s2p_map)

        map_atoms_u2p = []
        for iatom_s in p2s_map:
            map_atoms_u2p.append(
                [i for i, v in enumerate(s2p_map) if v == iatom_s])

        if sum(len(v) for v in map_atoms_u2p) != natoms_u:
            raise ValueError("Mapping of atoms_u2p is failed.")

        self._map_atoms_u2p = map_atoms_u2p

    def get_map_elements(self):
        return self._map_elements

    def get_map_atoms_u2p(self):
        return self._map_atoms_u2p

    def get_reduced_elements(self):
        return self._reduced_elements

    def get_number_of_elements(self):
        return len(self._reduced_elements)

    def run_star(self, vectors, ndims=3):
        """

        Parameters
        ----------
        vectors : (narms, natoms_u * ndims, nbands) array
        ndims : Integer
            number of dimensions of the space.

        Returns
        -------
        weights : (narms, nelements, natoms_p, nbands) array
        """
        weights = []
        for vectors_arm in vectors:
            weights_arm = self.run(vectors_arm, ndims)
            weights.append(weights_arm)
        return np.array(weights)

    def run(self, vectors, ndims=3):
        """

        Parameters
        ----------
        vectors : (natoms_u * ndims, nbands) array
        ndims : Integer
            number of dimensions of the space.

        Returns
        -------
        weights : (natoms_p, nelements, nbands) array
        """
        map_atoms_u2p = self._map_atoms_u2p
        map_elements = self._map_elements

        shape = vectors.shape
        nbands = shape[1]
        tmp = vectors.reshape(shape[0] // ndims, ndims, nbands)
        weights_atoms = np.linalg.norm(tmp, axis=1) ** 2

        shape_weights = (len(map_atoms_u2p), len(map_elements), nbands)
        weights = np.full(shape_weights, np.nan)  # Initialization

        for ip, lp in enumerate(map_atoms_u2p):
            for ie, le in enumerate(map_elements):
                indices = sorted(set(lp) & set(le))
                weights[ip, ie] = np.sum(weights_atoms[indices], axis=0)

        return weights

    def project_vectors(self, vectors, ndims=3):
        map_atoms_u2p = self._map_atoms_u2p
        map_elements = self._map_elements

        natoms_p = len(map_atoms_u2p)
        num_elements = len(map_elements)

        tmp = np.zeros_like(vectors[None, None])  # Add two dimensions
        projected_vectors = (
            np.repeat(np.repeat(tmp, natoms_p, axis=0), num_elements, axis=1))

        for ip, lp in enumerate(map_atoms_u2p):
            for ie, le in enumerate(map_elements):
                indices_tmp = sorted(set(lp) & set(le))
                indices = MappingsModifier(indices_tmp).expand_mappings(ndims)
                if len(indices) > 0:  # The element "le" exists on the sublattice.
                    projected_vectors[ip, ie, indices] = vectors[indices]

        return projected_vectors
