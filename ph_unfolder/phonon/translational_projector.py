#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np
from ph_unfolder.structure.structure_analyzer import (
    StructureAnalyzer, find_lattice_vectors)
from ph_unfolder.analysis.mappings_modifier import MappingsModifier


class TranslationalProjector(object):
    """This class makes the projection of the given vectors.

    The given vectors could be both eigenvectors and displacements.
    This class can treat spaces with any numbers of dimensions.
    The number of dimensions is determined from the given k.
    """
    def __init__(self, primitive, unitcell_ideal, ndim=3):
        """

        Parameters
        ----------
        mappings :
        scaled_positions :
            Atomic positions in fractional coordinates for ideal unit cell.
        ndim : Integer
            The number of dimensions of space
        """
        self._primitive = primitive
        self._unitcell_ideal = unitcell_ideal
        self._ndim = ndim

        lattice_vectors = self._create_lattice_vectors_in_sc()
        mappings = self._create_mappings(lattice_vectors)

        print("lattice_vectors:", lattice_vectors.shape)
        print(lattice_vectors)
        print("mappings:", mappings.shape)
        print(mappings)
        if np.any(mappings == -1):
            raise ValueError("Mapping is failed.")

        self._create_expanded_mappings(mappings, ndim)

        self._ncells = mappings.shape[0]

    def _create_expanded_mappings(self, mappings, ndim):
        mappings_modifier = MappingsModifier(mappings)
        self._expanded_mappings = mappings_modifier.expand_mappings(ndim)

    def _create_lattice_vectors_in_sc(self):
        """

        Returns
        -------
        lattice_vectors : (ncells, 3) array
            Lattice vectors in SC in fractional coordinates for "SC".

        TODO
        ----
        Creations of lattice_vectors and mappings should be separated.
        """
        primitive_matrix = self._primitive.get_primitive_matrix()
        supercell_matrix = np.linalg.inv(primitive_matrix)
        lattice_vectors = find_lattice_vectors(supercell_matrix)
        return lattice_vectors

    def _create_mappings(self, lattice_vectors):
        """

        Parameters
        ----------
        lattice_vectors : (ncells, 3) array
            Lattice vectors in SC in fractional coordinates for "SC".

        Returns
        -------
        mappings : (ncells, natoms) array
            Indices are for atoms after symmetry operations and
            elements are for atoms before symmetry opeerations.
        """
        structure_analyzer = StructureAnalyzer(self._unitcell_ideal)

        eye = np.eye(3, dtype=int)
        mappings = []
        for lv in lattice_vectors:
            mapping = structure_analyzer.extract_mapping_for_symopr(eye, lv)
            mappings.append(mapping)

        mappings = np.array(mappings)

        return mappings

    def project_vectors_old(self, vectors, kpoint):
        """
        Project vectors onto kpoint

        Parameters
        ----------
        vectors : (..., natoms * ndim, nbands) array
            Vectors for SC at kpoint.  Each "column" vector is an eigenvector.
        kpoint : (ndim) array
            Reciprocal space point in fractional coordinates for SC.

        Returns
        -------
        projected_vectors : (..., natoms * ndim, nbands) array
            Projection of the given vectors.
        """
        ncells = self._ncells
        ndim = self._ndim

        expanded_mappings = self._expanded_mappings

        projected_vectors = np.zeros_like(vectors)
        for expanded_mapping in expanded_mappings:
            projected_vectors += vectors.take(expanded_mapping, axis=-2)

        projected_vectors /= ncells

        return projected_vectors

    def project_vectors(self, vectors, kpoint):
        # TODO(ikeda): None should be replaced
        PO_creator = POCreator(self._expanded_mappings, None, kpoint)
        PO_creator.create_projection_operator()
        projection_operator = PO_creator.get_projection_operator()
        projected_vectors = np.einsum('ij,...jk->...ik', projection_operator, vectors)
        print(projected_vectors.shape)

        return projected_vectors


class POCreator(object):
    def __init__(self, expanded_mappings, lattice_vectors, kpoint):
        self._expanded_mappings = expanded_mappings
        self._lattice_vectors = lattice_vectors
        self._kpoint = kpoint

    def create_projection_operator(self):
        expanded_mappings = self._expanded_mappings
        lattice_vectors = self._lattice_vectors
        kpoint = self._kpoint

        noprs, nelms = expanded_mappings.shape
        projection_operator = np.zeros((nelms, nelms), dtype=complex)
        for expanded_mapping in expanded_mappings:
            for i, j in enumerate(expanded_mapping):
                projection_operator[i, j] += 1.0
        projection_operator /= noprs

        self._projection_operator = projection_operator

    def get_projection_operator(self):
        return self._projection_operator
