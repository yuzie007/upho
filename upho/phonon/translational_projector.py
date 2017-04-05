#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np
from upho.structure.structure_analyzer import (
    StructureAnalyzer, find_lattice_vectors)
from upho.analysis.mappings_modifier import MappingsModifier


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
        primitive : Phonopy Primitive object
            Primitive w.r.t. ideal unit cell
        unitcell_ideal : Phonopy Atoms object
            Ideal (or average) unit cell
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

    def project_vectors(self, vectors, kpoint):
        """Project vectors onto kpoint

        Parameters
        ----------
        vectors : (..., natoms * ndim, nbands) array
            Vectors for SC at kpoint.  Each "column" vector is an eigenvector.
        kpoint : (ndim) array
            Reciprocal space point in fractional coordinates for SC.

        Returns
        -------
        projected_vectors : (..., natoms_primitive * ndim, nbands) array
            Projection of the given vectors.
            This is reduced into the primitive cell.
        """
        ncells = self._ncells
        ndim = self._ndim
        primitive = self._primitive

        p2s_map = primitive.get_primitive_to_supercell_map()
        indices = MappingsModifier(p2s_map).expand_mappings(ndim)

        expanded_mappings = self._expanded_mappings

        shape = list(vectors.shape)
        shape[-2] //= ncells
        projected_vectors = np.zeros(shape, dtype=vectors.dtype)

        for expanded_mapping in expanded_mappings:
            jndices = expanded_mapping.take(indices)
            projected_vectors += vectors.take(jndices, axis=-2)

        # The following intend;
        #     # Definition of projection operators
        #     projected_vectors /= ncells
        #     # Reduction into primitive
        #     projected_vectors *= np.sqrt(ncells)
        projected_vectors /= np.sqrt(ncells)

        return projected_vectors

    def project_vectors_full(self, vectors, kpoint):
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
