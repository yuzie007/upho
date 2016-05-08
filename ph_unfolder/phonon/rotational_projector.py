#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np
from ph_unfolder.structure.structure_analyzer import StructureAnalyzer
from ph_unfolder.analysis.mappings_modifier import MappingsModifier
from ph_unfolder.irreps.irreps import Irreps
from ph_unfolder.structure.unfolder_symmetry import UnfolderSymmetry


class RotationalProjector(object):
    def __init__(self, atoms):
        """
        Decomposer of vectors according to rotations

        Parameters
        ----------
            atoms : Phonopy Atoms object.
        """
        self._scaled_positions = atoms.get_scaled_positions()
        self._atoms = atoms
        self._symmetry = UnfolderSymmetry(atoms)

    def _create_mappings(self, rotations, translations):
        structure_analyzer = StructureAnalyzer(self._atoms)

        mappings = []
        for r, t in zip(rotations, translations):
            mapping = structure_analyzer.extract_mapping_for_symopr(r, t)[0]
            mappings.append(mapping)
        self._mappings = np.array(mappings)
        self._mappings_modifier = MappingsModifier(mappings)

    def _invert_mappings(self):
        return self._mappings_modifier.invert_mappings()

    def _create_characters(self, rotations):
        irreps = Irreps(rotations)
        print("pointgroup_symbol:", irreps.get_pointgroup_symbol())
        character_table_data = irreps.get_character_table_data()

        self._characters = irreps.get_characters()
        self._ir_labels = character_table_data["ir_labels"]
        character_table = np.array(character_table_data["character_table"])
        self._ir_dimensions = character_table[:, 0]

    def _create_rotations_cart(self, rotations):
        rotations_cart = []
        cell = self._atoms.get_cell()
        for r in rotations:
            rotation_cart = np.dot(np.dot(cell.T, r), np.linalg.inv(cell.T))
            rotations_cart.append(rotation_cart)
        return np.array(rotations_cart)

    def project_vectors(self, vectors, kpoint):
        """

        Parameters
        ----------
        vectors : Vectors without phase factors.
        kpoint : K point in fractional coordinates for SC.
        """
        rotations, translations = self._symmetry.get_group_of_wave_vector(kpoint)
        print("len(rotations):", len(rotations))
        self._create_mappings(rotations, translations)
        # TODO(ikeda): To be modified for nonsymmorphic space groups.
        self._create_characters(rotations)

        rotations_cart = self._create_rotations_cart(rotations)

        ir_dimensions = self._ir_dimensions
        characters = self._characters
        scaled_positions = self._atoms.get_scaled_positions()

        natoms = self._atoms.get_number_of_atoms()
        ndim = kpoint.shape[0]  # The number of dimensions of space
        order = rotations.shape[0]

        expanded_mappings_inv = self._mappings_modifier.expand_mappings(
            ndim, is_inverse=True)

        phases = np.exp(2.0j * np.pi * np.dot(scaled_positions, kpoint))
        phases = np.repeat(phases, ndim)

        shape = (len(ir_dimensions), ) + vectors.shape
        projected_vectors = np.zeros(shape, dtype=vectors.dtype)
        for i, (r, expanded_mapping_inv) in enumerate(zip(rotations_cart, expanded_mappings_inv)):
            tmp = vectors[expanded_mapping_inv]

            tmp2 = np.zeros_like(vectors)
            for iatom in range(natoms):
                tmp2[(3 * iatom):(3 * (iatom + 1))] = np.dot(
                    r, tmp[(3 * iatom):(3 * (iatom + 1))])

            projected_vectors += np.conj(characters[i, :, None, None]) * tmp2[None, :, :]

        projected_vectors *= phases[None, :, None]
        projected_vectors *= ir_dimensions[:, None, None]
        projected_vectors /= order

        return projected_vectors, self._ir_labels
