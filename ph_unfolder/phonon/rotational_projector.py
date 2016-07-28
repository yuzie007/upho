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
        self._atoms = atoms
        self._symmetry = UnfolderSymmetry(atoms)

    def create_standard_rotations(self, kpoint):
        """
        Create standard rotations for IR labels

        Parameters
        ----------
        kpoint : 1d array
            Reciprocal space point in fractional coordinates for PC.
            This is the representative of the star.
            IR labels are determined using this.

        TODO
        ----
        To be modified for nonsymmorphic space groups.
        """
        symmetry = self._symmetry
        rotations, translations = symmetry.get_group_of_wave_vector(kpoint)
        self._create_irreps(rotations)
        self._standard_rotations = rotations

    def _assign_characters_to_rotations(self, rotations, arm_transformation):
        irreps = self._irreps

        rotation_labels = self._assign_class_labels_to_rotations(
            rotations, arm_transformation)
        characters = irreps.assign_characters_to_rotations(rotation_labels)

        return characters

    def _assign_class_labels_to_rotations(self, rotations, arm_transformation):
        standard_rotation_labels = self._standard_rotation_labels

        modulated_standard_rotations = (
            self._modulate_standard_rotations(arm_transformation))

        rotation_labels = []
        for r in rotations:
            for i, sr in enumerate(modulated_standard_rotations):
                if np.all(r == sr):
                    rotation_labels.append(standard_rotation_labels[i])
                    break

        if len(rotation_labels) != len(standard_rotation_labels):
            raise ValueError("Rotation labels cannot be correctly assigned.")

        return rotation_labels

    def _modulate_standard_rotations(self, arm_transformation):
        standard_rotations = self._standard_rotations

        modulated_standard_rotations = np.zeros_like(standard_rotations)
        for i, r in enumerate(standard_rotations):
            modulated_standard_rotations[i] = (
                np.dot(np.dot(np.linalg.inv(arm_transformation), r), arm_transformation))

        return np.array(modulated_standard_rotations, dtype=int)

    def _create_mappings(self, rotations, translations):
        structure_analyzer = StructureAnalyzer(self._atoms)

        mappings = []
        for r, t in zip(rotations, translations):
            mapping = structure_analyzer.extract_mapping_for_symopr(r, t)
            mappings.append(mapping)
        self._mappings = np.array(mappings)
        self._mappings_modifier = MappingsModifier(mappings)

    def _invert_mappings(self):
        return self._mappings_modifier.invert_mappings()

    def _create_irreps(self, rotations):
        irreps = Irreps(rotations)
        character_table_data = irreps.get_character_table_data()

        self._ir_labels = character_table_data["ir_labels"]
        character_table = np.array(character_table_data["character_table"])
        self._ir_dimensions = character_table[:, 0]

        self._standard_rotation_labels = irreps.get_rotation_labels()

        self._irreps = irreps

    def _create_rotations_cart(self, rotations):
        rotations_cart = []
        cell = self._atoms.get_cell()
        for r in rotations:
            rotation_cart = np.dot(np.dot(cell.T, r), np.linalg.inv(cell.T))
            rotations_cart.append(rotation_cart)
        return np.array(rotations_cart)

    def project_vectors(self, vectors, kpoint, arm_transformation):
        """

        Parameters
        ----------
        vectors : Vectors without phase factors.
        kpoint : K point in fractional coordinates for SC.
        arm_transformation : 3 x 3 array
            Matrix to get "kpoint" from the representative of the star.

        TODO
        ----
        To be modified for nonsymmorphic space groups.
        """
        rotations, translations = self._symmetry.get_group_of_wave_vector(kpoint)
        print("len(rotations):", len(rotations))
        self._create_mappings(rotations, translations)

        characters = self._assign_characters_to_rotations(
            rotations, arm_transformation)

        rotations_cart = self._create_rotations_cart(rotations)

        ir_dimensions = self._ir_dimensions

        natoms = self._atoms.get_number_of_atoms()
        ndim = kpoint.shape[0]  # The number of dimensions of space
        order = rotations.shape[0]

        expanded_mappings_inv = self._mappings_modifier.expand_mappings(
            ndim, is_inverse=True)

        # scaled_positions = self._atoms.get_scaled_positions()
        # phases = np.exp(2.0j * np.pi * np.dot(scaled_positions, kpoint))
        # phases = np.repeat(phases, ndim)

        shape = (len(ir_dimensions), ) + vectors.shape
        projected_vectors = np.zeros(shape, dtype=vectors.dtype)
        for i, (r, expanded_mapping_inv) in enumerate(zip(rotations_cart, expanded_mappings_inv)):
            tmp = vectors[..., expanded_mapping_inv, :]

            tmp2 = np.zeros_like(vectors)
            for iatom in range(natoms):
                tmp2[..., (3 * iatom):(3 * (iatom + 1)), :] = np.einsum(
                    'ij,...jk->...ik', r, tmp[..., (3 * iatom):(3 * (iatom + 1)), :]
                )

            projected_vectors += (tmp2[None].T * np.conj(characters[i, :])).T

        # projected_vectors *= phases[None, :, None]
        projected_vectors = (projected_vectors.T * ir_dimensions[:]).T
        projected_vectors /= order

        return projected_vectors

    def get_ir_labels(self):
        return self._ir_labels

    def get_num_irs(self):
        return len(self._ir_labels)

    def get_pointgroup_symbol(self):
        return self._irreps.get_pointgroup_symbol()
