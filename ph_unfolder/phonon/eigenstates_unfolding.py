#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

__author__ = "Yuji Ikeda"

import numpy as np
from phonopy.structure.cells import get_primitive
from ph_unfolder.phonon.star_creator import StarCreator
from ph_unfolder.phonon.vectors_projector import VectorsProjector
from ph_unfolder.structure.structure_analyzer import (
    StructureAnalyzer, find_lattice_vectors)


class EigenstatesUnfolding(object):
    def __init__(self,
                 dynamical_matrix,
                 unitcell_ideal,
                 primitive_matrix_ideal,
                 star="none",
                 mode="eigenvector",
                 verbose=False):
        self._verbose = verbose
        self._mode = mode

        self._cell = dynamical_matrix.get_primitive()  # Disordered
        self._dynamical_matrix = dynamical_matrix

        self._star = star
        self._unitcell_ideal = unitcell_ideal
        self._primitive_matrix_ideal = primitive_matrix_ideal

        self._build_star_creator()
        self._generate_vectors_projector()
        self._generate_bloch_recoverer()

    def _build_star_creator(self):
        if self._star == "all":
            is_overlapping = True
        else:  # "none" or "sym"
            is_overlapping = False

        primitive_ideal_wrt_unitcell = (
            get_primitive(self._unitcell_ideal, self._primitive_matrix_ideal))

        self._star_creator = StarCreator(
            is_overlapping=is_overlapping,
            atoms=primitive_ideal_wrt_unitcell)

        if self._star == "none":
            self._nopr = 1
        else:  # "sym" or "all"
            self._nopr = len(self._star_creator.reciprocal_operations)

        print("nopr:", self._nopr)

    def _generate_vectors_projector(self):
        lattice_vectors, mappings = self._generate_lattice_vectors_in_sc()
        print("lattice_vectors:", lattice_vectors.shape)
        print(lattice_vectors)
        print("mappings:", mappings.shape)
        print(mappings)
        if np.any(mappings == -1):
            raise ValueError("Mapping is failed.")
        scaled_positions = self._unitcell_ideal.get_scaled_positions()
        self._vectors_projector = VectorsProjector(mappings, scaled_positions)

    def _generate_bloch_recoverer(self):
        # Get the (disordered) unitcell.
        primitive = self._dynamical_matrix.get_primitive()
        scaled_positions = primitive.get_scaled_positions()
        self._bloch_recoverer = BlochRecoverer(scaled_positions)

    def create_q_star(self, q):
        """

        Args:
            q: Reciprocal space point in fractional coordinates for "PC".

        "sym" : Duplication is not allowed.
        "all" : Duplication is allowed.
        "none": The star of k is not considered.
        """
        if self._star == "none":
            q_star = np.array([q])
        else:  # "all" or "sym"
            q_star = self._star_creator.create_star_of_k(q)

        print("len(q_star):", len(q_star))
        print("q_star:")
        print(q_star)

        return q_star

    def _generate_lattice_vectors_in_sc(self):
        supercell_matrix = np.linalg.inv(self._primitive_matrix_ideal)
        # lattice_vectors: Fractional coordinates for "SC".
        lattice_vectors = find_lattice_vectors(supercell_matrix)

        structure_analyzer = StructureAnalyzer(self._unitcell_ideal)

        eye = np.eye(3, dtype=int)
        mappings = []
        for lv in lattice_vectors:
            mapping, diff_positions = (
                structure_analyzer.extract_mapping_for_symopr(eye, lv))
            mappings.append(mapping)

        mappings = np.array(mappings)

        return lattice_vectors, mappings

    def extract_eigenstates(self, q):
        """

        Args:
            q: Reciprocal space point in fractional coordinates for "PC".
        """
        print("=" * 40)
        print("q:", q)
        print("=" * 40)

        q_star = self.create_q_star(q)

        nband = self._cell.get_number_of_atoms() * 3
        nopr = self._nopr

        eigvals_all = np.zeros((nopr, nband), dtype=float) * np.nan
        weights_all = np.zeros((nopr, nband), dtype=float) * np.nan
        eigvecs_all = np.zeros((nopr, nband, nband), dtype=complex) * np.nan
        for i_star, q in enumerate(q_star):
            print("i_star:", i_star)
            print("q_pc:", q)
            eigvals, eigvecs, weights = self._extract_eigenstates_for_q(q)
            eigvals_all[i_star] = eigvals
            eigvecs_all[i_star] = eigvecs
            weights_all[i_star] = weights

        weights_all /= len(q_star)
        print("sum(weights_all):", np.sum(weights_all[:len(q_star)]))

        return eigvals_all, eigvecs_all, weights_all, len(q_star)

    def _extract_eigenstates_for_q(self, q_pc):
        """Extract eigenstates with their weights.

        Args:
            q_pc: Reciprocal space point in fractional coordinatees for PC.

        Returns:
            eigvals: Eigenvalues of "SC".
            eigvecs: Eigenvectors of "SC".
            weights: Weights for the phonon modes of SC on PC.
        """
        q_sc = get_q_sc_from_q_pc(q_pc, self._primitive_matrix_ideal)

        print("q_sc:", q_sc)

        self._dynamical_matrix.set_dynamical_matrix(q_sc)
        dm = self._dynamical_matrix.get_dynamical_matrix()
        eigvals, eigvecs = np.linalg.eigh(dm)

        weights = self._extract_weights(q_sc, eigvecs)

        return eigvals, eigvecs, weights

    def _extract_weights(self, q, eigvecs):
        """Extract weights.

        Args:
            q: Reciprocal space point in fractional coordinates for SC.
            eigvecs: Eigenvectors for SC at q.

        Returns:
            weights: Weights of eigenvectors on the primitive cell at q.
        """
        bloch_recoverer = self._bloch_recoverer
        vectors_projector = self._vectors_projector

        bloch_recoverer.set_q(q)

        recovered_eigvecs = bloch_recoverer.recover_Bloch(eigvecs)

        projected_eigvecs = vectors_projector.project_vectors_onto_k(
            vecs=recovered_eigvecs, k=q)

        weights = np.sum(
            np.conj(recovered_eigvecs) * projected_eigvecs, axis=0
        ).real
        return weights


class BlochRecoverer(object):
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


def get_displacements_from_eigvecs(eigvecs, supercell, q):
    """

    Args:
        eigvecs: Eigenvectors expanded to the size of the supercell.
        q: Fractional positions in reciprocal space (supercell).
    """
    displacements = np.zeros_like(eigvecs)
    masses = supercell.get_masses()
    scaled_positions = supercell.get_scaled_positions()
    for i, e in enumerate(eigvecs):
        iatom = i // 3
        p = scaled_positions[iatom]
        m = masses[iatom]
        phase = np.exp(2.0j * np.pi * np.dot(p, q))
        displacements[i] = e / np.sqrt(m) * phase
    return displacements


def get_q_sc_from_q_pc(q_pc, primitive_matrix):
    q_sc = np.dot(q_pc, np.linalg.inv(primitive_matrix))
    # For the current implementation, we should "not" wrap q_sc into the cell
    # with 0 <= q_sc_{1,2,3} < 1, so the next line is commented out.
    # q_sc -= np.rint(q_sc)
    return q_sc
