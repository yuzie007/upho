#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

__author__ = "Yuji Ikeda"

import numpy as np
from phonopy.structure.cells import get_primitive
from ph_unfolder.phonon.star_creator import StarCreator
from ph_unfolder.phonon.translational_projector import TranslationalProjector
from ph_unfolder.phonon.rotational_projector import RotationalProjector
from ph_unfolder.phonon.vectors_adjuster import VectorsAdjuster


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
        # In this module, primitive is w.r.t. the unit cell (may be disordered).
        self._primitive = get_primitive(
            self._unitcell_ideal, primitive_matrix_ideal)

        self._build_star_creator()
        self._generate_translational_projector()
        self._generate_vectors_adjuster()
        self._create_rotational_projector()

    def _build_star_creator(self):
        if self._star == "all":
            is_overlapping = True
        else:  # "none" or "sym"
            is_overlapping = False

        primitive_ideal_wrt_unitcell = self._primitive

        self._star_creator = StarCreator(
            is_overlapping=is_overlapping,
            atoms=primitive_ideal_wrt_unitcell)

        if self._star == "none":
            self._nopr = 1
        else:  # "sym" or "all"
            self._nopr = len(self._star_creator.get_rotations())

        print("nopr:", self._nopr)

    def _generate_translational_projector(self):
        self._translational_projector = TranslationalProjector(
            self._primitive, self._unitcell_ideal)

    def _create_rotational_projector(self):
        self._rotational_projector = RotationalProjector(self._primitive)

    def _generate_vectors_adjuster(self):
        # Get the (disordered) unitcell.
        primitive = self._dynamical_matrix.get_primitive()
        scaled_positions = primitive.get_scaled_positions()
        self._vectors_adjuster = VectorsAdjuster(scaled_positions)

    def create_q_star(self, q):
        """

        Args:
            q: Reciprocal space point in fractional coordinates for "PC".

        "sym" : Duplication is not allowed.
        "all" : Duplication is allowed.
        "none": The star of k is not considered.
        """
        if self._star == "none":
            q_star, transformation_matrices = (
                np.array(q)[None, :], np.eye(3, dtype=int)[None, :, :])
        else:  # "all" or "sym"
            q_star, transformation_matrices = (
                self._star_creator.create_star(q))

        print("len(q_star):", len(q_star))
        print("q_star:")
        print(q_star)

        return q_star, transformation_matrices

    def extract_eigenstates(self, q):
        """

        Parameters
        ----------
        q : Reciprocal space point in fractional coordinates for "PC".
        """
        print("=" * 40)
        print("q:", q)
        print("=" * 40)

        rotational_projector = self._rotational_projector
        rotational_projector.create_standard_rotations(q)
        max_irs = rotational_projector.get_max_irs()
        num_irs = rotational_projector.get_num_irs()
        print("pointgroup_symbol:", self.get_pointgroup_symbol())

        ir_labels = np.zeros(max_irs, dtype='S3')
        ir_labels[:num_irs] = rotational_projector.get_ir_labels()

        q_star, transformation_matrices = self.create_q_star(q)

        nband = self._cell.get_number_of_atoms() * 3
        nopr = self._nopr

        eigvals_all = np.zeros((nopr, nband), dtype=float) * np.nan
        weights_all = np.zeros((nopr, nband), dtype=float) * np.nan
        eigvecs_all = np.zeros((nopr, nband, nband), dtype=complex) * np.nan
        rot_weights_all = np.zeros((nopr, max_irs, nband), dtype=float) * np.nan
        for i_star, (q, transformation_matrix) in enumerate(zip(q_star, transformation_matrices)):
            print("i_star:", i_star)
            print("q_pc:", q)
            (eigvals,
             eigvecs,
             weights,
             rot_weights) = self._extract_eigenstates_for_q(q, transformation_matrix)
            eigvals_all[i_star] = eigvals
            eigvecs_all[i_star] = eigvecs
            weights_all[i_star] = weights
            rot_weights_all[i_star] = rot_weights

        weights_all /= len(q_star)
        print("sum(weights_all):", np.sum(weights_all[:len(q_star)]))

        rot_weights_all = np.array(rot_weights_all) / len(q_star)

        return eigvals_all, eigvecs_all, weights_all, len(q_star), rot_weights_all, num_irs, ir_labels

    def get_pointgroup_symbol(self):
        return self._rotational_projector.get_pointgroup_symbol()

    def _extract_eigenstates_for_q(self, q_pc, transformation_matrix):
        """Extract eigenstates with their weights.

        Args:
            q_pc: Reciprocal space point in fractional coordinatees for PC.

        Returns:
            eigvals: Eigenvalues of "SC".
            eigvecs: Eigenvectors of "SC".
            weights: Weights for the phonon modes of SC on PC.
        """
        primitive_matrix = self._primitive.get_primitive_matrix()
        q_sc = get_q_sc_from_q_pc(q_pc, primitive_matrix)

        print("q_sc:", q_sc)

        self._dynamical_matrix.set_dynamical_matrix(q_sc)
        dm = self._dynamical_matrix.get_dynamical_matrix()
        eigvals, eigvecs = np.linalg.eigh(dm)

        weights, t_proj_eigvecs = self._extract_weights(q_sc, eigvecs)

        rot_weights, rot_proj_vectors = self._create_rot_projection_weights(
            q_pc, transformation_matrix, t_proj_eigvecs)

        # if __debug__:
        #     self._print_debug(eigvals, rot_weights)

        return eigvals, eigvecs, weights, rot_weights

    def _extract_weights(self, q, eigvecs):
        """Extract weights.

        Args:
            q: Reciprocal space point in fractional coordinates for SC.
            eigvecs: Eigenvectors for SC at q.

        Returns:
            weights: Weights of eigenvectors on the primitive cell at q.
        """
        translational_projector = self._translational_projector

        projected_eigvecs = translational_projector.project_vectors(
            vectors=eigvecs, kpoint=q)

        weights = np.linalg.norm(projected_eigvecs, axis=0) ** 2

        return weights, projected_eigvecs

    def _create_rot_projection_weights(self, kpoint, transformation_matrix, t_proj_vectors):
        """

        Parameters
        ----------
        kpoint : 1d array
            Reciprocal space point in fractional coordinates for PC.
        t_proj_vectors : array
            Vectors for SC after translational projection.
        """
        vectors_adjuster = self._vectors_adjuster

        # TODO(ikeda): Finally phase_factors will not be considered explicitly.
        t_proj_vectors = vectors_adjuster.reduce_vectors_to_primitive(
            t_proj_vectors, self._primitive)

        rot_proj_vectors = self._rotational_projector.project_vectors(
            t_proj_vectors, kpoint, transformation_matrix)

        max_irs = self._rotational_projector.get_max_irs()
        num_irs = self._rotational_projector.get_num_irs()

        shape = (max_irs, t_proj_vectors.shape[-1])
        rot_weights = np.zeros(shape, dtype=float) * np.nan
        rot_weights[:num_irs] = np.linalg.norm(rot_proj_vectors, axis=1) ** 2

        self.check_rotational_projected_vectors(
            rot_proj_vectors, t_proj_vectors)

        print("sum(rot_weights):", np.sum(rot_weights[:num_irs]))

        return rot_weights, rot_proj_vectors

    def check_rotational_projected_vectors(self, rot_proj_vectors, vectors):
        sum_rot_proj_vectors = np.sum(rot_proj_vectors, axis=0)
        diff = sum_rot_proj_vectors - vectors
        if np.any(np.abs(diff) > 1e-12):
            np.save("tmp_t", vectors)
            np.save("tmp_r", sum_rot_proj_vectors)
            raise ValueError("Sum of rotationally projected vectors is not "
                             "equal to original vectors.")

    def _print_debug(self, eigvals, rot_weights):
        ir_labels = self._rotational_projector.get_ir_labels()
        num_irs = len(ir_labels)
        print(" " * 14, end="")
        print("".join("{:<12s}".format(v) for v in ir_labels))
        for i, values in enumerate(rot_weights.T):
            print("{:12.6f}  ".format(eigvals[i]), end="")
            print("".join("{:12.6f}".format(v) for v in values[:num_irs]), end="")
            print()


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
