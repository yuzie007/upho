#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
from phonopy.structure.cells import get_primitive
from phonopy.units import VaspToTHz
from ph_unfolder.phonon.star_creator import StarCreator
from ph_unfolder.phonon.translational_projector import TranslationalProjector
from ph_unfolder.phonon.rotational_projector import RotationalProjector
from ph_unfolder.phonon.vectors_adjuster import VectorsAdjuster
from ph_unfolder.phonon.element_weights_calculator import (
    ElementWeightsCalculator)
from ph_unfolder.analysis.time_measurer import TimeMeasurer


__author__ = "Yuji Ikeda"


class Eigenstates(object):
    def __init__(self,
                 dynamical_matrix,
                 unitcell_ideal,
                 primitive_matrix_ideal,
                 star="none",
                 mode="eigenvector",
                 factor=VaspToTHz,
                 verbose=False):
        self._verbose = verbose
        self._mode = mode

        self._factor = factor

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
        self._build_element_weights_calculator()

    def _build_element_weights_calculator(self):
        unitcell_orig   = self._cell
        primitive_ideal = self._primitive
        self._element_weights_calculator = ElementWeightsCalculator(
            unitcell_orig, primitive_ideal)

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
        self._vectors_adjuster = VectorsAdjuster(primitive)

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

        Notes
        -----
        weights_arms : dictionary
            'total' : (num_arms, nbands)
            'SR'    : (num_arms, num_irreps, nbands)
            'E1'    : (num_arms, natoms_p, nelements, nbands)
            'SR_E1' : (num_arms, num_irreps, natoms_p, nelements, natoms_p, nelements, nbands)
        """
        print("=" * 40)
        print("q:", q)
        print("=" * 40)

        rotational_projector = self._rotational_projector
        rotational_projector.create_standard_rotations(q)
        print("pointgroup_symbol:", self.get_pointgroup_symbol())

        q_star, transformation_matrices = self.create_q_star(q)

        eigvals_arms = []  # (num_arms, nbands)

        weights_arms = {}
        weights_keys = ['total', 'SR', 'E1', 'SR_E1', 'E2']
        for k in weights_keys:
            weights_arms[k] = []
        for i_star, (q, transformation_matrix) in enumerate(zip(q_star, transformation_matrices)):
            print("i_star:", i_star)
            print("q_pc:", q)
            eigvals, eigvecs, weights = self._extract_eigenstates_for_q(
                q, transformation_matrix)

            eigvals_arms.append(eigvals)
            for k in weights_keys:
                weights_arms[k].append(weights[k])

        frequencies_arms = calculate_frequencies(np.array(eigvals_arms), self._factor)

        for k in weights_keys:
            weights_arms[k] = np.array(weights_arms[k]) / len(q_star)

        print("Sum of trans_weights_arms  :", np.nansum(weights_arms['total']))
        print("Sum of rot_weights_arms    :", np.nansum(weights_arms['SR'   ]))
        print("Sum of element_weights_arms:", np.nansum(weights_arms['E1'   ]))
        print("Sum of rot_elm_weights_arms:", np.nansum(weights_arms['SR_E1']))
        print("Sum of weights_arms E2     :", np.nansum(weights_arms['E2'   ]))

        self._q_star = q_star
        self._point = q

        self._frequencies_arms     = frequencies_arms
        self._weights_arms = weights_arms

    def get_point(self):
        return self._point

    def get_frequencies_arms(self):
        return self._frequencies_arms

    def get_narms(self):
        return len(self._q_star)

    def get_pointgroup_symbol(self):
        return np.array(
            self._rotational_projector.get_pointgroup_symbol(), dtype='S')

    # def get_ir_labels(self):
    #     rotational_projector = self._rotational_projector

    #     ir_labels = [''] * MAX_IRREPS
    #     for i, l in enumerate(rotational_projector.get_ir_labels()):
    #         ir_labels[i] = l

    #     return ir_labels

    def get_ir_labels(self):
        return np.array(
            self._rotational_projector.get_ir_labels(), dtype='S')

    def get_num_irreps(self):
        return self._rotational_projector.get_num_irs()

    def _extract_eigenstates_for_q(self, q_pc, transformation_matrix):
        """Extract eigenstates with their weights.

        Parameters
        ----------
        q_pc : Reciprocal space point in fractional coordinatees for PC.
        transformation_matrix

        Returns
        -------
        eigvals : Eigenvalues of "SC".
        eigvecs : Eigenvectors of "SC".
        weights : Weights for the phonon modes of SC on PC.
            'E1' : (natoms_p, nelms, natoms_p, nelms, nbands) complex array
        """
        primitive_matrix = self._primitive.get_primitive_matrix()
        q_sc = get_q_sc_from_q_pc(q_pc, primitive_matrix)

        print("q_sc:", q_sc)

        self._dynamical_matrix.set_dynamical_matrix(q_sc)
        dm = self._dynamical_matrix.get_dynamical_matrix()
        with TimeMeasurer('Solve eigenproblem'):
            eigvals, eigvecs = np.linalg.eigh(dm)

        weights = {}

        with TimeMeasurer('Calculate weights for wavevectors'):
            weights['total'], t_proj_eigvecs = self._extract_weights(q_sc, eigvecs)

        weights['SR'], rot_proj_vectors = self._create_rot_projection_weights(
            q_pc, transformation_matrix, t_proj_eigvecs)

        # if __debug__:
        #     self._print_debug(eigvals, rot_weights)

        vectors_elements = self._create_vectors_elements(eigvecs)
        weights['E1'], t_proj_elm_vecs = self._create_weights_e1(vectors_elements, q_sc            )
        weights['E2']                  = self._create_weights_e2(vectors_elements, weights['total'])

        weights['SR_E1'], rot_proj_elm_vecs = self._create_rotational_weights_for_elements(
            q_pc, transformation_matrix, t_proj_elm_vecs
        )

        return eigvals, eigvecs, weights

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
        t_proj_vectors : (natoms_p * ndims, nbands) array
            Vectors for SC after translational projection.
        """
        rot_proj_vectors = self._rotational_projector.project_vectors(
            t_proj_vectors, kpoint, transformation_matrix)

        rot_weights = np.linalg.norm(rot_proj_vectors, axis=1) ** 2

        self.check_rotational_projected_vectors(
            rot_proj_vectors, t_proj_vectors)

        return rot_weights, rot_proj_vectors

    def _create_rotational_weights_for_elements(self, kpoint, transformation_matrix, vectors):
        """

        Parameters
        ----------
        kpoint : 1d array
            Reciprocal space point in fractional coordinates for PC.
        vectors : (..., natoms_p * ndims, nbands) array
            Vectors for SC after translational projection.
        """
        projected_vectors = self._rotational_projector.project_vectors(
            vectors, kpoint, transformation_matrix)

        nirreps, natoms_p, nelms, tmp, nbands = projected_vectors.shape

        shape = (nirreps, natoms_p, nelms, natoms_p, nelms, nbands)
        weights = np.zeros(shape, dtype=complex)
        for i in range(nirreps):
            for j in range(nbands):
                weights[i, ..., j] = np.inner(
                    np.conj(projected_vectors[i, ..., j]), projected_vectors[i, ..., j])

        return weights, projected_vectors

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

    def _create_vectors_elements(self, vectors):
        elemental_projector = self._element_weights_calculator
        vectors_elements = elemental_projector.project_vectors(vectors)
        return vectors_elements

    def _create_weights_e1(self, vectors_elements, kpoint):
        """

        Parameters
        ----------
        vectors_elements : (natoms_p, nelms, natoms_u * ndims, nbands) array
        kpoint : Reciprocal space point in fractional coordinates for SC.

        Returns
        -------
        weights_e1 : (natoms_p, nelms, natoms_p, nelms, nbands) array
            Elemental weights.
        projected_vectors : (natoms_p, nelms, natoms_u * ndims, nbands) array
            Elemental projected vectors.
        """
        translational_projector = self._translational_projector

        projected_vectors = translational_projector.project_vectors(
            vectors=vectors_elements, kpoint=kpoint)

        natoms_p, nelms, tmp, nbands = projected_vectors.shape

        weights_e1 = np.zeros((natoms_p, nelms, natoms_p, nelms, nbands), dtype=complex)
        for i in range(nbands):
            weights_e1[..., i] = np.inner(
                np.conj(projected_vectors[..., i]), projected_vectors[..., i])

        return weights_e1, projected_vectors

    def _create_weights_e2(self, vectors_elements, weights_total):
        """
        
        Parameters
        ----------
        vectors_elements : (natoms_p, nelms, natoms_u * ndims, nbands) array
        weights_total : (nbands) array

        Returns
        -------
        weights_e2 : (natoms_p, nelms, nbands) array
        """
        weights_tmp = np.linalg.norm(vectors_elements, axis=2) ** 2  # (natoms_p, nelms, nbands)
        weights_e2 = weights_total * weights_tmp
        return weights_e2

    def get_distance(self):
        return self._distance

    def set_distance(self, distance):
        self._distance = distance

    def get_reduced_elements(self):
        return self._element_weights_calculator.get_reduced_elements()

    def write_hdf5(self, hdf5_file, group=''):
        """

        Parameters
        ----------
        hdf5_file : HDF5 file object
        group : String
            Indices for the present q-point.
        """
        natoms_primitive = self._cell.get_number_of_atoms()

        data_dict = {
            'point'            : self.get_point(),
            'q_star'           : self._q_star,
            'distance'         : self.get_distance(),
            'natoms_primitive' : natoms_primitive,
            'elements'         : self.get_reduced_elements(),
            'num_arms'         : self.get_narms(),
            'pointgroup_symbol': self.get_pointgroup_symbol(),
            'num_irreps'       : self.get_num_irreps(),
            'ir_labels'        : self.get_ir_labels(),
            'frequencies'      : self.get_frequencies_arms(),
            'weights_t'        : self._weights_arms['total'],
            'weights_e'        : self._weights_arms['E1'   ],
            'weights_s'        : self._weights_arms['SR'   ],
            'weights_s_e'      : self._weights_arms['SR_E1'],
            'weights_e2'       : self._weights_arms['E2'   ],
        }

        for k, v in data_dict.items():
            hdf5_file.create_dataset(group + k, data=v)

def calculate_frequencies(eigenvalues, factor):
    frequencies = np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues)
    frequencies *= factor
    return frequencies

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
