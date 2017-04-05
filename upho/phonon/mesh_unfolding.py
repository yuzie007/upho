#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

__author__ = "Yuji Ikeda"

import numpy as np
from phonopy.units import VaspToTHz
from phonopy.structure.grid_points import GridPoints
from phonopy.phonon.mesh import Mesh
from phonopy.structure.cells import get_primitive
from upho.phonon.eigenstates import Eigenstates


class MeshUnfolding(Mesh):
    def __init__(self,
                 dynamical_matrix,
                 unitcell_ideal,
                 primitive_matrix_ideal,
                 mesh,
                 shift=None,
                 is_time_reversal=True,
                 is_mesh_symmetry=True,
                 is_eigenvectors=False,
                 is_gamma_center=False,
                 star="none",
                 group_velocity=None,
                 rotations=None, # Point group operations in real space
                 factor=VaspToTHz,
                 use_lapack_solver=False,
                 mode="eigenvector"):

        self._mesh = np.array(mesh, dtype='intc')
        self._is_eigenvectors = is_eigenvectors
        self._factor = factor

        primitive_ideal_wrt_unitcell = (
            get_primitive(unitcell_ideal, primitive_matrix_ideal))
        self._cell = primitive_ideal_wrt_unitcell

        # ._dynamical_matrix must be assigned for calculating DOS
        # using the tetrahedron method.
        # In the "DOS", this is used to get "primitive".
        # In the "TetrahedronMesh", this is used just as the "Atoms".
        # Currently, we must not use the tetrahedron method,
        # because now we do not give the primitive but the unitcell for the
        # self._dynamical_matrix.
        self._dynamical_matrix = dynamical_matrix
        self._use_lapack_solver = use_lapack_solver

        self._gp = GridPoints(self._mesh,
                              np.linalg.inv(self._cell.get_cell()),
                              q_mesh_shift=shift,
                              is_gamma_center=is_gamma_center,
                              is_time_reversal=is_time_reversal,
                              rotations=rotations,
                              is_mesh_symmetry=is_mesh_symmetry)

        self._qpoints = self._gp.get_ir_qpoints()
        self._weights = self._gp.get_ir_grid_weights()

        self._star = star

        self._eigenstates_unfolding = Eigenstates(
            dynamical_matrix,
            unitcell_ideal,
            primitive_matrix_ideal,
            mode=mode,
            star=star,
            verbose=False)

        self._frequencies = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._set_phonon()

        self._group_velocities = None
        if group_velocity is not None:
            self._set_group_velocities(group_velocity)

    def get_pr_weights(self):
        return self._pr_weights

    def write_yaml(self):
        w = open('mesh.yaml', 'w')
        eigenvalues = self._eigenvalues
        natom = self._cell.get_number_of_atoms()
        lattice = np.linalg.inv(self._cell.get_cell()) # column vectors
        w.write("mesh: [ %5d, %5d, %5d ]\n" % tuple(self._mesh))
        w.write("nqpoint: %-7d\n" % self._qpoints.shape[0])
        w.write("natom:   %-7d\n" % natom)
        w.write("reciprocal_lattice:\n")
        for vec, axis in zip(lattice.T, ('a*', 'b*', 'c*')):
            w.write("- [ %12.8f, %12.8f, %12.8f ] # %2s\n" %
                    (tuple(vec) + (axis,)))
        w.write("phonon:\n")

        for i, q in enumerate(self._qpoints):
            w.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % tuple(q))
            w.write("  weight: %-5d\n" % self._weights[i])
            w.write("  band:\n")

            for j, freq in enumerate(self._frequencies[i]):
                w.write("  - # %d\n" % (j + 1))
                w.write("    frequency:  %15.10f\n" % freq)
                w.write("    pr_weight:  %15.10f\n" % self._pr_weights[i, j])

                if self._group_velocities is not None:
                    w.write("    group_velocity: ")
                    w.write("[ %13.7f, %13.7f, %13.7f ]\n" %
                            tuple(self._group_velocities[i, j]))

                if self._is_eigenvectors:
                    w.write("    eigenvector:\n")
                    for k in range(natom):
                        w.write("    - # atom %d\n" % (k+1))
                        for l in (0,1,2):
                            w.write("      - [ %17.14f, %17.14f ]\n" %
                                    (self._eigenvectors[i,k*3+l,j].real,
                                     self._eigenvectors[i,k*3+l,j].imag))
            w.write("\n")

    def _set_phonon(self):
        # For the unfolding method.
        # num_band = self._cell.get_number_of_atoms() * 3
        cell = self._dynamical_matrix.get_primitive()
        num_band = cell.get_number_of_atoms() * 3
        num_qpoints = len(self._qpoints)

        self._eigenvalues = np.zeros((num_qpoints, num_band), dtype='double')
        self._frequencies = np.zeros_like(self._eigenvalues)
        # TODO(ikeda): the folling is very bad if we turn on is_star
        self._pr_weights  = np.zeros_like(self._eigenvalues)
        if self._is_eigenvectors or self._use_lapack_solver:
            self._eigenvectors = np.zeros(
                (num_qpoints, num_band, num_band,), dtype='complex128')

        if self._use_lapack_solver:
            print("ERROR: _use_lapack_solver is not considered for this script.")
            raise ValueError
        else:
            for i, q in enumerate(self._qpoints):
                eigvals, eigvecs, self._pr_weights[i], nstar = (
                    self._eigenstates_unfolding.extract_eigenstates(q)
                )
                self._eigenvalues[i] = eigvals.real
                if self._is_eigenvectors:
                    self._eigenvectors[i] = eigvecs
            self._frequencies = np.array(np.sqrt(abs(self._eigenvalues)) *
                                         np.sign(self._eigenvalues),
                                         dtype='double',
                                         order='C') * self._factor
