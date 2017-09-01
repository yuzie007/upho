#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import h5py
import numpy as np
from phonopy.units import VaspToTHz
from phonopy.structure.cells import get_primitive
from upho.phonon.eigenstates import Eigenstates

__author__ = 'Yuji Ikeda'


class BandStructure(object):
    def __init__(self,
                 paths,
                 dynamical_matrix,
                 unitcell_ideal,
                 primitive_matrix_ideal,
                 is_eigenvectors=False,
                 is_band_connection=False,
                 group_velocity=None,
                 factor=VaspToTHz,
                 star="none",
                 mode="eigenvector",
                 verbose=False):
        """

        Args:
            dynamical_matrix:
                Dynamical matrix for the (disordered) supercell.
            primitive_ideal_wrt_unitcell:
                Primitive cell w.r.t. the unitcell (not the supercell).
        """
        # ._dynamical_matrix must be assigned for calculating DOS
        # using the tetrahedron method.
        self._dynamical_matrix = dynamical_matrix

        # self._cell is used for write_yaml and _shift_point.
        # This must correspond to the "ideal" primitive cell.
        primitive_ideal_wrt_unitcell = (
            get_primitive(unitcell_ideal, primitive_matrix_ideal))
        self._cell = primitive_ideal_wrt_unitcell

        self._factor = factor
        self._is_eigenvectors = is_eigenvectors
        self._is_band_connection = is_band_connection
        if is_band_connection:
            self._is_eigenvectors = True
        self._group_velocity = group_velocity

        self._paths = [np.array(path) for path in paths]
        self._distances = []
        self._distance = 0.
        self._special_point = [0.]
        self._eigenvalues = None
        self._eigenvectors = None
        self._frequencies = None

        self._star = star
        self._mode = mode

        self._eigenstates = Eigenstates(
            dynamical_matrix,
            unitcell_ideal,
            primitive_matrix_ideal,
            mode=mode,
            star=star,
            verbose=verbose)

        with h5py.File('band.hdf5', 'w') as f:
            self._hdf5_file = f
            self._write_hdf5_header()
            self._set_band(verbose=verbose)

    def _write_hdf5_header(self):
        self._hdf5_file.create_dataset('paths', data=self._paths)

    def _set_initial_point(self, qpoint):
        self._lastq = qpoint.copy()

    def _shift_point(self, qpoint):
        self._distance += np.linalg.norm(
            np.dot(qpoint - self._lastq,
                   np.linalg.inv(self._cell.get_cell()).T))
        self._lastq = qpoint.copy()

    def _set_band(self, verbose=False):
        for ipath, path in enumerate(self._paths):
            self._set_initial_point(path[0])
            self._solve_dm_on_path(ipath, path, verbose)

            self._special_point.append(self._distance)

    def _solve_dm_on_path(self, ipath, path, verbose):
        eigenstates = self._eigenstates

        is_nac = self._dynamical_matrix.is_nac()

        for ip, q in enumerate(path):
            self._shift_point(q)

            if is_nac:
                raise ValueError('NAC is not implemented yet for unfolding')

            eigenstates.set_distance(self._distance)
            eigenstates.extract_eigenstates(q)

            group = '{}/{}/'.format(ipath, ip)
            eigenstates.write_hdf5(self._hdf5_file, group=group)

    def get_unitcell_orig(self):
        unitcell_orig = self._dynamical_matrix.get_primitive()
        return unitcell_orig

    def get_reduced_elements(self):
        unitcell_orig = self.get_unitcell_orig()
        elements = unitcell_orig.get_chemical_symbols()
        reduced_elements = sorted(set(elements), key=elements.index)
        return reduced_elements
