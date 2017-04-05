#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import h5py
from phonopy.units import VaspToTHz
from upho.phonon.eigenstates import Eigenstates


class SinglePoint(object):
    def __init__(self,
                 qpoint,
                 distance,
                 dynamical_matrix,
                 unitcell_ideal,
                 primitive_matrix_ideal,
                 density_extractor,
                 factor=VaspToTHz,
                 star="none",
                 mode="eigenvector",
                 verbose=False):

        self._qpoint = qpoint
        self._distance = distance

        self._factor = factor

        self._eigenstates = Eigenstates(
            dynamical_matrix,
            unitcell_ideal,
            primitive_matrix_ideal,
            mode=mode,
            star=star,
            verbose=verbose)

        # self._density_extractor = density_extractor

        # fn_sf_atoms = "spectral_functions_atoms.dat"
        # fn_sf_irs   = "spectral_functions_irs.dat"
        # with open(fn_sf_atoms, "w") as fatoms, open(fn_sf_irs, "w") as firs:
        #     self._file_sf_atoms = fatoms
        #     self._file_sf_irs   = firs

        #     self.run()

        with h5py.File('point.hdf5', 'w') as f:
            self._hdf5_file = f
            self.run()

    def run(self):
        qpoint = self._qpoint
        distance = self._distance

        eigenstates = self._eigenstates

        if True:
            eigenstates.set_distance(distance)
            eigenstates.extract_eigenstates(qpoint)

            eigenstates.write_hdf5(self._hdf5_file, group='')

            # # Print spectral functions
            # density_extractor = self._density_extractor

            # density_extractor.calculate_density(
            #     distance, narms, frequencies,
            #     weights_data=pr_weights,
            #     eigenvectors_data=eigvecs)
            # density_extractor.print_partial_density(self._file_sf_atoms)

            # density_extractor.calculate_density(
            #     distance, narms, frequencies,
            #     weights_data=rot_pr_weights[:, :num_irs])
            # density_extractor.print_partial_density(self._file_sf_irs)
