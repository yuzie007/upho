#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "Yuji Ikeda"

import numpy as np
from phonopy.phonon.band_structure import BandStructure
from phonopy.units import VaspToTHz
from phonopy.structure.cells import get_primitive
from eigenstates_unfolding import EigenstatesUnfolding


class BandStructureUnfolding(BandStructure):
    def __init__(self,
                 paths,
                 dynamical_matrix,
                 unitcell_ideal,
                 primitive_matrix_ideal,
                 density_extractor,
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

        self._eigenstates_unfolding = EigenstatesUnfolding(
            dynamical_matrix,
            unitcell_ideal,
            primitive_matrix_ideal,
            mode=mode,
            star=star,
            verbose=verbose)

        self._density_extractor = density_extractor

        fn_sf_atoms = "spectral_functions_atoms.dat"
        fn_sf_irs   = "spectral_functions_irs.dat"
        with open(fn_sf_atoms, "w") as fatoms, open(fn_sf_irs, "w") as firs:
            self._file_sf_atoms = fatoms
            self._file_sf_irs   = firs

            self._set_band(verbose=verbose)

    def write_hdf5(self):
        import h5py
        with h5py.File('band.hdf5', 'w') as w:
            w.create_dataset('paths', data=self._paths)
            w.create_dataset('distances', data=self._distances)
            w.create_dataset('nqstars', data=self._nqstars)
            w.create_dataset('num_irs', data=self._num_irs)
            w.create_dataset('frequencies', data=self._frequencies)
            w.create_dataset('pr_weights', data=self._pr_weights)
            w.create_dataset('rot_pr_weights', data=self._rot_pr_weights)
            w.create_dataset('pg_symbols', data=self._pg_symbols)
            w.create_dataset('ir_labels', data=self._ir_labels)
            if self._group_velocity is not None:
                w.create_dataset('group_velocities', data=self._group_velocity)
            if self._eigenvectors is not None:
                w.create_dataset('eigenvectors_data', data=self._eigenvectors)

    def write_yaml(self):
        w = open('band.yaml', 'w')
        natom = self._cell.get_number_of_atoms()
        lattice = np.linalg.inv(self._cell.get_cell())  # column vectors
        nqpoint = 0
        for qpoints in self._paths:
            nqpoint += len(qpoints)
        w.write("nqpoint: %-7d\n" % nqpoint)
        w.write("npath: %-7d\n" % len(self._paths))
        w.write("natom: %-7d\n" % (natom))
        w.write("reciprocal_lattice:\n")
        for vec, axis in zip(lattice.T, ('a*', 'b*', 'c*')):
            w.write("- [ %12.8f, %12.8f, %12.8f ] # %2s\n" %
                    (tuple(vec) + (axis,)))
        w.write("phonon:\n")
        for i, (qpoints, distances, frequencies, weights) in enumerate(zip(
            self._paths,
            self._distances,
            self._frequencies,
            self._weights)):
            for j, q in enumerate(qpoints):
                w.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % tuple(q))
                w.write("  distance: %12.7f\n" % distances[j])
                w.write("  band:\n")
                for k, freq in enumerate(frequencies[j]):
                    w.write("  - # %d\n" % (k + 1))
                    w.write("    frequency: %15.10f\n" % freq)
                    w.write("    weight:    %15.10f\n" % weights[j][k])

                    if self._group_velocity is not None:
                        gv = self._group_velocities[i][j, k]
                        w.write("    group_velocity: ")
                        w.write("[ %13.7f, %13.7f, %13.7f ]\n" % tuple(gv))

                    if self._is_eigenvectors:
                        eigenvectors = self._eigenvectors[i]
                        w.write("    eigenvector:\n")
                        for l in range(natom):
                            w.write("    - # atom %d\n" % (l + 1))
                            for m in (0, 1, 2):
                                w.write("      - [ %17.14f, %17.14f ]\n" %
                                        (eigenvectors[j, l * 3 + m, k].real,
                                         eigenvectors[j, l * 3 + m, k].imag))

                w.write("\n")

    def _set_band(self, verbose=False):
        frequencies = []
        eigvecs = []
        pr_weights = []
        nqstars = []
        group_velocities = []
        distances = []
        rot_pr_weights = []
        is_nac = self._dynamical_matrix.is_nac()
        num_irs = []
        ir_labels = []
        pg_symbols = []

        for path in self._paths:
            self._set_initial_point(path[0])

            (distances_on_path,
             frequencies_on_path,
             eigvecs_on_path,
             pr_weights_on_path,
             nqstars_on_path,
             gv_on_path,
             rot_pr_weights_on_path,
             num_irs_on_path,
             ir_labels_on_path) = self._solve_dm_on_path(path, verbose)

            frequencies.append(np.array(frequencies_on_path))
            pr_weights.append(np.array(pr_weights_on_path))
            rot_pr_weights.append(rot_pr_weights_on_path)
            nqstars.append(np.array(nqstars_on_path))
            num_irs.append(np.array(num_irs_on_path))
            ir_labels.append(np.array(ir_labels_on_path))
            pg_symbols.append(self.get_pg_symbol_on_path())

            if self._is_eigenvectors:
                eigvecs.append(np.array(eigvecs_on_path))
            if self._group_velocity is not None:
                group_velocities.append(np.array(gv_on_path))
            distances.append(np.array(distances_on_path))
            self._special_point.append(self._distance)

        self._frequencies = frequencies
        self._pr_weights  = pr_weights
        self._nqstars     = nqstars
        self._num_irs     = num_irs
        self._rot_pr_weights = rot_pr_weights
        self._ir_labels = ir_labels
        self._pg_symbols = np.array(pg_symbols, dtype=str)

        if self._is_eigenvectors:
            self._eigenvectors = eigvecs
        if self._group_velocity is not None:
            self._group_velocities = group_velocities
        self._distances = distances

    def _solve_dm_on_path(self, path, verbose):
        is_nac = self._dynamical_matrix.is_nac()
        distances_on_path = []
        frequencies_on_path = []
        eigvecs_on_path = []
        pr_weights_on_path = []
        nqstar_on_path = []
        gv_on_path = []
        rot_pr_weights_on_path = []
        pg_symbol_on_path = []
        num_irs_on_path = []
        ir_labels_on_path = []

        # Probably group_velocity has not worked for the unfolding so far.
        if self._group_velocity is not None:
            self._group_velocity.set_q_points(path)
            gv = self._group_velocity.get_group_velocity()

        for i, q in enumerate(path):
            self._shift_point(q)
            distances_on_path.append(self._distance)

            if is_nac:
                print("ERROR: NAC is not implemented yet for unfolding")
                raise ValueError
            else:
                eigvals, eigvecs, pr_weights, nqstar, rot_pr_weights, num_irs, ir_labels = (
                    self._eigenstates_unfolding.extract_eigenstates(q)
                )
                frequencies = self._calculate_frequencies(eigvals)

                # Print spectral functions
                density_extractor = self._density_extractor

                density_extractor.calculate_density(
                    self._distance, nqstar, frequencies,
                    weights_data=pr_weights,
                    eigenvectors_data=eigvecs)
                density_extractor.print_partial_density(self._file_sf_atoms)

                density_extractor.calculate_density(
                    self._distance, nqstar, frequencies,
                    weights_data=rot_pr_weights[:, :num_irs])
                density_extractor.print_partial_density(self._file_sf_irs)

            frequencies_on_path.append(frequencies)
            pr_weights_on_path.append(pr_weights)
            rot_pr_weights_on_path.append(rot_pr_weights)
            ir_labels_on_path.append(ir_labels)
            pg_symbol_on_path.append(
                self._eigenstates_unfolding.get_pointgroup_symbol())

            if self._is_eigenvectors:
                eigvecs_on_path.append(eigvecs)
            if self._group_velocity is not None:
                gv_on_path.append(gv[i])

            nqstar_on_path.append(nqstar)
            num_irs_on_path.append(num_irs)

        ir_labels_on_path = np.array(ir_labels_on_path)

        self._pg_symbol_on_path = pg_symbol_on_path

        return (
            distances_on_path,
            frequencies_on_path,
            eigvecs_on_path,
            pr_weights_on_path,
            nqstar_on_path,
            gv_on_path,
            rot_pr_weights_on_path,
            num_irs_on_path,
            ir_labels_on_path)

    def get_pg_symbol_on_path(self):
        return self._pg_symbol_on_path

    def _calculate_frequencies(self, eigenvalues):
        frequencies = np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues)
        frequencies *= self._factor
        return frequencies
