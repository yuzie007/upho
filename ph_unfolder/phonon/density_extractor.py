#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

__author__ = "Yuji Ikeda"

import h5py
import numpy as np
from ph_unfolder.analysis.smearing import Smearing, create_points


def square_frequencies(frequencies):
    frequencies_2 = np.sign(frequencies) * frequencies ** 2
    return frequencies_2


class DensityExtractor(object):
    def __init__(self,
                 filename=None,
                 function="gaussian",
                 fmin=0.0,
                 fmax=10.0,
                 fpitch=0.05,
                 sigma=1.0,
                 is_squared=True):

        self._is_squared = is_squared

        self._smearing = Smearing(
            function_name=function,
            sigma=sigma,
        )

        frequencies = create_points(fmin, fmax, fpitch)
        self.set_evaluated_energies(frequencies)

        if is_squared:
            energies = square_frequencies(frequencies)
        else:
            energies = frequencies
        self._smearing.set_xs(energies)

        with h5py.File(filename, 'r') as f:
            self._band_data = f
            self._run()

    def set_evaluated_energies(self, evaluated_energies):
        self._evaluated_energies = evaluated_energies

    def get_evaluated_energies(self):
        return np.copy(self._evaluated_energies)

    def _run(self):
        band_data = self._band_data

        npaths, npoints = band_data['paths'].shape[:2]
        filename_sf = 'sf.hdf5'
        with h5py.File(filename_sf, 'w') as f:
            self.print_header(f)
            for ipath in range(npaths):
                for ip in range(npoints):
                    print(ipath, ip)
                    group = '{}/{}/'.format(ipath, ip)
                    frequencies     = band_data[group + 'frequencies'    ]
                    trans_weights   = band_data[group + 'trans_weights'  ]
                    element_weights = band_data[group + 'element_weights']
                    rot_weights     = band_data[group + 'rot_weights'    ]
                    rot_elemental_weights = band_data[group + 'rot_elemental_weights']

                    frequencies = np.array(frequencies)
                    if self._is_squared:
                        energies = square_frequencies(frequencies)
                    else:
                        energies = frequencies

                    total_sf = self.calculate_density(
                        energies, trans_weights)
                    partial_sf_e = self.calculate_density(
                        energies, element_weights)
                    partial_sf_s = self.calculate_density(
                        energies, rot_weights)
                    partial_sf_s_e = self.calculate_density(
                        energies, rot_elemental_weights)

                    self._write(
                        f,
                        group,
                        total_sf,
                        partial_sf_e,
                        partial_sf_s,
                        partial_sf_s_e,
                    )

    def _write(self,
               file_out,
               group,
               total_sf,
               partial_sf_e,
               partial_sf_s,
               partial_sf_s_e):

        keys = [
            'natoms_primitive',
            'elements',
            'distance',
            'pointgroup_symbol',
            'num_irreps',
            'ir_labels',
        ]

        for k in keys:
            file_out.create_dataset(
                group + k, data=np.array(self._band_data[group + k])
            )
        file_out.create_dataset(group + 'total_sf'      , data=total_sf      )
        file_out.create_dataset(group + 'partial_sf_e'  , data=partial_sf_e  )
        file_out.create_dataset(group + 'partial_sf_s'  , data=partial_sf_s  )
        file_out.create_dataset(group + 'partial_sf_s_e', data=partial_sf_s_e)

    def print_header(self, file_output):
        function_name = self._smearing.get_function_name()
        sigma         = self._smearing.get_sigma()
        is_squared = self._is_squared
        if is_squared:
            unit = 'THz^2'
        else:
            unit = 'THz'
        frequencies = self._evaluated_energies

        file_output.create_dataset('function', data=function_name)
        file_output.create_dataset('sigma', data=sigma)  # For THz^2 or THz
        file_output.create_dataset('is_squared', data=is_squared)
        file_output.create_dataset('frequencies', data=frequencies)
        file_output.create_dataset('paths', data=self._band_data['paths'])

    def calculate_density(self, frequencies, weights):
        """

        Parameters
        ----------
        frequencies : (num_arms, nbands) array
        weights : (num_arms, ... , nbands) array
        """
        density_data = []
        for f, w in zip(frequencies, weights):
            density_data.append(self._smearing.run(f, w))
        density_data = np.sum(density_data, axis=0)  # Sum over arms
        return density_data

    def _create_atom_weights(self, weights, vectors, ndim=3):
        """

        Parameters
        ----------
        weights : (nbands) array
            Original weights on mode eigenvectors.
        vectors : (nbands, nbands) array
            Original mode eigenvectors.
        ndim : Integer
            # of dimensions of the considered space.

        Returns
        -------
        atom_weights : (natoms, nbands) array
            natoms should be equal to nbands // ndim.

        Note
        ----
        This method is used to not care Cartesian coordinates.
        This is because it can be confusing when we average contributions
        from arms of the star.
        """
        shape = vectors.shape
        tmp = vectors.reshape(shape[0] // ndim, ndim, shape[1])
        atom_weights = (np.linalg.norm(tmp, axis=1) ** 2) * weights
        return atom_weights

    def _create_cartesian_weights(self, weights, vectors):
        cartesian_weights = (np.abs(vectors) ** 2) * weights
        return cartesian_weights


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename",
                        default="band.hdf5",
                        type=str,
                        help="Filename for band.hdf5.")
    parser.add_argument("--function",
                        default="gaussian",
                        type=str,
                        choices=["gaussian", "lorentzian", "histogram"],
                        help="Maximum plotted frequency (THz).")
    parser.add_argument("--fmax",
                        default=10.0,
                        type=float,
                        help="Maximum plotted frequency (THz).")
    parser.add_argument("--fmin",
                        default=-2.5,
                        type=float,
                        help="Minimum plotted frequency (THz).")
    parser.add_argument("--fpitch",
                        default=0.05,
                        type=float,
                        help="Pitch for frequencies (THz).")
    parser.add_argument("-s", "--sigma",
                        default=0.1,
                        type=float,
                        help="Sigma for frequencies (THz).")
    parser.add_argument("--nosquared", dest='is_squared',
                        action='store_false',
                        help="Squared frequencies are not considered.")
    args = parser.parse_args()

    DensityExtractor(
        filename=args.filename,
        function=args.function,
        fmax=args.fmax,
        fmin=args.fmin,
        fpitch=args.fpitch,
        sigma=args.sigma,
        is_squared=args.is_squared,
    )


if __name__ == "__main__":
    main()
