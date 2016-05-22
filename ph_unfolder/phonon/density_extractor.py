#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

__author__ = "Yuji Ikeda"

import h5py
import numpy as np
from ph_unfolder.analysis.smearing import Smearing
from ph_unfolder.file_io import read_band_hdf5


class DensityExtractor(object):
    def __init__(self,
                 filename=None,
                 function="gaussian",
                 fmin=0.0,
                 fmax=10.0,
                 fpitch=0.05,
                 sigma=1.0):

        print("# sigma:", sigma)

        self._smearing = Smearing(
            function_name=function,
            sigma=sigma,
            xmin=fmin,
            xmax=fmax,
            xpitch=fpitch,
        )

        with h5py.File(filename, 'r') as f:
            self._band_data = f
            self._run()

    def load_data(self, filename):
        print("# Reading band.hdf5: ", end="")
        self._band_data = read_band_hdf5(filename)
        print("Finished")
        return self

    def _run(self):
        band_data = self._band_data

        npaths, npoints = band_data['paths'].shape[:2]
        fn_elements = 'sf_elements.dat'
        fn_irreps   = 'sf_irreps.dat'
        with open(fn_elements, 'w') as fe, open(fn_irreps, 'w') as fi:
            for ipath in range(npaths):
                for ip in range(npoints):
                    print(ipath, ip)
                    group = '{}/{}/'.format(ipath, ip)
                    distance        = band_data[group + 'distance'       ]
                    frequencies     = band_data[group + 'frequencies'    ]
                    rot_weights     = band_data[group + 'rot_weights'    ]
                    element_weights = band_data[group + 'element_weights']

                    self.set_distance(float(np.array(distance)))

                    self.calculate_density(frequencies, element_weights)
                    self._print_sf_elements(fe)

                    self.calculate_density(frequencies, rot_weights)
                    self._print_sf_irreps(fi)

    def calculate_density(self, frequencies, weights):
        """

        Parameters
        ----------
        frequencies : (num_arms, nbands) array
        weights : (num_arms, nbands) array
        """
        density_data = []
        for f, w in zip(frequencies, weights):
            density_data.append(self._smearing.run(f, w))
        density_data = np.sum(density_data, axis=0)  # Sum over arms
        self._density_data = density_data

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

    def print_total_density(self, file_output):
        """

        Parameters
        ----------
        file_output : A file object to print density.
        """
        distance = self._distance
        density_data = self._density_data
        xs = self._smearing.get_xs()
        for x, density in zip(xs, density_data):
            file_output.write("{:12.6f}".format(distance))
            file_output.write("{:12.6f}".format(x))
            file_output.write("{:12.6f}".format(density))
            file_output.write("\n")
        file_output.write("\n")

    def _print_sf_irreps(self, file_output):
        """

        Parameters
        ----------
        file_output : A file object to print density.
        """
        distance = self._distance
        density_data = self._density_data
        xs = self._smearing.get_xs()
        for x, densities in zip(xs, density_data):
            file_output.write("{:12.6f}".format(distance))
            file_output.write("{:12.6f}".format(x))
            file_output.write("{:12.6f}".format(np.sum(densities)))
            file_output.write("  ")
            for ir_density in densities:
                file_output.write("{:12.6f}".format(ir_density))
            file_output.write("\n")
        file_output.write("\n")

    def print_partial_density(self, file_output):
        """

        Parameters
        ----------
        file_output : A file object to print density.
        """
        distance = self._distance
        density_data = self._density_data
        xs = self._smearing.get_xs()
        for x, densities in zip(xs, density_data):
            file_output.write("{:12.6f}".format(distance))
            file_output.write("{:12.6f}".format(x))
            file_output.write("{:12.6f}".format(np.sum(densities)))
            file_output.write("  ")
            for partial_density in densities:
                file_output.write("{:12.6f}".format(partial_density))
            file_output.write("\n")
        file_output.write("\n")

    def _print_sf_elements(self, file_output):
        """

        Parameters
        ----------
        file_output : A file object to print density.
        """
        distance = self._distance
        density_data = self._density_data
        xs = self._smearing.get_xs()
        for x, densities in zip(xs, density_data):
            file_output.write("{:12.6f}".format(distance))
            file_output.write("{:12.6f}".format(x))
            file_output.write("{:12.6f}".format(np.sum(densities)))
            file_output.write("  ")
            for sf_elements in densities:
                file_output.write("  ")
                for sf_element in sf_elements:
                    file_output.write("{:12.6f}".format(sf_element))
            file_output.write("\n")
        file_output.write("\n")

    def set_distance(self, distance):
        self._distance = distance


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
    args = parser.parse_args()

    DensityExtractor(
        filename=args.filename,
        function=args.function,
        fmax=args.fmax,
        fmin=args.fmin,
        fpitch=args.fpitch,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()
