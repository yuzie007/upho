#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

__author__ = "Yuji Ikeda"

import numpy as np
from ph_unfolder.analysis.smearing import Smearing
from ph_unfolder.file_io import read_band_hdf5


class DensityExtractor(object):
    def __init__(self,
                 filename=None,
                 outfile=None,
                 function="gaussian",
                 fmin=0.0,
                 fmax=10.0,
                 fpitch=0.05,
                 sigma=1.0,
                 weight_label="pr_weights"):

        print("# sigma:", sigma)

        self._smearing = Smearing(
            function_name=function,
            sigma=sigma,
            xmin=fmin,
            xmax=fmax,
            xpitch=fpitch,
        )

        self._weight_label = weight_label

        self._outfile = outfile

        if filename is not None:
            self.load_data(filename)

    def load_data(self, filename):
        print("# Reading band.hdf5: ", end="")
        self._band_data = read_band_hdf5(filename)
        print("Finished")
        return self

    def run(self):
        weight_label = self._weight_label
        print("# weight: {}".format(weight_label))

        distances   = self._band_data["distances"]
        frequencies = self._band_data["frequencies"]
        weights     = self._band_data[weight_label]
        nqstars     = self._band_data["nqstars"]
        num_irs_list = self._band_data["num_irs"]
        if "eigenvectors_data" in self._band_data:
            eigenvectors_data = self._band_data["eigenvectors_data"]
            print_density = self.print_partial_density
        elif weight_label == "rot_pr_weights":
            print_density = self.print_partial_density
        else:
            eigenvectors_data = None
            print_density = self.print_total_density

        filename = self._outfile
        with open(filename, "w") as f:
            npath, nqpoint = frequencies.shape[:2]
            for ipath in range(npath):
                for i, d in enumerate(distances[ipath]):
                    if weight_label == "rot_pr_weights":
                        num_irs = num_irs_list[ipath, i]
                        weights_data = weights[ipath, i, :, :num_irs]
                    else:
                        weights_data = weights[ipath, i]

                    self.calculate_density(
                        d,
                        nqstar=nqstars[ipath, i],
                        frequencies_data=frequencies[ipath, i],
                        eigenvectors_data=None,
                        weights_data=weights_data)
                    print_density(f)

    def calculate_density(self,
                          distance,
                          nqstar,
                          frequencies_data,
                          weights_data,
                          eigenvectors_data=None):
        """

        Parameters
        ----------
        distance :
        nqstar : integer
        frequencies : (nqstar, nband) array
        weights : (nqstar, nband) array
        eigenvectors_data : (nqstar, nband, nband) array
        """
        self._distance = distance
        density_data = []
        for istar in range(nqstar):
            f = frequencies_data[istar]
            w = weights_data[istar]
            if eigenvectors_data is not None:
                eigenvectors = eigenvectors_data[istar]
                w = self._create_atom_weights(w, eigenvectors)
            density_data.append(self._smearing.run(f, w))
        density_data = np.sum(density_data, axis=0)  # Sum over star
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

    def print_ir_density(self, file_output):
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

    def set_distance(self, distance):
        self._distance = distance


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename",
                        default="band.hdf5",
                        type=str,
                        help="Filename for band.hdf5.")
    parser.add_argument("-o", "--outfile",
                        default="spectral_functions.dat",
                        type=str,
                        help="Output filename for spectral functions")
    parser.add_argument("--function",
                        default="gaussian",
                        type=str,
                        choices=["gaussian", "lorentzian", "histogram"],
                        help="Maximum plotted frequency (THz).")
    parser.add_argument("--weight_label",
                        default="pr_weights",
                        type=str,
                        choices=["pr_weights", "rot_pr_weights"],
                        help="Weight label to plot.")
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
        outfile=args.outfile,
        weight_label=args.weight_label,
    ).run()


if __name__ == "__main__":
    main()
