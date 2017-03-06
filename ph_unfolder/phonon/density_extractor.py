#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import h5py
import numpy as np
from ph_unfolder.analysis.smearing import Smearing, create_points


__author__ = "Yuji Ikeda"


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
                 is_squared=True,
                 group=None):

        self._is_squared = is_squared
        self._group = group

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
        raise NotImplementedError

    def load_weights(self, group):
        """

        Parameters
        ----------
        group : string

        Returns
        -------

        """
        weights = {}
        frequencies = self._band_data[group + 'frequencies']
        weights['total'          ] = self._band_data[group + 'weights_t'  ]
        weights['elements'       ] = self._band_data[group + 'weights_e'  ]
        weights['irreps'         ] = self._band_data[group + 'weights_s'  ]
        # weights['irreps_elements'] = self._band_data[group + 'weights_s_e']

        distance                   = self._band_data[group + 'distance'   ]
        distance = float(np.array(distance))

        return distance, frequencies, weights

    def calculate_spectral_functions(self, frequencies, weights):
        spectral_functions = {}
        for k, v in weights.items():
            spectral_functions[k] = self.calculate_density(frequencies, v)
        return spectral_functions

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


class DensityExtractorHDF5(DensityExtractor):
    def _run(self):
        band_data = self._band_data

        npaths, npoints = band_data['paths'].shape[:2]
        filename_sf = 'sf.hdf5'
        with h5py.File(filename_sf, 'w') as f:
            self._print_header(f)
            for ipath in range(npaths):
                for ip in range(npoints):
                    print(ipath, ip)
                    group = '{}/{}/'.format(ipath, ip)
                    frequencies = band_data[group + 'frequencies']
                    weights_t   = band_data[group + 'weights_t'  ]
                    weights_e   = band_data[group + 'weights_e'  ]
                    weights_s   = band_data[group + 'weights_s'  ]
                    weights_s_e = band_data[group + 'weights_s_e']

                    frequencies = np.array(frequencies)
                    if self._is_squared:
                        energies = square_frequencies(frequencies)
                    else:
                        energies = frequencies

                    total_sf       = self.calculate_density(energies, weights_t  )
                    partial_sf_e   = self.calculate_density(energies, weights_e  )
                    partial_sf_s   = self.calculate_density(energies, weights_s  )
                    partial_sf_s_e = self.calculate_density(energies, weights_s_e)

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

    def _print_header(self, file_output):
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


class DensityExtractorText(DensityExtractor):
    def _run(self):
        band_data = self._band_data
        npaths, npoints = band_data['paths'].shape[:2]
        fn_irreps   = 'sf_irreps.dat'
        fn_elements = 'sf_elements.dat'
        with open(fn_irreps, 'w') as fi, open(fn_elements, 'w') as fe:
            self._print_header(fi)
            self._print_header(fe)
            if self._group is None:
                for ipath in range(npaths):
                    for ip in range(npoints):
                        print(ipath, ip)
                        group = '{}/{}/'.format(ipath, ip)
                        self._run_point(group, fi, fe)
            else:
                self._run_point(self._group, fi, fe)

    def _run_point(self, group, fi, fe):
        distance, frequencies, weights = self.load_weights(group)

        frequencies = np.array(frequencies)
        if self._is_squared:
            energies = square_frequencies(frequencies)
        else:
            energies = frequencies

        spectral_functions = self.calculate_spectral_functions(
            energies, weights)

        self._write_irreps  (fi, group, distance, spectral_functions)
        self._write_elements(fe, group, distance, spectral_functions)

    def _write_irreps(self, file_out, group, distance, sf):
        ir_labels = np.array(self._band_data[group + 'ir_labels'])

        file_out.write('# {:10s}'.format('Dist.'))
        file_out.write('{:12s}'.format('Freq. (THz)'))
        file_out.write('{:12s}'.format('Total'))
        for ir_label in ir_labels:
            file_out.write('{:12s}'.format(ir_label))
        file_out.write('\n')

        frequencies = self._evaluated_energies
        sf_total   = sf['total']
        sf_partial = sf['irreps']

        for i, frequency in enumerate(frequencies):
            file_out.write('{:12.6f}'.format(distance))
            file_out.write('{:12.6f}'.format(frequency))
            file_out.write('{:12.6f}'.format(sf_total[i]))
            for v in sf_partial[i, :]:
                file_out.write('{:12.6f}'.format(v))
            file_out.write('\n')
        file_out.write('\n')

    def _write_elements(self, file_out, group, distance, sf):
        elements = np.array(self._band_data[group + 'elements'])
        ne = len(elements)

        file_out.write('# {:10s}'.format('Dist.'))
        file_out.write('{:12s}'.format('Freq. (THz)'))
        file_out.write('{:12s}'.format('Total'))
        for ie in range(ne):
            for je in range(ie, ne):
                label = elements[ie] + '-' + elements[je]
                file_out.write('{:12s}'.format(label))
        file_out.write('\n')

        frequencies = self._evaluated_energies
        sf_total   = sf['total']
        sf_partial = sf['elements']

        for i, frequency in enumerate(frequencies):
            file_out.write('{:12.6f}'.format(distance))
            file_out.write('{:12.6f}'.format(frequency))
            file_out.write('{:12.6f}'.format(sf_total[i]))
            for ie in range(ne):
                for je in range(ie, ne):
                    if ie == je:
                        v = np.real(np.sum(sf_partial[i, :, ie, :, je]))
                    else:
                        v = np.real(
                            np.sum(sf_partial[i, :, ie, :, je]) +
                            np.sum(sf_partial[i, :, je, :, ie])
                        )
                    file_out.write('{:12.6f}'.format(float(v)))
            file_out.write('\n')
        file_out.write('\n')

    def _print_header(self, file_output):
        function_name = self._smearing.get_function_name()
        sigma         = self._smearing.get_sigma()
        is_squared = self._is_squared

        file_output.write('# function: {}\n'.format(function_name))
        file_output.write('# sigma: {}\n'.format(sigma))  # For THz^2 or THz
        file_output.write('# is_squared: {}\n'.format(is_squared))
        file_output.write('#\n')
