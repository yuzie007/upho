#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

__author__ = "Yuji Ikeda"

import numpy as np
from .functions import lorentzian


def gaussian(x, mu, sigma):
    tmp = np.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return 1.0 / np.sqrt(2.0 * np.pi) / sigma * tmp


def histogram(x, positions, width):
    tmp = np.zeros((x.shape[0], positions.shape[-1]))
    x_tmp = np.zeros(x.shape[0] + 1)
    x_tmp[:-1] = x[:, 0] - width * 0.5
    x_tmp[-1] = x[-1, 0] + width * 0.5
    for i, p in enumerate(positions.T):  # only one value for each loop
        tmp[:, i] = np.histogram(p, x_tmp)[0]
    tmp /= width
    return tmp


def create_points(xmin, xmax, xpitch):
    n = int(round((xmax - xmin) / xpitch)) + 1
    points = np.linspace(xmin, xmax, n)
    return points


class Smearing(object):
    def __init__(self,
                 function_name="gaussian",
                 sigma=0.1,
                 xmin=None,
                 xmax=None,
                 xpitch=None):

        self._function_name = function_name
        self.set_smearing_function(function_name)
        self.set_sigma(sigma)
        if xmin is not None and xmax is not None and xpitch is not None:
            self.build_xs(xmin, xmax, xpitch)
        elif not (xmin is None and xmax is None and xpitch is None):
            raise ValueError('Some of xmin, xmax, and xpitch are None')

    def set_smearing_function(self, function_name):
        if function_name == "gaussian":
            self._smearing_function = gaussian
        elif function_name == "lorentzian":
            self._smearing_function = lorentzian
        elif function_name == "histogram":
            self._smearing_function = histogram
        else:
            raise ValueError("Invalid smaering function name.")
        return self

    def build_xs(self, xmin, xmax, xpitch):
        self.set_xs(create_points(xmin, xmax, xpitch))
        return self

    def set_xs(self, xs):
        self._xs = xs

    def get_xs(self):
        return self._xs

    def set_sigma(self, sigma):
        self._sigma = sigma

    def get_sigma(self):
        return self._sigma

    def get_function_name(self):
        return self._function_name

    def run(self, peaks, weights=None):
        """Get smeared values.

        Args:
            peaks:
            weights:
                Weight factors for "peaks".
                Now this can be one-dimeansional and multi-dimensional arrays.
                The last dimension must have the same order as the "peaks".
        """
        smearing_function = self._smearing_function
        xs = self._xs
        sigma = self._sigma

        tmp = smearing_function(xs[:, None], peaks[None, :], sigma)
        if weights is not None:
            values = np.inner(tmp, weights)
        else:
            values = np.sum(tmp, axis=1)

        return values
