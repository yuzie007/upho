#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

__author__ = "Yuji Ikeda"

import numpy as np


def lorentzian(x, position, width):
    return 1.0 / (np.pi * width * (1.0 + ((x - position) / width) ** 2))


def gaussian(x, mu, sigma):
    tmp = np.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return 1.0 / np.sqrt(2.0 * np.pi) / sigma * tmp


class Smearing(object):
    def __init__(self,
                 function_name="gaussian",
                 sigma=0.1,
                 xmin=0.0,
                 xmax=1.0,
                 xpitch=0.1):

        self.set_smearing_function(function_name)
        self.set_sigma(sigma)
        self.build_xs(xmin, xmax, xpitch)

    def set_smearing_function(self, function_name):
        if function_name == "gaussian":
            self._smearing_function = gaussian
        elif function_name == "lorentzian":
            self._smearing_function = lorentzian
        else:
            raise ValueError("Invalid smaering function name.")
        return self

    def build_xs(self, xmin, xmax, xpitch):
        n = int(round((xmax - xmin) / xpitch)) + 1
        self._xs = np.linspace(xmin, xmax, n)
        return self

    def set_xs(self, xs):
        self._xs = xs

    def get_xs(self):
        return self._xs

    def set_sigma(self, sigma):
        self._sigma = sigma

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
