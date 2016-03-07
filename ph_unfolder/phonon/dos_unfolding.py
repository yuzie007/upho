#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

__author__ = "Yuji Ikeda"

import numpy as np
from phonopy.phonon.dos import TotalDos


class TotalDosUnfolding(TotalDos):
    def __init__(self, mesh_object, sigma=None, tetrahedron_method=False):
        TotalDos.__init__(
            self,
            mesh_object,
            sigma=sigma,
            tetrahedron_method=tetrahedron_method)
        self._pr_weights = mesh_object.get_pr_weights()

    def _get_density_of_states_at_freq(self, f):
        tmp = self._smearing_function.calc(self._frequencies - f)
        tmp *= self._pr_weights
        return np.sum(np.dot(self._weights, tmp)) / np.sum(self._weights)
