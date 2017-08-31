#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from .lebedev_write import Lebedev

__author__ = 'Yuji Ikeda'


class QpointsCreator(object):
    def __init__(self, radii, lebedev):
        self._radii = np.asarray(radii)
        self._lebedev = lebedev
        self._create()

    def _create(self):
        radii = self._radii
        xyzw = np.array(Lebedev(self._lebedev))
        self._qpoints = (radii[:, None, None] * xyzw[None, :, :3]).reshape(-1, 3)
        self._weights = np.tile(xyzw[:, 3], radii.size)
        self._groups = np.repeat(np.arange(xyzw.shape[0]), radii.size)

    def write(self):
        self._write_qpoints()
        self._write_weights()
        self._write_groups()

    def _write_qpoints(self):
        with open('QPOINTS', 'w') as f:
            for q in self._qpoints:
                for x in q:
                    f.write('{:20.16f}'.format(x))
                f.write('\n')

    def _write_weights(self):
        with open('WEIGHTS', 'w') as f:
            for w in self._weights:
                f.write('{:20.16f}'.format(w))
                f.write('\n')

    def _write_groups(self):
        with open('GROUPS', 'w') as f:
            for g in self._groups:
                f.write('{:6d}'.format(g))
                f.write('\n')

    def get_qpoints(self):
        return self._qpoints

    def get_weights(self):
        return self._weights

    def get_groups(self):
        return self._groups
