#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np
from phonopy.structure.symmetry import Symmetry, get_pointgroup


class UnfolderSymmetry(Symmetry):
    def get_group_of_wave_vector(self, kpoint):
        rotations = self._symmetry_operations["rotations"]
        translations = self._symmetry_operations["translations"]
        lattice = self._cell.get_cell()
        rotations_kpoint = []
        translations_kpoint = []
        for r, t in zip(rotations, translations):
            diff = np.dot(kpoint, r) - kpoint
            diff -= np.rint(diff)
            dist = np.linalg.norm(np.dot(np.linalg.inv(lattice), diff))
            if dist < self._symprec:
                rotations_kpoint.append(r)
                translations_kpoint.append(t)
        return np.array(rotations_kpoint), np.array(translations_kpoint)
