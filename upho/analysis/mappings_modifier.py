#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np


class MappingsModifier(object):
    def __init__(self, mappings):
        self.set_mappings(mappings)

    def set_mappings(self, mappings):
        self._mappings = np.array(mappings)

    def get_mappings(self):
        return self._mappings.copy()

    def invert_mappings(self):
        mappings = self._mappings
        mappings_inverse = -1 * np.zeros_like(mappings)
        for i, mapping in enumerate(mappings):
            mappings_inverse[i] = self._invert_mapping(mapping)
        return mappings_inverse

    def _invert_mapping(self, mapping):
        mapping_inverse = -1 * np.zeros_like(mapping)
        for index, value in enumerate(mapping):
            mapping_inverse[value] = index
        return mapping_inverse

    def expand_mappings(self, n, is_inverse=False):
        """Expand the last dimension of mappings by n"""
        if is_inverse:
            mappings = self.invert_mappings()
        else:
            mappings = self._mappings

        expanded_mappings = np.repeat(mappings, n, axis=-1)
        expanded_mappings *= n
        for i in range(n):
            expanded_mappings[..., i::n] += i
        return expanded_mappings
