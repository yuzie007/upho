#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np
from phonopy.structure.symmetry import get_pointgroup
from ph_unfolder.structure.character_tables import character_tables
from group.mathtools import similarity_transformation


class Irreps(object):
    def __init__(self, rotations):
        self._rotations = rotations

    def run(self):
        self._create_conventional_rotations()
        self._assign_character_table_data()
        self._assign_class_labels_to_rotations()

    def _create_conventional_rotations(self):
        rotations = self._rotations

        pointgroup = get_pointgroup(rotations)
        pointgroup_symbol = pointgroup[0]
        transformation_matrix = pointgroup[1]

        conventional_rotations = self._transform_rotations(
            transformation_matrix, rotations)

        self._conventional_rotations = conventional_rotations
        self._transformation_matrix = transformation_matrix
        self._pointgroup_symbol = pointgroup_symbol

    def _assign_character_table_data(self):
        self._character_table_data = character_tables[self._pointgroup_symbol]

    def _assign_class_labels_to_rotations(self):
        rotation_labels = []
        for rconv in self._conventional_rotations:
            label = self._assign_class_label_to_rotation(rconv)
            rotation_labels.append(label)
        self._rotation_labels = rotation_labels

    def _assign_class_label_to_rotation(self, rconv):
        """

        Input
        -----
            rconv: conventional rotation obtained using the transformation
                   matrix.
        """
        class_to_rotations = self._character_table_data["class_to_rotations"]
        for label, rotations in class_to_rotations.items():
            for rotation in rotations:
                if np.all(rconv == rotation):
                    return label
        raise ValueError("Label cannot be found.")

    def _transform_rotations(self, tmat, rotations):
        trans_rots = []
        for r in rotations:
            r_conv = similarity_transformation(np.linalg.inv(tmat), r)
            trans_rots.append(r_conv)
        return np.rint(trans_rots).astype(int)

    def get_character_table_data(self):
        return self._character_table_data

    def get_pointgroup_symbol(self):
        return self._pointgroup_symbol

    def get_conventional_rotations(self):
        return self._conventional_rotations

    def get_transformation_matrix(self):
        return self._transformation_matrix

    def get_rotation_labels(self):
        return self._rotation_labels


