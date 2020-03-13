#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Note
----
Characters are generally not integer.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from phonopy.structure.symmetry import get_pointgroup
from upho.irreps.character_tables import character_tables
from group.mathtools import similarity_transformation

__author__ = "Yuji Ikeda"


def extract_degeneracy_from_ir_label(ir_label):
    if ir_label[0] == 'E':
        degeneracy = 2
    elif ir_label[0] == 'T':
        degeneracy = 3
    else:
        degeneracy = 1
    return degeneracy


class Irreps(object):
    """

    Note
    ----
    This does not work correctly for nonsymmorphic space groups.
    """
    def __init__(self, rotations):
        """

        Parameters
        ----------
        rotations : A set of rotational symmetry operations.
            This can be a set of symmetry operations of the point group of
            a wave vector.
        """
        self._rotations = rotations
        self._run()

    def _run(self):
        self._create_conventional_rotations()
        self._assign_character_table_data()
        self._assign_class_labels_to_rotations()

        self._characters = self.assign_characters_to_rotations(
            self._rotation_labels)

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
        class_to_rotations_list = (
            self._character_table_data["class_to_rotations_list"])
        for class_to_rotations in class_to_rotations_list:
            rotation_labels = []
            for rconv in self._conventional_rotations:
                label = self._assign_class_label_to_rotation(
                    rconv, class_to_rotations)
                rotation_labels.append(label)
            if False not in rotation_labels:
                self._rotation_labels = rotation_labels
                return
        msg = "Class labels cannot be assigned to rotations.\n"
        msg += str(rotation_labels)
        raise ValueError(msg)

    def _assign_class_label_to_rotation(self, rconv, class_to_rotations):
        """

        Parameters
        ----------
        rconv :
            Conventional rotation obtained using the transformation matrix.
        class_to_rotations : Dictionary
            Keys are class labels and values are corresponding
            conventional rotations.
        """
        for label, rotations in class_to_rotations.items():
            for rotation in rotations:
                if np.all(rconv == rotation):
                    return label
        return False

    def assign_characters_to_rotations(self, rotation_labels):
        """

        Parameters
        ----------
        rotation_labels : 1d list
            Elements : Class labels.

        Returns
        -------
        characters : 2d list
            Row : IR labels
            Column : Rotations
        """
        character_table_data = self._character_table_data

        character_table = np.array(character_table_data["character_table"])
        label_order = character_table_data["rotation_labels"]

        num_rotations = len(rotation_labels)
        num_irreps = len(character_table_data["ir_labels"])

        characters = np.zeros((num_rotations, num_irreps), dtype=complex)
        for i, rotation_label in enumerate(rotation_labels):
            rotation_index = label_order.index(rotation_label)
            characters[i] = character_table[:, rotation_index]
        return characters

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

    def get_ir_labels(self):
        return self._character_table_data['ir_labels']

    def get_conventional_rotations(self):
        return self._conventional_rotations

    def get_transformation_matrix(self):
        return self._transformation_matrix

    def get_rotation_labels(self):
        return self._rotation_labels

    def get_characters(self):
        return self._characters
